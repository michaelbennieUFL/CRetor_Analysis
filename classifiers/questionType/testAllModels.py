import os
import time
import threading
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------
# Heartbeat: proves the run is alive
# --------------------------
def start_heartbeat(tag="TestAllModels", every_sec=60):
    def _hb():
        while True:
            print(f"[HEARTBEAT] {tag} still running…")
            time.sleep(every_sec)
    t = threading.Thread(target=_hb, daemon=True)
    t.start()


# --------------------------
# Threshold wrapper
# --------------------------
class ThresholdClassifier(BaseEstimator):
    def __init__(self, classifier, threshold=0.5):
        self.classifier = classifier
        self.threshold = threshold

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)[:, 1]
            return (probs >= self.threshold).astype(int)
        elif hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(X)
            return (scores >= 0).astype(int)
        else:
            return self.classifier.predict(X)

    def predict_proba(self, X):
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X)
        raise AttributeError("Underlying classifier does not support predict_proba")

    def get_params(self, deep=True):
        return {"classifier": self.classifier, "threshold": self.threshold}

    def set_params(self, **params):
        if "threshold" in params:
            self.threshold = params.pop("threshold")
        self.classifier.set_params(**params)
        return self


# --------------------------
# Embedding transformer (singleton per model name)
# --------------------------
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    _model_instance = None
    _model_name = None
    global_cache: Dict[str, np.ndarray] = {}

    def __init__(self, model_name="thenlper/gte-base-zh", debug=False, encode_batch_size=8):
        self.model_name = model_name
        self.debug = debug
        self.encode_batch_size = int(encode_batch_size)

        if (EmbeddingTransformer._model_instance is None) or (EmbeddingTransformer._model_name != model_name):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.debug:
                print(f"[EmbeddingTransformer] Loading model: {model_name} on device={device}")
            model = SentenceTransformer(model_name, device=device)
            # half precision only if on GPU
            if device == "cuda":
                try:
                    model.half()
                except Exception:
                    pass
            EmbeddingTransformer._model_instance = model
            EmbeddingTransformer._model_name = model_name
        elif self.debug:
            print(f"[EmbeddingTransformer] Reusing model: {model_name}")

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        out = []
        # one tqdm bar across the dataset (your inner list per sample is small, so keep show_progress_bar=False)
        for text in tqdm(X, desc=f"Encoding ({self.model_name})", mininterval=1.0):
            if text in EmbeddingTransformer.global_cache:
                out.append(EmbeddingTransformer.global_cache[text])
                continue

            sentences = [
                text,
                f'Does the input "{text}" talk about race, ethnicity, gender, sex, occupations, or sexuality and hint at discrimination?',
                (
                    '下面是一些示例： "那你有哪里来的样本说明贵州人素质低了？" (True)， '
                    '"你喜欢美国人歧视亚裔的文化吗？你喜欢中国人歧视越南人的文化吗？你不喜欢，仅仅是受益人不是你罢了，人家没违法没犯罪，你为什么讨厌？" (True)， '
                    '"hub么？我在国外都不用翻墙🐶" (False)， '
                    '"你搁发廊作市场调研是吧，剩下的一个是不是扫黄的警察？说实话是不是集体蹲墙角刚被放出来？" (False)， '
                    '"作者你有父母吗？来把匿名打开，你在那装你妈" (False)。'
                    f"请问以上示例是否涉及敏感话题或含有歧视性内容？ {text}"
                )
            ]
            # Keep batch size small so progress updates and VRAM stays reasonable
            embeds = EmbeddingTransformer._model_instance.encode(
                sentences, show_progress_bar=False, batch_size=self.encode_batch_size
            )
            vec = np.concatenate(embeds)
            EmbeddingTransformer.global_cache[text] = vec
            out.append(vec)

        return np.vstack(out)


# --------------------------
# Build classifiers dict for a given embedding model
# --------------------------
def build_classifiers(embedding_model_name: str, threshold: float = 0.5) -> Dict[str, Pipeline]:
    def pipe(est):
        return Pipeline([
            ('embed', EmbeddingTransformer(model_name=embedding_model_name, debug=True, encode_batch_size=8)),
            ('clf', ThresholdClassifier(est, threshold=threshold))
        ])
    random_state = 1
    return {
        "LogReg":        pipe(LogisticRegression(max_iter=10000)),
        "SVM_C1":        pipe(SVC(C=1, probability=True, random_state=42)),
        "SVM_Linear":    pipe(SVC(kernel='linear', probability=True)),
        "SVM_Sigmoid":   pipe(SVC(kernel='sigmoid', probability=True)),
        "RandomForest":  pipe(RandomForestClassifier(n_estimators=100, random_state=random_state)),
        "KNN_k10":       pipe(KNeighborsClassifier(10)),
        "DecisionTree":  pipe(DecisionTreeClassifier(max_depth=10, random_state=random_state)),
        "AdaBoost":      pipe(AdaBoostClassifier(random_state=random_state)),
    }


# --------------------------
# Evaluate a dict of pipelines with cross-validate
# --------------------------
def evaluate_models(models: Dict[str, Pipeline], X: List[str], y: List[int], cv: int = 5
                   ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    acc, f1w, precw = {}, {}, {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for name, pipe in models.items():
        try:
            print(f"[INFO] Evaluating {name} with {cv}-fold CV…")
            scores = cross_validate(
                pipe, X, y, cv=skf,
                scoring=['accuracy', 'f1_weighted', 'precision_weighted'],
                n_jobs=1,                # keep 1 to avoid multiple encode() in parallel
                return_train_score=False,
                verbose=1,               # prints fold-level progress
            )
            acc[name]   = float(np.mean(scores['test_accuracy']))
            f1w[name]   = float(np.mean(scores['test_f1_weighted']))
            precw[name] = float(np.mean(scores['test_precision_weighted']))
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")
            acc[name], f1w[name], precw[name] = np.nan, np.nan, np.nan
    return acc, f1w, precw


# --------------------------
# Upsert a row into a TSV (row = embedding model, columns = classifier names)
# --------------------------
def upsert_row(tsv_path: str, row_name: str, values: Dict[str, float]) -> None:
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    else:
        df = pd.DataFrame()

    for col in values.keys():
        if col not in df.columns:
            df[col] = np.nan

    df.loc[row_name, list(values.keys())] = list(values.values())
    df = df.reindex(sorted(df.columns), axis=1)  # stable column order
    df.to_csv(tsv_path, sep="\t")


# --------------------------
# Main
# --------------------------
def main():
    start_heartbeat("TestAllModels", every_sec=60*5)

    # ---- config you can tweak ----
    data_path = "../../data/biasLabeling/training/combined_fix_baike_STATE_3_ChineseHarm_ToxiCN_MM_ChineseHarm.tsv"
    text_col = "Question"
    label_col = "Potentially_Pejorative"
    cv = 10
    decision_threshold = 0.5

    # Embedding models to test (smaller ones first)
    embedding_models = [
        "Qwen/Qwen3-Embedding-0.6B",
        "thenlper/gte-base-zh",
        "lier007/xiaobu-embedding-v2",
        "intfloat/multilingual-e5-large-instruct",
        "richinfoai/ritrieve_zh_v1",
        "Qwen/Qwen3-Embedding-4B",
        "sensenova/piccolo-large-zh-v2",
        "Classical/Yinka",
        "iampanda/zpoint_large_embedding_zh"
    ]

    # Output TSVs
    ACC_PATH  = "results_accuracy.tsv"
    F1_PATH   = "results_f1.tsv"
    PREC_PATH = "results_precision.tsv"

    # ---- Hugging Face caches (override in your shell to /blue if you want) ----
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/hf_datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/hf_transformers"))

    # ---- load data ----
    df = pd.read_csv(data_path, sep="\t")
    X = df[text_col].astype(str).tolist()
    y = [
        1 if str(v).strip() not in ("", "None", "none", "nan", "NaN") and v is not None else 0
        for v in df[label_col].tolist()
    ]

    # ---- loop over embeddings; update TSVs each round ----
    for emb in embedding_models:
        print(f"\n=== Embedding: {emb} ===")
        models = build_classifiers(emb, threshold=decision_threshold)
        acc, f1w, precw = evaluate_models(models, X, y, cv=cv)

        # Persist results after each embedding (so you keep progress)
        upsert_row(ACC_PATH,  emb, acc)
        upsert_row(F1_PATH,   emb, f1w)
        upsert_row(PREC_PATH, emb, precw)

        print(f"Saved rows for {emb} to:")
        print(f"  - {ACC_PATH}")
        print(f"  - {F1_PATH}")
        print(f"  - {PREC_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Partial results (TSVs) are preserved.")
