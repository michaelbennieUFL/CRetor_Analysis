#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train on combined_fix6.tsv using SVM (C=1)+Rules with Qwen/Qwen3-Embedding-8B
and label baike_qa_valid.json, plus export the 1% least-certain subset.

Outputs:
  - ./output/predicted_labels.tsv
      Columns: Question, desc, Potentially_Pejorative (True/False)
  - ./output/lowest_confidence_1pct.tsv
      Same columns, but only the 1% with lowest self-confidence (via cleanlab)

Assumptions:
  - Your project provides: SemanticClassifier and HybridRuleSemanticClassifier
    (from the file where you defined them), and they behave like in your snippet.
  - combined_fix6.tsv has columns: Question, Potentially_Pejorative (string; empty/None => negative)
  - baike_qa_valid.json is either JSONL (one object per line) or a JSON array.
    We expect fields: "title" (question) and "desc" (optional).
"""

import os
import json
import math
import argparse
from typing import List, Dict, Any, Tuple
import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from cleanlab.filter import find_label_issues  # (unused, but keep if you plan to extend)
from tqdm import tqdm

# ---- Import the classifiers ----
# If these classes live in another module/file, adjust the import path accordingly.
from semantic_embedding_model import SemanticClassifier, HybridRuleSemanticClassifier

# ---------------------------
# Paths / constants
# ---------------------------
DEFAULT_TRAIN_PATH = "../../data/biasLabeling/training/combined_fix_baike_STATE_3.tsv"
DEFAULT_INPUT_JSON = "../../data/biasLabeling/testing/baike_qa_train.json"
DEFAULT_OUT_DIR = "./labeled_data"

# Use the 8B Qwen embedding model
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# Match your custom threshold to bias for precision/thresholding downstream of predict_proba
THRESHOLD = 0.5

# Lexicon used by the Rules layer
LEXICON_PATH = "../../data/biasLabeling/externalDatasets/STATE-ToxiCN/filtered_lexicon.json"
HARD_OVERRIDE = False
BOOST_FLOOR = 0.98  # used when HARD_OVERRIDE=False

def _sanitize_tsv_field(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    # Replace raw control characters with safe string markers
    s = s.replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")
    return s.strip()

def _sanitize_tsv_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(_sanitize_tsv_field)
    return out

# ---------------------------
# I/O helpers
# ---------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_training_tsv(tsv_path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(tsv_path, sep="\t")
    if "Question" not in df.columns or "Potentially_Pejorative" not in df.columns:
        raise ValueError(f"{tsv_path} must contain 'Question' and 'Potentially_Pejorative' columns.")
    X = df["Question"].astype(str).tolist()
    # Treat non-empty/non-None as positive
    y = [
        1 if str(v).strip() not in ("", "None", "none", "nan", "NaN") and v is not None else 0
        for v in df["Potentially_Pejorative"].tolist()
    ]
    return X, y

def _iter_json_objects(path: str):
    """Yield dicts from either JSONL or a JSON array file."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        stripped = head.lstrip()
        if stripped.startswith("["):
            # JSON array
            data = json.load(f)
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        else:
            # JSONL
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    continue

def load_inference_set(json_path: str) -> pd.DataFrame:
    rows = []
    for obj in _iter_json_objects(json_path):
        q = obj.get("title")
        d = obj.get("desc")
        if q is None or str(q).strip().lower() in {"", "none", "nan"}:
            continue
        q_str = str(q)
        d_str = "" if d is None else str(d)
        rows.append({
            "Question": q_str,
            "desc": d_str,
            "Question+Desc": f"{q_str} {d_str}"
        })
    if not rows:
        raise ValueError("No valid items found in input JSON.")
    return pd.DataFrame(rows, columns=["Question", "desc", "Question+Desc"])

# ---------------------------
# Lexicon helper
# ---------------------------
def load_lexicon_terms(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [t["term"] for t in data.get("terms", []) if "term" in t and str(t["term"]).strip()]

# ---------------------------
# Training: SVM (C=1)+Rules on 8B embeddings
# ---------------------------
def train_hybrid_classifier(X_train: List[str], y_train: List[int]) -> HybridRuleSemanticClassifier:
    """Builds SVM (C=1) semantic model on Qwen 8B embeddings and wraps with rules."""
    # Base semantic model
    semantic = SemanticClassifier(
        model=SVC(C=1, probability=True, random_state=42),
        embedding_model_name=MODEL_NAME,
        debug=False,
        threshold=THRESHOLD,
    )
    # Load lexicon and create hybrid
    lexicon_terms = load_lexicon_terms(LEXICON_PATH)
    hybrid = HybridRuleSemanticClassifier(
        semantic_model=semantic,
        lexicon_terms=lexicon_terms,
        hard_override=HARD_OVERRIDE,
        boost_floor=BOOST_FLOOR
    )
    # Train hybrid on full dataset (no CV for this labeling script)
    hybrid.train_model(X_train, y_train, cv=0)
    return hybrid

# ---------------------------
# Inference helpers
# ---------------------------
def predict_proba_batched(clf, texts, batch_size=512, desc="Scoring"):
    """
    Calls clf.predict_proba(texts) in mini-batches with a tqdm progress bar.
    Works with HybridRuleSemanticClassifier that exposes predict_proba(list[str]) -> (N,2).
    """
    probs_list = []
    N = len(texts)
    for i in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc=desc):
        batch = texts[i:i+batch_size]
        probs_list.append(clf.predict_proba(batch))  # shape (B, 2)
    return np.vstack(probs_list)  # (N, 2)

def predict_with_probs(clf, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    probs = predict_proba_batched(clf, texts, batch_size=2048, desc="Predicting")
    preds = (probs[:, 1] >= THRESHOLD).astype(int)
    return preds, probs

def build_outputs(df_infer: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    out = df_infer.copy()
    out["Potentially_Pejorative"] = preds.astype(bool)
    return out[["Question", "desc", "Potentially_Pejorative"]]

def lowest_confidence_1pct(pred_labels: np.ndarray, pred_probs: np.ndarray, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the 1% lowest self-confidence samples, but include only rows with self_confidence < 0.51.
    self_conf[i] = pred_probs[i, pred_labels[i]]
    """
    labels = pred_labels.astype(int)
    n = len(labels)
    if n == 0:
        return base_df.iloc[[]].assign(self_confidence=[])
    self_conf = pred_probs[np.arange(n), labels]

    # 1% of the whole set (at least 1)
    k = max(1, math.ceil(0.01 * n))
    # take the bottom 1% first
    chosen = np.argsort(self_conf)[:k]

    sub = base_df.iloc[chosen].copy()
    sub["self_confidence"] = self_conf[chosen]

    # only include self_confidence < 0.51
    sub = sub[sub["self_confidence"] < 0.51]

    # ensure boolean True/False (not strings)
    if sub["Potentially_Pejorative"].dtype != bool:
        sub["Potentially_Pejorative"] = sub["Potentially_Pejorative"].astype(bool)

    # sort by confidence ascending
    sub = sub.sort_values("self_confidence", ascending=True)

    return sub[["Question", "desc", "Potentially_Pejorative", "self_confidence"]]

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train SVM (C=1)+Rules (Qwen 8B) and predict + uncertainty on baike_qa_valid.json")
    parser.add_argument("--train_tsv", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print(f"[1/5] Loading training set: {args.train_tsv}")
    X_train, y_train = load_training_tsv(args.train_tsv)
    print(f"    Training rows: {len(X_train)}")

    print(f"[2/5] Training Hybrid (SVM C=1 + Rules, threshold={THRESHOLD}, emb={MODEL_NAME})")
    clf = train_hybrid_classifier(X_train, y_train)
    print("    Training complete.")

    print(f"[3/5] Loading inference set: {args.input_json}")
    df_infer = load_inference_set(args.input_json)
    print(f"    Inference rows: {len(df_infer)}")

    print("[4/5] Predicting labels and probabilities…")
    preds, probs = predict_with_probs(clf, df_infer["Question+Desc"].tolist())

    # Build full labeled TSV
    labeled_df = build_outputs(df_infer, preds)

    # sanitize text columns to prevent TSV breakage
    labeled_df = _sanitize_tsv_df(labeled_df, ["Question", "desc"])

    labeled_path = os.path.join(args.out_dir, "predicted_labels.tsv")
    labeled_df.to_csv(
        labeled_path,
        sep="\t",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    print(f"    Wrote: {labeled_path}")

    # --- create the low-confidence file with filtering + sanitize + stable floats ---
    low_conf_df = lowest_confidence_1pct(preds, probs, labeled_df)
    low_conf_path = os.path.join(args.out_dir, "lowest_confidence_1pct.tsv")

    low_conf_df.to_csv(
        low_conf_path,
        sep="\t",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
        float_format="%.6f",
    )
    print(f"    Wrote: {low_conf_path}")

    print("[5/5] Done.")

if __name__ == "__main__":
    main()
