#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic embedding-based classifiers using Kingsoft-LLM/QZhou-Embedding-Zh
with optional rule-based augmentation and adjustable decision thresholds.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Iterable, Any, Optional

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

# Import your base classifier and testing utilities
from classifiers.base_classifier import BaseClassifier
from classifiers.classifer_tests import testAllParameters

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
#  USER-CONFIGURABLE VARIABLES
# =============================================================================

# Path to the combined training TSV file
DATA_TSV_PATH = "../../data/biasLabeling/training/combined_fix_baike_STATE_3_ChineseHarm_ToxiCN_MM_ChineseHarm.tsv"

# Column names in the TSV
TEXT_COLUMN = "Question"
LABEL_COLUMN = "Potentially_Pejorative"

# Path to lexicon JSON for rule-based classifier
LEXICON_PATH = "../../data/biasLabeling/externalDatasets/STATE-ToxiCN/filtered_lexicon.json"

# Hugging Face sentence-transformers model name
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# MRL / embedding dimension for QZhou-Embedding-Zh (one of 128, 256, 512, 768, 1024, 1280, 1536, 1792)
EMBEDDING_DIM = 512

# Decision threshold for positive class (applied on predict_proba[:, 1])
CUSTOM_THRESHOLD = 0.5

# Random state for classical ML models
RANDOM_STATE = 10

# Cross-validation folds
CV_FOLDS = 4

# How many top models to print in testAllParameters
TOP_N_MODELS = 3


# =============================================================================
# 0. ThresholdClassifier: Wrapper to Adjust Decision Threshold
# =============================================================================

class ThresholdClassifier(BaseEstimator):
    """
    A wrapper for classifiers to adjust the decision threshold based on predicted probabilities.
    """

    def __init__(self, classifier, threshold: float = 0.9):
        self.classifier = classifier
        self.threshold = float(threshold)

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        # If predict_proba is available, use it and apply the threshold
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)[:, 1]
            return (probs >= self.threshold).astype(int)
        # Otherwise, fall back on decision_function if available
        elif hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(X)
            return (scores >= 0).astype(int)
        else:
            return self.classifier.predict(X)

    def predict_proba(self, X):
        if not hasattr(self.classifier, "predict_proba"):
            raise AttributeError("Underlying classifier does not support predict_proba")
        return self.classifier.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.classifier, "decision_function"):
            return self.classifier.decision_function(X)
        else:
            raise AttributeError("Underlying classifier does not support decision_function")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"classifier": self.classifier, "threshold": self.threshold}

    def set_params(self, **params):
        if "threshold" in params:
            self.threshold = params["threshold"]
            del params["threshold"]
        self.classifier.set_params(**params)
        return self


# =============================================================================
# 1. Custom Embedding Transformer
# =============================================================================

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    A reusable embedding transformer built on top of SentenceTransformer.

    - Uses a singleton SentenceTransformer instance (per model_name).
    - Supports QZhou-Embedding-Zh with trust_remote_code and PromptEOL template.
    - Supports cropping to a desired embedding dimension (EMBEDDING_DIM).
    - Caches embeddings across calls.
    """

    _model_instance: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None
    global_cache: Dict[Any, np.ndarray] = {}  # shared across instances

    def __init__(self,
                 model_name: str = "sentence-transformers/LaBSE",
                 debug: bool = False,
                 output_dim: Optional[int] = None):
        if (EmbeddingTransformer._model_instance is None) or (model_name != EmbeddingTransformer._model_name):
            if debug:
                print(f"[Singleton] Initializing the embedding model: {model_name}")

            # Special handling for QZhou-Embedding-Zh
            if model_name == "Kingsoft-LLM/QZhou-Embedding-Zh":
                model = SentenceTransformer(
                    "Kingsoft-LLM/QZhou-Embedding-Zh",
                    model_kwargs={"device_map": "cuda", "trust_remote_code": True},
                    tokenizer_kwargs={"padding_side": "left", "trust_remote_code": True},
                    trust_remote_code=True
                )
            else:
                # fallback for other models
                model = SentenceTransformer(model_name)

            # Optional: FP16 if supported
            try:
                model.half()
            except Exception:
                pass

            EmbeddingTransformer._model_instance = model
            EmbeddingTransformer._model_name = model_name
        else:
            if debug:
                print(f"[Singleton] Reusing existing embedding model: {model_name}")

        self.model_name = model_name
        self.debug = debug
        self.output_dim = output_dim  # e.g. 256

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _prompteol_input(text: str) -> str:
        """
        PromptEOL-style input as described in the QZhou-Embedding-Zh model card.
        """
        return f'This sentence: <|im_start|>“{text}” means in one word: “'

    def transform(self, X: List[str]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        model = EmbeddingTransformer._model_instance

        for text in X:
            # Include model_name and output_dim in the cache key to avoid shape mismatches
            cache_key = (self.model_name, self.output_dim, text)

            if cache_key in EmbeddingTransformer.global_cache:
                emb_vec = EmbeddingTransformer.global_cache[cache_key]
            else:
                if self.model_name == "Kingsoft-LLM/QZhou-Embedding-Zh":
                    # QZhou: single PromptEOL-style input, MRL cropping
                    inp = self._prompteol_input(text)
                    emb = model.encode(
                        [inp],
                        show_progress_bar=False,
                        batch_size=32,
                        normalize_embeddings=False
                    )[0]  # shape: (1792,)

                    if self.output_dim is not None:
                        emb = emb[: self.output_dim]

                    emb_vec = emb  # 1D vector
                else:
                    # Default behavior for other models: 3 prompts + concatenation
                    sentences = [
                        text,
                        f'Does the input "{text}" talk about race, ethnicity, gender, sex, occupations, or sexuality and hint at discrimination?',
                        (
                            '下面是一些示例： "那你有哪里来的样本说明贵州人素质低了？" (True)， '
                            '"你喜欢美国人歧视亚裔的文化吗？你喜欢中国人歧视越南人的文化吗？你不喜欢，'
                            '仅仅是受益人不是你罢了，人家没违法没犯罪，你为什么讨厌？" (True)， '
                            '"hub么？我在国外都不用翻墙🐶" (False)， '
                            '"你搁发廊作市场调研是吧，剩下的一个是不是扫黄的警察？说实话是不是集体蹲墙角刚被放出来？" (False)， '
                            '"作者你有父母吗？来把匿名打开，你在那装你妈" (False)。'
                            f"请问以上示例是否涉及敏感话题或含有歧视性内容？ {text}"
                        )
                    ]

                    sent_embeds = model.encode(
                        sentences,
                        show_progress_bar=False,
                        batch_size=32
                    )  # shape: (3, dim)

                    if self.output_dim is not None:
                        sent_embeds = sent_embeds[:, : self.output_dim]

                    emb_vec = np.concatenate(sent_embeds, axis=-1)  # shape: (3 * dim,)

                EmbeddingTransformer.global_cache[cache_key] = emb_vec

            all_embeddings.append(emb_vec)

        return np.array(all_embeddings)


# =============================================================================
# 2. Semantic Classifier with Threshold Adjustment
# =============================================================================

class SemanticClassifier(BaseClassifier):
    """
    A semantic embedding-based classifier that uses EmbeddingTransformer
    followed by a downstream scikit-learn model.
    """

    def __init__(self,
                 model=None,
                 embedding_model_name: str = EMBEDDING_MODEL_NAME,
                 embedding_dim: Optional[int] = EMBEDDING_DIM,
                 debug: bool = False,
                 threshold: float = CUSTOM_THRESHOLD):
        if model is None:
            model = SVC(kernel='linear', probability=True)

        wrapped_model = ThresholdClassifier(model, threshold=threshold)

        self.model_pipeline = Pipeline([
            ('embed', EmbeddingTransformer(
                model_name=embedding_model_name,
                debug=debug,
                output_dim=embedding_dim,
            )),
            ('clf', wrapped_model)
        ])

        self.is_fitted = False
        self.debug = debug
        self.threshold = threshold
        self.embedding_dim = embedding_dim

    def train_model(self, X: List[str], y: List[bool], cv: int = 5) -> Dict[str, float]:
        scoring = ['accuracy', 'f1_weighted', 'precision_weighted']

        if cv and cv > 1:
            scores = cross_validate(self.model_pipeline, X, y, cv=cv, scoring=scoring)

            mean_accuracy = scores['test_accuracy'].mean()
            mean_f1 = scores['test_f1_weighted'].mean()
            mean_precision = scores['test_precision_weighted'].mean()

            print(f"[SemanticClassifier] CV={cv}")
            print(f" - Mean Accuracy: {mean_accuracy:.4f}")
            print(f" - Mean F1 Score: {mean_f1:.4f}")
            print(f" - Mean Precision: {mean_precision:.4f}")
        else:
            mean_accuracy = mean_f1 = mean_precision = float("nan")

        self.model_pipeline.fit(X, y)
        self.is_fitted = True
        print("[SemanticClassifier] Model trained on the full dataset.")

        return {
            "mean_accuracy": mean_accuracy,
            "mean_f1": mean_f1,
            "mean_precision": mean_precision
        }

    def predict(self, text_list: Union[List[str], str]) -> List[bool]:
        if not self.is_fitted:
            raise RuntimeError("SemanticClassifier model is not fitted yet.")
        if isinstance(text_list, str):
            text_list = [text_list]
        return self.model_pipeline.predict(text_list)

    def predict_proba(self, text_list: Union[List[str], str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SemanticClassifier model is not fitted yet.")
        if isinstance(text_list, str):
            text_list = [text_list]
        if not hasattr(self.model_pipeline, "predict_proba"):
            raise AttributeError("Underlying classifier does not support predict_proba.")
        return self.model_pipeline.predict_proba(text_list)

    def decision_function(self, text_list: Union[List[str], str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SemanticClassifier model is not fitted yet.")
        if isinstance(text_list, str):
            text_list = [text_list]
        if not hasattr(self.model_pipeline, "decision_function"):
            raise AttributeError("Underlying classifier does not support decision_function.")
        return self.model_pipeline.decision_function(text_list)


# =============================================================================
# 3. Hybrid Rule + Semantic Classifier
# =============================================================================

class HybridRuleSemanticClassifier(BaseEstimator, ClassifierMixin):
    """
    Rules-first + Semantic fallback.

    - If any lexicon term appears literally in the text: predict True or boost proba.
    - Otherwise, use the underlying SemanticClassifier pipeline.
    """

    def __init__(self,
                 semantic_model: SemanticClassifier,
                 lexicon_terms: Iterable[str],
                 hard_override: bool = True,
                 boost_floor: float = 0.95):
        self.semantic_model = semantic_model
        self.lexicon_terms = set([t.strip() for t in lexicon_terms if str(t).strip()])
        self.hard_override = bool(hard_override)
        self.boost_floor = float(boost_floor)
        self.is_fitted_ = False

    # ---- utils ----
    def _contains_any_term(self, text: str) -> bool:
        s = text or ""
        return any(term in s for term in self.lexicon_terms)

    def _ensure_list(self, X: Union[str, List[str]]) -> List[str]:
        return [X] if isinstance(X, str) else list(X)

    # ---- training ----

    def train_model(self, X: List[str], y: List[int], cv: int = 5, random_state: int = RANDOM_STATE):
        """
        Cross-validate the full hybrid (rules + semantic), then fit on full data.
        """
        X = list(X)
        y = np.asarray(y, dtype=int)

        if cv and cv > 1 and len(np.unique(y)) > 1:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

            accs, f1s, precs = [], [], []
            for train_idx, val_idx in skf.split(X, y):
                X_train = [X[i] for i in train_idx]
                y_train = y[train_idx]
                X_val = [X[i] for i in val_idx]
                y_val = y[val_idx]

                # Fit the underlying semantic model on THIS fold's train set
                self.semantic_model.model_pipeline.fit(X_train, y_train)
                self.semantic_model.is_fitted = True
                self.is_fitted_ = True

                # Predict with rules + semantic on the val set
                y_pred = self.predict(X_val)

                accs.append(accuracy_score(y_val, y_pred))
                f1s.append(f1_score(y_val, y_pred, average="weighted", zero_division=0))
                precs.append(precision_score(y_val, y_pred, average="weighted", zero_division=0))

            mean_accuracy = float(np.mean(accs))
            mean_f1 = float(np.mean(f1s))
            mean_precision = float(np.mean(precs))

            print(f"[HybridRuleSemanticClassifier] CV={cv}")
            print(f" - Mean Accuracy: {mean_accuracy:.4f}")
            print(f" - Mean F1 Score: {mean_f1:.4f}")
            print(f" - Mean Precision: {mean_precision:.4f}")
        else:
            mean_accuracy = mean_f1 = mean_precision = float("nan")

        # Fit once more on full data
        self.semantic_model.model_pipeline.fit(X, y)
        self.semantic_model.is_fitted = True
        self.is_fitted_ = True

        return {
            "mean_accuracy": mean_accuracy,
            "mean_f1": mean_f1,
            "mean_precision": mean_precision,
        }

    def fit(self, X: List[str], y: List[int]):
        # Fit underlying semantic model on full data
        self.semantic_model.train_model(X, y, cv=0)
        self.is_fitted_ = True
        return self

    # ---- inference ----

    def predict(self, X: Union[str, List[str]]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("HybridRuleSemanticClassifier is not fitted yet. Call fit() first.")
        X_list = self._ensure_list(X)

        rule_hits = np.array([self._contains_any_term(x) for x in X_list], dtype=bool)

        preds = np.zeros(len(X_list), dtype=int)
        if rule_hits.any():
            preds[rule_hits] = 1  # rule-based positive

        if (~rule_hits).any():
            model_preds = self.semantic_model.predict([X_list[i] for i in np.where(~rule_hits)[0]])
            preds[~rule_hits] = np.array(model_preds, dtype=int)

        return preds

    def predict_proba(self, X: Union[str, List[str]]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("HybridRuleSemanticClassifier is not fitted yet. Call fit() first.")
        X_list = self._ensure_list(X)

        base_proba = self.semantic_model.predict_proba(X_list)  # shape: (n, 2)

        out = base_proba.copy()
        for i, text in enumerate(X_list):
            if self._contains_any_term(text):
                if self.hard_override:
                    out[i, :] = np.array([0.0, 1.0])
                else:
                    out[i, 1] = max(out[i, 1], self.boost_floor)
                    out[i, 0] = 1.0 - out[i, 1]
        return out

    def decision_function(self, X: Union[str, List[str]]) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        eps = 1e-12
        proba = np.clip(proba, eps, 1 - eps)
        return np.log(proba / (1 - proba))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "semantic_model": self.semantic_model,
            "lexicon_terms": list(self.lexicon_terms),
            "hard_override": self.hard_override,
            "boost_floor": self.boost_floor,
        }

    def set_params(self, **params):
        if "semantic_model" in params:
            self.semantic_model = params["semantic_model"]
        if "lexicon_terms" in params:
            self.lexicon_terms = set([t.strip() for t in params["lexicon_terms"] if str(t).strip()])
        if "hard_override" in params:
            self.hard_override = bool(params["hard_override"])
        if "boost_floor" in params:
            self.boost_floor = float(params["boost_floor"])
        return self


# =============================================================================
# 4. Define a Set of Semantic Classifiers
# =============================================================================

def build_semantic_classifiers(embedding_model_name: str,
                               embedding_dim: Optional[int],
                               lexicon_terms: Iterable[str]) -> Dict[str, Any]:
    """
    Create a dictionary of semantic and hybrid classifiers to be evaluated.
    """
    semantic_classifiers: Dict[str, Any] = {
        "Logistic Regression": SemanticClassifier(
            model=LogisticRegression(max_iter=10000),
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            debug=False,
            threshold=CUSTOM_THRESHOLD
        ),
        "Logistic Regression+Rules": HybridRuleSemanticClassifier(
            semantic_model=SemanticClassifier(
                model=LogisticRegression(max_iter=10000),
                embedding_model_name=embedding_model_name,
                embedding_dim=embedding_dim,
                debug=False,
                threshold=CUSTOM_THRESHOLD
            ),
            lexicon_terms=lexicon_terms,
            hard_override=False,
            boost_floor=0.98
        ),
        "SVM (C=0.1)": SemanticClassifier(
            model=SVC(C=0.1, probability=True, random_state=42),
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            debug=False,
            threshold=CUSTOM_THRESHOLD
        ),
        "SVM (C=1)+Rules": HybridRuleSemanticClassifier(
            semantic_model=SemanticClassifier(
                model=SVC(C=1, probability=True, random_state=42),
                embedding_model_name=embedding_model_name,
                embedding_dim=embedding_dim,
                debug=False,
                threshold=CUSTOM_THRESHOLD
            ),
            lexicon_terms=lexicon_terms,
            hard_override=False,
            boost_floor=0.98
        ),
        "SVM (Linear Kernel)": SemanticClassifier(
            model=SVC(kernel='linear', probability=True),
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            debug=False,
            threshold=CUSTOM_THRESHOLD
        ),
        "SVM (Sigmoid Kernel)": SemanticClassifier(
            model=SVC(kernel='sigmoid', probability=True),
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            debug=False,
            threshold=CUSTOM_THRESHOLD
        ),
        "Random Forest": SemanticClassifier(
            model=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            debug=False,
            threshold=CUSTOM_THRESHOLD
        ),
        "Nearest Neighbors": SemanticClassifier(
            model=KNeighborsClassifier(10),
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            debug=False,
            threshold=CUSTOM_THRESHOLD
        ),
        "Decision Tree": SemanticClassifier(
            model=DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            debug=False,
            threshold=CUSTOM_THRESHOLD
        ),
        "AdaBoost": SemanticClassifier(
            model=AdaBoostClassifier(random_state=RANDOM_STATE),
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            debug=False,
            threshold=CUSTOM_THRESHOLD
        ),
    }
    return semantic_classifiers


# =============================================================================
# 5. Main: Load Data and Test the Classifiers
# =============================================================================

if __name__ == "__main__":
    # Load the combined TSV dataset
    data_df = pd.read_csv(DATA_TSV_PATH, sep="\t")

    # Extract text and labels
    X = data_df[TEXT_COLUMN].astype(str).tolist()
    raw_labels = data_df[LABEL_COLUMN].tolist()
    # Convert labels: non-empty and not "None"/"none"/"nan" -> True, else False
    y = [
        True if str(val).strip() not in ("", "None", "none", "nan", None) else False
        for val in raw_labels
    ]

    # Load lexicon terms
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        lexicon_json = json.load(f)
    lexicon_terms = [t["term"] for t in lexicon_json.get("terms", [])]

    # Build classifiers
    classifiers = build_semantic_classifiers(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        lexicon_terms=lexicon_terms
    )

    print("\n====== Testing Semantic Classifiers ======")
    # testAllParameters is expected to run CV tests for each classifier
    testAllParameters(classifiers, (X, y), cv=CV_FOLDS, top_n=TOP_N_MODELS)
