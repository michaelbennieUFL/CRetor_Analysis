import json
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Iterable, Any

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
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Import your base classifier and testing utilities
from classifiers.base_classifier import BaseClassifier
from classifiers.classifer_tests import testAllParameters
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#############################
# 0. ThresholdClassifier: A Wrapper to Adjust Decision Threshold
#############################
class ThresholdClassifier(BaseEstimator):
    """
    A wrapper for classifiers to adjust the decision threshold based on predicted probabilities.
    By default, a probability threshold of 0.7 is used to decide on the positive class.
    """

    def __init__(self, classifier, threshold=0.9):
        self.classifier = classifier
        self.threshold = threshold

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        # If predict_proba is available, use it and apply the threshold
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)[:, 1]
            return (probs >= self.threshold).astype(int)
        # Otherwise, fall back on decision_function if available (though thresholding here is less direct)
        elif hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(X)
            return (scores >= 0).astype(int)
        else:
            return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.classifier, "decision_function"):
            return self.classifier.decision_function(X)
        else:
            raise AttributeError("Underlying classifier does not support decision_function")

    def get_params(self, deep=True):
        return {"classifier": self.classifier, "threshold": self.threshold}

    def set_params(self, **params):
        if "threshold" in params:
            self.threshold = params["threshold"]
            del params["threshold"]
        self.classifier.set_params(**params)
        return self


#############################
# 1. Custom Embedding Transformer
#############################
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    _model_instance = None
    _model_name = None
    global_cache = {}  # Global cache shared across all instances

    def __init__(self, model_name="sentence-transformers/LaBSE", debug=False):
        if (EmbeddingTransformer._model_instance is None) or (model_name != EmbeddingTransformer._model_name):
            if debug:
                print(f"[Singleton] Initializing the model: {model_name}")
            model = SentenceTransformer(model_name)
            try:
                model.half()  # optional: use half precision if supported
            except Exception:
                pass
            EmbeddingTransformer._model_instance = model
            EmbeddingTransformer._model_name = model_name
        else:
            if debug:
                print(f"[Singleton] Reusing the existing model: {model_name}")
        self.model_name = model_name
        self.debug = debug

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        all_embeddings = []
        for text in X:
            if text in EmbeddingTransformer.global_cache:
                concatenated = EmbeddingTransformer.global_cache[text]
            else:
                sentences = [
                    text,
                    f'Does the input "{text}" talk about race, ethnicity, gender, sex, occupations, or sexuality and hint at discrimination?',
                    f"""下面是一些示例： "那你有哪里来的样本说明贵州人素质低了？" (True)， "你喜欢美国人歧视亚裔的文化吗？你喜欢中国人歧视越南人的文化吗？你不喜欢，仅仅是受益人不是你罢了，人家没违法没犯罪，你为什么讨厌？" (True)， "hub么？我在国外都不用翻墙🐶" (False)， "你搁发廊作市场调研是吧，剩下的一个是不是扫黄的警察？说实话是不是集体蹲墙角刚被放出来？" (False)， "作者你有父母吗？来把匿名打开，你在那装你妈" (False)。请问以上示例是否涉及敏感话题或含有歧视性内容？ {text}"""
                ]
                # Compute embeddings for each sentence in the list
                sent_embeds = EmbeddingTransformer._model_instance.encode(
                    sentences, show_progress_bar=False, batch_size=32
                )
                # Concatenate the embeddings into one long vector
                concatenated = np.concatenate(sent_embeds)
                EmbeddingTransformer.global_cache[text] = concatenated
            all_embeddings.append(concatenated)
        return np.array(all_embeddings)


#############################
# 2. Semantic Classifier with Threshold Adjustment
#############################
class SemanticClassifier(BaseClassifier):
    """
    A semantic embedding-based classifier that uses the custom EmbeddingTransformer followed
    by a downstream scikit-learn model. A ThresholdClassifier wrapper is used to adjust
    the decision threshold to favor higher precision.
    """

    def __init__(self, model=None, embedding_model_name="lier007/xiaobu-embedding-v2", debug=False, threshold=0.95):
        if model is None:
            model = SVC(kernel='linear', probability=True)
        # Wrap the model with ThresholdClassifier to adjust its threshold
        wrapped_model = ThresholdClassifier(model, threshold=threshold)
        self.model_pipeline = Pipeline([
            ('embed', EmbeddingTransformer(model_name=embedding_model_name, debug=debug)),
            ('clf', wrapped_model)
        ])
        self.is_fitted = False
        self.debug = debug
        self.threshold = threshold

    def train_model(self, X: List[str], y: List[bool], cv: int = 5) -> Dict[str, float]:
        scoring = ['accuracy', 'f1_weighted', 'precision_weighted']
        if cv!=0:
            scores = cross_validate(self.model_pipeline, X, y, cv=cv, scoring=scoring)

            mean_accuracy = scores['test_accuracy'].mean()
            mean_f1 = scores['test_f1_weighted'].mean()
            mean_precision = scores['test_precision_weighted'].mean()

            print(f"[SemanticClassifier] CV={cv}")
            print(f" - Mean Accuracy: {mean_accuracy:.4f}")
            print(f" - Mean F1 Score: {mean_f1:.4f}")
            print(f" - Mean Precision: {mean_precision:.4f}")

        self.model_pipeline.fit(X, y)
        self.is_fitted = True
        print("[SemanticClassifier] Model trained on the full dataset.")

        if cv != 0:
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
        # The pipeline's predict method will use the wrapped classifier's threshold
        return self.model_pipeline.predict(text_list)

    def predict_proba(self, text_list: Union[List[str], str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SemanticClassifier model is not fitted yet.")
        if not hasattr(self.model_pipeline, "predict_proba"):
            raise AttributeError("Underlying classifier does not support predict_proba.")
        if isinstance(text_list, str):
            text_list = [text_list]
        return self.model_pipeline.predict_proba(text_list)

    # (optional) expose decision function too
    def decision_function(self, text_list: Union[List[str], str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SemanticClassifier model is not fitted yet.")
        if not hasattr(self.model_pipeline, "decision_function"):
            raise AttributeError("Underlying classifier does not support decision_function.")
        if isinstance(text_list, str):
            text_list = [text_list]
        return self.model_pipeline.decision_function(text_list)


class HybridRuleSemanticClassifier(BaseEstimator, ClassifierMixin):
    """
    Rules-first + Semantic fallback.
    - If any lexicon term appears literally in the text: predict True (and proba=1.0 for positive).
    - Otherwise, use the underlying (already defined) SemanticClassifier pipeline.
    Options:
      hard_override=True  -> force True / 1.0 when matched
      hard_override=False -> only boost: proba = max(model_proba, boost_floor)
    """

    def __init__(self,
                 semantic_model,                 # e.g., an instance of your SemanticClassifier
                 lexicon_terms: Iterable[str],   # iterable of literal terms/phrases
                 hard_override: bool = True,
                 boost_floor: float = 0.95):
        self.semantic_model = semantic_model
        self.lexicon_terms = set([t.strip() for t in lexicon_terms if str(t).strip()])  # de-dup & clean
        self.hard_override = hard_override
        self.boost_floor = float(boost_floor)
        self.is_fitted_ = False

    # ---- utils ----
    def _contains_any_term(self, text: str) -> bool:
        s = text or ""
        # literal substring check (not regex!)
        return any(term in s for term in self.lexicon_terms)

    def _ensure_list(self, X: Union[str, List[str]]) -> List[str]:
        return [X] if isinstance(X, str) else list(X)

    # ---- sklearn API ----

    def train_model(self, X: List[str], y: List[int], cv: int = 5, random_state: int = 42):
        """
        Cross-validate the FULL hybrid (rules + semantic) so reported scores match
        what you'll see at inference time. Then fit on the full data.
        Returns: dict with mean_accuracy, mean_f1, mean_precision (weighted).
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
                # (no CV inside to avoid nested CV; we just fit)
                self.semantic_model.model_pipeline.fit(X_train, y_train)
                self.semantic_model.is_fitted = True
                self.is_fitted_ = True

                # Predict with rules + semantic on the val set
                y_pred = self.predict(X_val)

                # Metrics (weighted)
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
            # Not enough folds or single-class data; skip CV
            mean_accuracy = mean_f1 = mean_precision = float("nan")

        # Fit once more on FULL data so the model is ready for use
        self.semantic_model.model_pipeline.fit(X, y)
        self.semantic_model.is_fitted = True
        self.is_fitted_ = True

        return {
            "mean_accuracy": mean_accuracy,
            "mean_f1": mean_f1,
            "mean_precision": mean_precision,
        }

    def fit(self, X: List[str], y: List[int]):
        # Fit underlying semantic model
        self.semantic_model.train_model(X, y, cv=0)  # disable CV here; you can do it outside as needed
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[str, List[str]]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("HybridRuleSemanticClassifier is not fitted yet. Call fit() first.")
        X_list = self._ensure_list(X)

        # rules-first decisions
        rule_hits = np.array([self._contains_any_term(x) for x in X_list], dtype=bool)

        # model fallback for non-hits
        preds = np.zeros(len(X_list), dtype=int)
        if rule_hits.any():
            preds[rule_hits] = 1  # force positive

        if (~rule_hits).any():
            model_preds = self.semantic_model.predict([X_list[i] for i in np.where(~rule_hits)[0]])
            preds[~rule_hits] = np.array(model_preds, dtype=int)

        return preds

    def predict_proba(self, X: Union[str, List[str]]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("HybridRuleSemanticClassifier is not fitted yet. Call fit() first.")
        X_list = self._ensure_list(X)

        # base probabilities from the semantic model
        base_proba = self.semantic_model.predict_proba(X_list)  # shape: (n, 2) assuming [neg, pos]

        # apply rule layer
        out = base_proba.copy()
        for i, text in enumerate(X_list):
            if self._contains_any_term(text):
                if self.hard_override:
                    out[i, :] = np.array([0.0, 1.0])
                else:
                    # only boost positive to at least boost_floor, keep calibration-ish
                    out[i, 1] = max(out[i, 1], self.boost_floor)
                    out[i, 0] = 1.0 - out[i, 1]
        return out

    # optional: decision_function passthrough (only for convenience)
    def decision_function(self, X: Union[str, List[str]]) -> np.ndarray:
        # Define a simple mapping: use logit of predict_proba
        proba = self.predict_proba(X)[:, 1]
        # avoid infs
        eps = 1e-12
        proba = np.clip(proba, eps, 1 - eps)
        return np.log(proba / (1 - proba))

    def get_params(self, deep=True) -> Dict[str, Any]:
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


#############################
# 3. Define a Set of Semantic Classifiers with Adjusted Thresholds
#############################
random_state = 10
model_name = "thenlper/gte-base-zh"
custom_threshold = 0.5  # Adjust this value to favor higher precision


with open("../../data/biasLabeling/externalDatasets/STATE-ToxiCN/filtered_lexicon.json", "r", encoding="utf-8") as f:
    lexicon_json = json.load(f)
lexicon_terms = [t["term"] for t in lexicon_json.get("terms", [])]

semantic_classifiers = {
    
        "SVM (C=0.1)": SemanticClassifier(
        model=SVC(C=0.1, probability=True, random_state=42),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),
    "Logistic Regression": SemanticClassifier(
        model=LogisticRegression(max_iter=10000),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),
    "Logistic Regression+Rules": HybridRuleSemanticClassifier(
        semantic_model=SemanticClassifier(
            model=LogisticRegression(max_iter=10000),
            embedding_model_name=model_name,
            debug=False,
            threshold=custom_threshold
        ),
        lexicon_terms=lexicon_terms,
        hard_override=False,
        boost_floor=0.98  # used only when hard_override=False
    ),
    "SVM (C=1)+Rules": HybridRuleSemanticClassifier(
        semantic_model=SemanticClassifier(
            model=SVC(C=1, probability=True, random_state=42),
            embedding_model_name=model_name,
            debug=False,
            threshold=custom_threshold
        ),
        lexicon_terms=lexicon_terms,
        hard_override=False,
        boost_floor=0.98  # used only when hard_override=False
    ),

    "SVM (Linear Kernel)": SemanticClassifier(
        model=SVC(kernel='linear', probability=True),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),
    "SVM (Sigmoid Kernel)": SemanticClassifier(
        model=SVC(kernel='sigmoid', probability=True),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),

    "Random Forest": SemanticClassifier(
        model=RandomForestClassifier(n_estimators=100, random_state=random_state),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),
    "Nearest Neighbors": SemanticClassifier(
        model=KNeighborsClassifier(10),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),
    "Decision Tree": SemanticClassifier(
        model=DecisionTreeClassifier(max_depth=10, random_state=random_state),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),
    "AdaBoost": SemanticClassifier(
        model=AdaBoostClassifier(random_state=random_state),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),

}


#############################
# 4. Main: Load Data and Test the Classifiers
#############################
if __name__ == "__main__":
    # Load the combined TSV dataset
    data_df = pd.read_csv("../../data/biasLabeling/training/combined_fix_baike_STATE_3_ChineseHarm.tsv", sep="\t")
    # Assuming "Question" column has the text and "Potentially_Pejorative" is a label that is "Potentially" if True and "None" or empty if False.
    X = data_df["Question"].tolist()
    # Convert target: non-empty (and not "None") -> True, else False.
    y = [True if str(val).strip() not in ("", "None", "none", "nan", None) else False
         for val in data_df["Potentially_Pejorative"].tolist()]

    cv = 10

    print("\n====== Testing Semantic Classifiers ======")
    # testAllParameters is assumed to run cross-validation tests for each classifier.
    testAllParameters(semantic_classifiers, (X, y), cv=cv, top_n=3)
