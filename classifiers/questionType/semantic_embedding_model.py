import json
import numpy as np
import pandas as pd
from typing import List, Union, Dict

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
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

    def __init__(self, model=None, embedding_model_name="thenlper/gte-base-zh", debug=False, threshold=0.95):
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


#############################
# 3. Define a Set of Semantic Classifiers with Adjusted Thresholds
#############################
random_state = 1
model_name = "thenlper/gte-base-zh"
custom_threshold = 0.5  # Adjust this value to favor higher precision

semantic_classifiers = {

    "Logistic Regression": SemanticClassifier(
        model=LogisticRegression(max_iter=10000),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),
    "SVM (C=1)": SemanticClassifier(
        model=SVC(C=1, probability=True, random_state=42),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),
    "Gaussian Process": SemanticClassifier(
        model=GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
    ),
    "MLP (Neural Net)": SemanticClassifier(
        model=MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        embedding_model_name=model_name,
        debug=False,
        threshold=custom_threshold
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
    data_df = pd.read_csv("../../data/biasLabeling/training/combined.tsv", sep="\t")
    # Assuming "Question" column has the text and "Potentially_Pejorative" is a label that is "Potentially" if True and "None" or empty if False.
    X = data_df["Question"].tolist()
    # Convert target: non-empty (and not "None") -> True, else False.
    y = [True if str(val).strip() not in ("", "None", "none", "nan", None) else False
         for val in data_df["Potentially_Pejorative"].tolist()]

    cv = 20

    print("\n====== Testing Semantic Classifiers ======")
    # testAllParameters is assumed to run cross-validation tests for each classifier.
    testAllParameters(semantic_classifiers, (X, y), cv=cv, top_n=3)
