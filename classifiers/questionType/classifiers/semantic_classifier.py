from typing import List, Union
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from classifiers.base_classifier import BaseClassifier


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that uses a sentence-transformers model to
    generate embeddings for the input texts (singleton pattern).
    """
    _model_instance = None
    _model_name = None

    def __init__(self, model_name="thenlper/gte-base-zh", debug=False):
        if (EmbeddingTransformer._model_instance is None) or \
           (model_name != EmbeddingTransformer._model_name):
            if debug:
                print(f"[Singleton] Initializing the model: {model_name}")
            model = SentenceTransformer(model_name)
            # optional: use half precision if GPU supports
            try:
                model.half()
            except:
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

    def transform(self, X):
        embeddings = EmbeddingTransformer._model_instance.encode(
            X, show_progress_bar=False, batch_size=32
        )
        return embeddings


class SemanticClassifier(BaseClassifier):
    """
    A semantic embedding-based classifier that extends BaseClassifier.
    Uses sentence-transformers embeddings + a downstream model (default SVC).
    """

    def __init__(self, model=None, embedding_model_name="thenlper/gte-base-zh", debug=False):
        """
        :param model: A scikit-learn classifier. Defaults to SVC(kernel='linear', probability=True).
        :param embedding_model_name: The huggingface or local path to the sentence-transformers model.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        if model is None:
            model = SVC(kernel='linear', probability=True)

        self.model_pipeline = Pipeline([
            ('embed', EmbeddingTransformer(model_name=embedding_model_name, debug=debug)),
            ('clf', model)
        ])
        self.is_fitted = False
        self.debug = debug

    def train_model(self, X: List[str], y: List[str], cv: int = 5):
        """
        Train the embedding-based pipeline using cross-validation.

        :param X: List of text samples
        :param y: List of labels
        :param cv: Number of cross-validation folds
        :return: Dictionary containing 'mean_accuracy' and 'mean_f1'
        """
        scoring = ['accuracy', 'f1_weighted']
        scores = cross_validate(self.model_pipeline, X, y, cv=cv, scoring=scoring)

        mean_accuracy = scores['test_accuracy'].mean()
        mean_f1 = scores['test_f1_weighted'].mean()

        print(f"[SemanticClassifier] CV={cv}")
        print(f" - Mean Accuracy: {mean_accuracy:.4f}")
        print(f" - Mean F1 Score: {mean_f1:.4f}")

        self.model_pipeline.fit(X, y)
        self.is_fitted = True
        print("[SemanticClassifier] Model trained on the full dataset.")

        return {
            "mean_accuracy": mean_accuracy,
            "mean_f1": mean_f1
        }

    def predict(self, text_list: Union[List[str], str]) -> List[str]:
        """
        Predict labels using the embedding-based pipeline.

        :param text_list: Single text or list of texts
        :return: List of predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("SemanticClassifier model is not fitted yet.")
        if isinstance(text_list, str):
            text_list = [text_list]
        return self.model_pipeline.predict(text_list)
