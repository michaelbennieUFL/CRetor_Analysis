import numpy as np
from typing import List, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from classifiers.base_classifier import BaseClassifier


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    A second embedding transformer class to avoid conflicts with the
    other 'EmbeddingTransformer'. Uses the same singleton pattern.
    """
    _model_instance = None
    _model_name = None

    def __init__(self, model_name="thenlper/gte-base-zh", debug=False):
        if (EmbeddingTransformer._model_instance is None) or \
           (model_name != EmbeddingTransformer._model_name):
            if debug:
                print(f"[Singleton] Initializing the model: {model_name}")
            model = SentenceTransformer(model_name)
            try:
                model.half()
            except:
                pass
            EmbeddingTransformer._model_instance = model
            EmbeddingTransformer._model_name = model_name
        else:
            if debug:
                print(f"[Singleton] Reusing the existing model: {model_name}")

        self.debug = debug
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return EmbeddingTransformer._model_instance.encode(
            X, show_progress_bar=False, batch_size=32
        )


class MixedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Creates a combined feature vector from TF-IDF + embeddings.
    """

    def __init__(self, embedding_model_name="thenlper/gte-base-zh", tfidf_params=None, debug=False):
        self.debug = debug
        self.embedding_model_name = embedding_model_name
        self._embed_transformer = EmbeddingTransformer(model_name=embedding_model_name, debug=debug)
        if tfidf_params is None:
            tfidf_params = {
                "ngram_range": (1, 2),
                "lowercase": True,
                "stop_words": "english",
                "max_df": 0.95,
                "min_df": 2
            }
        self._tfidf_vectorizer = TfidfVectorizer(**tfidf_params)
        self.tfidf_params=tfidf_params

    def fit(self, X, y=None):
        self._tfidf_vectorizer.fit(X, y)
        self._embed_transformer.fit(X, y)
        return self

    def transform(self, X):
        tfidf_vec = self._tfidf_vectorizer.transform(X).toarray()
        embed_vec = self._embed_transformer.transform(X)

        combined = np.hstack([tfidf_vec, embed_vec])
        if self.debug:
            print("TF-IDF shape:", tfidf_vec.shape)
            print("Embedding shape:", embed_vec.shape)
            print("Combined shape:", combined.shape)
        return combined


class MixedMethodClassifier(BaseClassifier):
    """
    A classifier that uses both TF-IDF and embeddings as features.
    """

    def __init__(self, model=None, embedding_model_name="thenlper/gte-base-zh", tfidf_params=None, debug=False):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        if model is None:
            model = SVC(kernel='linear', probability=True)

        self.debug = debug
        self.pipeline = Pipeline([
            (
                "mixed_features",
                MixedFeaturesTransformer(
                    embedding_model_name=embedding_model_name,
                    tfidf_params=tfidf_params,
                    debug=debug
                )
            ),
            ("clf", model)
        ])
        self.is_fitted = False

    def train_model(self, X: List[str], y: List[str], cv: int = 5):
        """
        Train on combined TF-IDF + embedding features using cross-validation.

        :param X: List of texts
        :param y: List of labels
        :param cv: CV folds
        :return: Dictionary containing 'mean_accuracy' and 'mean_f1'
        """
        scoring = ['accuracy', 'f1_weighted']
        scores = cross_validate(self.pipeline, X, y, cv=cv, scoring=scoring)

        mean_accuracy = scores['test_accuracy'].mean()
        mean_f1 = scores['test_f1_weighted'].mean()

        print(f"[MixedMethodClassifier] CV={cv}")
        print(f" - Mean Accuracy: {mean_accuracy:.4f}")
        print(f" - Mean F1 Score: {mean_f1:.4f}")

        # Train on full dataset
        self.pipeline.fit(X, y)
        self.is_fitted = True
        print("[MixedMethodClassifier] Model trained on the full dataset.")

        return {
            "mean_accuracy": mean_accuracy,
            "mean_f1": mean_f1
        }

    def predict(self, text_list: Union[List[str], str]) -> List[str]:
        """
        Predict using the combined TF-IDF + embedding pipeline.
        """
        if not self.is_fitted:
            raise RuntimeError("MixedMethodClassifier model is not fitted yet.")
        if isinstance(text_list, str):
            text_list = [text_list]
        return self.pipeline.predict(text_list)
