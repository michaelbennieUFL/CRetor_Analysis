from typing import List, Union
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer

from classifiers.base_classifier import BaseClassifier


class TFIDFClassifier(BaseClassifier):
    """
    A TF-IDF based classifier that extends BaseClassifier.
    """

    def __init__(self, model=None):
        """
        :param model: A scikit-learn classifier. Defaults to SVC(kernel='linear', probability=True).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        if model is None:
            model = SVC(kernel='linear', probability=True)

        # Pipeline: TfidfVectorizer + model
        self.model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2),
                                      lowercase=True,
                                      stop_words='english',
                                      max_df=0.95,
                                      min_df=2)),
            ('clf', model)
        ])
        self.is_fitted = False

    def train_model(self, X: List[str], y: List[str], cv: int = 5):
        """
        Train (and cross-validate) the TF-IDF pipeline.

        :param X: List of text documents
        :param y: Corresponding labels
        :param cv: Number of cross-validation folds
        :return: Dictionary containing 'mean_accuracy' and 'mean_f1'
        """
        scoring = ['accuracy', 'f1_weighted']
        scores = cross_validate(self.model_pipeline, X, y, cv=cv, scoring=scoring)

        mean_accuracy = scores['test_accuracy'].mean()
        mean_f1 = scores['test_f1_weighted'].mean()

        print(f"[TFIDFClassifier] CV={cv}")
        print(f" - Mean Accuracy: {mean_accuracy:.4f}")
        print(f" - Mean F1 Score: {mean_f1:.4f}")

        # Train on the full dataset
        self.model_pipeline.fit(X, y)
        self.is_fitted = True
        print("[TFIDFClassifier] Model trained on the full dataset.")

        return {
            "mean_accuracy": mean_accuracy,
            "mean_f1": mean_f1
        }

    def predict(self, text_list: Union[List[str], str]) -> List[str]:
        """
        Predict labels using the trained TF-IDF model pipeline.

        :param text_list: Single text string or a list of text strings
        :return: List of predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("TFIDFClassifier model is not fitted yet.")
        if isinstance(text_list, str):
            text_list = [text_list]
        return self.model_pipeline.predict(text_list)