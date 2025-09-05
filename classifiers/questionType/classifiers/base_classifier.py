import json
from typing import Union, List, Tuple, Dict


class BaseClassifier:
    """
    A base class that outlines the common interface for all classifiers.
    """

    def load_data_from_json(self, json_path: str) -> List[dict]:
        """
        Load data from a JSON file.

        :param json_path: Path to the JSON file.
        :return: A list of dictionaries containing the loaded data.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def preprocess_data(self, data: List[dict]) -> Dict[str, List[List[str]]]:
        """
        Convert JSON list of dicts into multiple (X, y) pairs for different labels.

        :param data: list of records from the JSON
        :return: dict with keys corresponding to different label sets
                 e.g. verification_timeline, evidence_quality, etc.
                 Each value is a list [X_list, y_list]
        """
        processed_data = {
            "promise_status": [[], []],
            "evidence_status": [[], []],
            "evidence_quality": [[], []],
            "verification_timeline": [[], []],
            "promise_string": [[], []],
            "evidence_string": [[], []],
        }

        for record in data:
            text = record.get("extracted_text", "")
            label = record.get("promise_status", "No")
            evidence_label = record.get("evidence_status", "No")
            verification_label = record.get("verification_timeline", "No")
            evidence_quality_label = record.get("evidence_quality", "N/A")
            evidence_string = record.get("evidence_string", "")
            promise_string = record.get("promise_string", "")

            # process_promise_classifier_data
            if label == "Yes":
                promise = record.get('promise_string', "")
                if promise:
                    processed_data["promise_status"][0].extend([text, promise])
                    processed_data["promise_status"][1].extend(["Yes", "Yes"])
                    processed_data["promise_string"][0].append(label)
                    processed_data["promise_string"][1].append(promise_string)
            else:
                processed_data["promise_status"][0].append(text)
                processed_data["promise_status"][1].append(label)

            # process_evidence_classifier_data
            if label == "Yes":
                if evidence_label == "Yes":
                    evidence = record.get('evidence_string', "")
                    if evidence:
                        processed_data["evidence_status"][0].extend([text, evidence])
                        processed_data["evidence_status"][1].extend(["Yes", "Yes"])
                        processed_data["evidence_string"][0].append(text)
                        processed_data["evidence_string"][1].append(evidence_string)
                else:
                    processed_data["evidence_status"][0].append(text)
                    processed_data["evidence_status"][1].append(evidence_label)

                # process_evidence_quality_data
                if evidence_label == "Yes":
                    processed_data["evidence_quality"][0].append(text)
                    processed_data["evidence_quality"][1].append(evidence_quality_label)

                # process_verification_timeline_data
                processed_data["verification_timeline"][0].append(text)
                processed_data["verification_timeline"][1].append(verification_label)

        return processed_data

    def preprocess_english_data(self, data: List[dict]) -> Dict[str, List[List[dict]]]:
        """
        Convert JSON list of dicts into multiple (X, y) pairs for different labels.

        :param data: list of records from the JSON
        :return: dict with keys corresponding to different label sets
                     e.g. verification_timeline, evidence_quality, etc.
                     Each value is a list [X_list, y_list]
        """
        processed_data = {
            "promise_status": [[], []],
            "evidence_status": [[], []],
            "evidence_quality": [[], []],
            "verification_timeline": [[], []],
        }

        for record in data:
            # Use the entire record as the X value
            x_value = record["data"]

            # Extract labels
            promise_status = record.get("promise_status", "No")
            evidence_status = record.get("evidence_status", "No")
            verification_timeline = record.get("verification_timeline", "N/A")
            evidence_quality = record.get("evidence_quality", "N/A")

            # Process promise_status data
            processed_data["promise_status"][0].append(x_value)
            processed_data["promise_status"][1].append(promise_status)

            if promise_status == "Yes":
                # Process evidence_status data
                processed_data["evidence_status"][0].append(x_value)
                processed_data["evidence_status"][1].append(evidence_status)

                # Process evidence_quality data
                if evidence_status == "Yes":  # Only relevant when evidence exists
                    processed_data["evidence_quality"][0].append(x_value)
                    processed_data["evidence_quality"][1].append(evidence_quality)

                # Process verification_timeline data
                processed_data["verification_timeline"][0].append(x_value)
                processed_data["verification_timeline"][1].append(verification_timeline)

        return processed_data


    def train_model(self, X: List[str], y: List[str], cv: int = 5):
        """
        Train a model on the given data, returning cross-validation scores.

        :param X: List of text samples (features)
        :param y: List of labels
        :param cv: Number of cross-validation folds
        :return: A dictionary of cross-validation scores (e.g., accuracy, f1).
        """
        raise NotImplementedError("Subclasses must implement train_model().")

    def predict(self, text_list: Union[List[str], str]) -> List[str]:
        """
        Predict labels for the given text snippet(s).

        :param text_list: Single text string or a list of text strings.
        :return: List of predicted labels.
        """
        raise NotImplementedError("Subclasses must implement predict().")
