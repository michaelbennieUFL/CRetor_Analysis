# main_test.py

from typing import Dict, Tuple, List




def testClassifiers(
    classifiers: Dict[str, object],
    dataset: Tuple[List[str], List[str]],
    cv: int = 5,
    top_n: int = 3
) -> List[str]:
    """
    Trains and evaluates each classifier in `classifiers` on the given dataset.
    Prints cross-validation results and returns the top_n classifier names
    (ordered by F1 score).

    :param classifiers: Dictionary of named classifiers
    :param dataset: A tuple (X, y) from the preprocessed data
    :param cv: Number of CV folds
    :param top_n: Return the top 'n' classifiers, default=3
    :return: List of top_n classifier names (ordered by descending F1 score).
    """
    X, y = dataset

    # We'll store (classifier_name, f1_score) so we can sort later
    results = []

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        scores = clf.train_model(X, y, cv=cv)
        # We assume 'scores' is a dict with keys: mean_accuracy, mean_f1
        mean_precision = scores["mean_precision"]
        results.append((name, mean_precision))

    # Sort results by F1 desc
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Overall Results (sorted by precision) ---")
    for i, (clf_name, mean_precision) in enumerate(results, start=1):
        print(f"{i}. {clf_name}: Precision={mean_precision:.4f}")

    return results[:top_n]


def testAllParameters(
    classifiers: Dict[str, object],
    processed_data: Dict[str, List[List[str]]],
    cv: int = 5,
    top_n: int = 3
):
    """
    Tests the given `classifiers` on four label sets, returning the top_n
    classifiers (by F1) for each label set.
    """
    # 1) promise_status
    print("\n=== PEJORATIVE STATUS CLASSIFICATION ===")
    top_ps = testClassifiers(classifiers, processed_data, cv=cv, top_n=top_n)




    print("\n=== Results Summary ===")
    print(f"Pejorative Status top {top_n}: {top_ps}")


