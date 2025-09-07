#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train on combined_fix6.tsv and label baike_qa_valid.json, plus export the 1% least-certain subset.

Outputs:
  - ./output/predicted_labels.tsv
      Columns: Question, desc, Potentially_Pejorative (True/False)
  - ./output/lowest_confidence_1pct.tsv
      Same columns, but only the 1% with lowest self-confidence (via cleanlab)

Assumptions:
  - Your project provides: semantic_embedding_model.SemanticClassifier
  - combined_fix6.tsv has columns: Question, Potentially_Pejorative (string; empty/None => negative)
  - baike_qa_valid.json is either JSONL (one object per line) or a JSON array.
    We expect fields: "title" (question) and "desc" (optional).
"""

import os
import json
import math
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from cleanlab.filter import find_label_issues
from tqdm import tqdm

# Import your existing wrapper
from semantic_embedding_model import SemanticClassifier


# ---------------------------
# Paths / constants
# ---------------------------
DEFAULT_TRAIN_PATH = "../../data/biasLabeling/training/combined_fix6.tsv"
DEFAULT_INPUT_JSON = "../../data/biasLabeling/testing/questions_to_self_annotate.json"
DEFAULT_OUT_DIR = "./labeled_data"

MODEL_NAME = "thenlper/gte-base-zh"
THRESHOLD = 0.5


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
# Main logic
# ---------------------------
def train_semantic_classifier(X_train: List[str], y_train: List[int]) -> SemanticClassifier:
    clf = SemanticClassifier(
        model=SVC(C=1, probability=True, random_state=42),
        embedding_model_name=MODEL_NAME,
        debug=False,
        threshold=THRESHOLD,
    )
    # Train on full dataset; you can add CV if you want logs, but user requested full training
    clf.train_model(X_train, y_train, cv=0)
    return clf


def predict_proba_batched(clf, texts, batch_size=512, desc="Scoring"):
    """
    Calls clf.predict_proba(texts) in mini-batches with a tqdm progress bar.
    Works with your SemanticClassifier that exposes predict_proba(list[str]) -> (N,2).
    """
    probs_list = []
    N = len(texts)
    for i in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc=desc):
        batch = texts[i:i+batch_size]
        probs_list.append(clf.predict_proba(batch))  # shape (B, 2)
    return np.vstack(probs_list)  # (N, 2)

def predict_with_probs(clf: SemanticClassifier, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    probs = predict_proba_batched(clf, texts, batch_size=512, desc="Predicting")
    preds = (probs[:, 1] >= THRESHOLD).astype(int)
    return preds, probs

def build_outputs(df_infer: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    out = df_infer.copy()
    out["Potentially_Pejorative"] = preds.astype(bool)
    return out[["Question", "desc", "Potentially_Pejorative"]]


def lowest_confidence_1pct(pred_labels: np.ndarray, pred_probs: np.ndarray, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use cleanlab's self-confidence to identify least-certain items.
    For unlabeled data, we treat the predicted label as the 'given' label;
    self-confidence = p(model assigns to that predicted class).
    """
    # Cleanlab wants labels in {0,1,...} and pred_probs with same columns.
    labels = pred_labels.astype(int)
    # Compute indices ranked by self-confidence (lowest first) using cleanlab
    issue_idx = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    n = len(labels)
    print(issue_idx)
    print("[INFO] Found {} labels.".format(n))
    print("[INFO] Found {} issues.".format(len(issue_idx)))
    k = max(1, math.ceil(0.01 * n))
    chosen = issue_idx[:k]

    # Build a small frame with the same columns; keep order by increasing uncertainty
    sub = base_df.iloc[chosen].copy()
    # Add the self-confidence column to make it inspectable
    self_conf = pred_probs[np.arange(n), labels][chosen]
    sub["self_confidence"] = self_conf
    # Sort by self-confidence ascending (lowest certainty first)
    sub = sub.sort_values("self_confidence", ascending=True)
    return sub[["Question", "desc", "Potentially_Pejorative", "self_confidence"]]


def main():
    parser = argparse.ArgumentParser(description="Train on combined_fix6.tsv and predict + uncertainty on baike_qa_valid.json")
    parser.add_argument("--train_tsv", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print(f"[1/5] Loading training set: {args.train_tsv}")
    X_train, y_train = load_training_tsv(args.train_tsv)
    print(f"    Training rows: {len(X_train)}")

    print(f"[2/5] Training SemanticClassifier (SVM C=1, threshold={THRESHOLD}, emb={MODEL_NAME})")
    clf = train_semantic_classifier(X_train, y_train)
    print("    Training complete.")

    print(f"[3/5] Loading inference set: {args.input_json}")
    df_infer = load_inference_set(args.input_json)
    print(f"    Inference rows: {len(df_infer)}")

    print("[4/5] Predicting labels and probabilities…")
    preds, probs = predict_with_probs(clf, df_infer["Question+Desc"].tolist())

    # Build full labeled TSV
    labeled_df = build_outputs(df_infer, preds)
    labeled_path = os.path.join(args.out_dir, "predicted_labels.tsv")
    labeled_df.to_csv(labeled_path, sep="\t", index=False, encoding="utf-8")
    print(f"    Wrote: {labeled_path}")

    # 1% least certain via cleanlab self-confidence
    low_conf_df = lowest_confidence_1pct(preds, probs, labeled_df)
    low_conf_path = os.path.join(args.out_dir, "lowest_confidence_1pct.tsv")
    low_conf_df.to_csv(low_conf_path, sep="\t", index=False, encoding="utf-8")
    print(f"    Wrote: {low_conf_path}")

    print("[5/5] Done.")


if __name__ == "__main__":
    main()
