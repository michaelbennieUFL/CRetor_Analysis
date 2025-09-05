#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find likely label issues in a TSV dataset with columns:
  - Question
  - Potentially_Pejorative  (values like 'Potentially' vs 'None'/empty)

Outputs:
  - label_issues.csv        (ranked suspicious rows)
  - data_dropped.tsv        (dataset with likely issues removed)
  - embeddings.npy          (cached sentence embeddings)
  - pred_probs.npy          (CV out-of-sample predicted probabilities)

Requirements:
  pip install sentence-transformers scikit-learn cleanlab pandas numpy
"""

import argparse
import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer


# ------------------------------
# Config dataclass
# ------------------------------
@dataclass
class Config:
    tsv_path: str
    text_col: str = "Question"
    label_col: str = "Potentially_Pejorative"
    pos_values: Tuple[str, ...] = ("Potentially",)  # treated as positive class (1)
    embed_model: str = "thenlper/gte-base-zh"
    cv_folds: int = 10
    random_state: int = 42
    out_dir: str = "./label_issue_outputs"
    cache_embeddings: bool = True
    cache_pred_probs: bool = True


# ------------------------------
# I/O utilities
# ------------------------------
def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_tsv(tsv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Input file must contain columns '{text_col}' and '{label_col}'. "
            f"Found: {list(df.columns)}"
        )
    return df[[text_col, label_col]].copy()


def normalize_labels(raw: pd.Series, pos_values: Tuple[str, ...]) -> np.ndarray:
    y = []
    for v in raw.astype(str).tolist():
        v_stripped = v.strip().lower()
        if v_stripped in {p.lower() for p in pos_values}:
            y.append(1)
        elif v_stripped in ("", "none", "nan"):
            y.append(0)
        else:
            # default: anything not in positive list counts as negative
            y.append(0)
    return np.array(y, dtype=int)


# ------------------------------
# Embeddings
# ------------------------------
def get_embeddings(
    texts: List[str],
    model_name: str,
    out_dir: str,
    cache: bool = True,
) -> np.ndarray:
    emb_path = os.path.join(out_dir, "embeddings.npy")
    if cache and os.path.exists(emb_path):
        return np.load(emb_path)

    model = SentenceTransformer(model_name)
    # Batch encode for speed; disable progress bar to keep logs clean
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    if cache:
        np.save(emb_path, embeddings)
    return embeddings


# ------------------------------
# CV predicted probabilities
# ------------------------------
def get_oof_pred_probs(
    X_embed: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
    out_dir: str,
    cache: bool = True,
) -> np.ndarray:
    pred_path = os.path.join(out_dir, "pred_probs.npy")
    if cache and os.path.exists(pred_path):
        return np.load(pred_path)

    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # For cleanlab we need calibrated class probabilities for all samples (out-of-sample)
    pred_probs = cross_val_predict(
        estimator=clf,
        X=X_embed,
        y=y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )
    if cache:
        np.save(pred_path, pred_probs)
    return pred_probs


# ------------------------------
# Cleanlab: find label issues
# ------------------------------
def find_issues(
    y: np.ndarray,
    pred_probs: np.ndarray,
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    out_dir: str,
    row_min: Optional[int] = None,
    row_max: Optional[int] = None,
) -> pd.DataFrame:
    issue_idx = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    given_label_probs = pred_probs[np.arange(len(y)), y]
    alt_label = 1 - y
    alt_prob = pred_probs[np.arange(len(y)), alt_label]

    report = pd.DataFrame({
        "row_index": np.arange(len(y)),
        "text": df[text_col].values,
        "given_label_raw": df[label_col].values,
        "given_label_binary": y,
        "given_label_prob": given_label_probs,
        "alt_label_suggestion": alt_label,
        "alt_label_prob": alt_prob,
    })

    report["suspicious"] = False
    report.loc[issue_idx, "suspicious"] = True

    # 🔹 Apply range filter if provided
    if row_min is not None and row_max is not None:
        report = report[(report["row_index"] >= row_min) & (report["row_index"] <= row_max)]

    report.sort_values(
        by=["suspicious", "given_label_prob"],
        ascending=[False, True],
        inplace=True
    )

    report_path = os.path.join(out_dir, "label_issues.csv")
    report.to_csv(report_path, index=False, encoding="utf-8")
    return report



# ------------------------------
# Save a dropped version (remove suspicious rows)
# ------------------------------
def save_dropped_dataset(
    df: pd.DataFrame,
    issues_report: pd.DataFrame,
    text_col: str,
    label_col: str,
    out_dir: str,
) -> str:
    suspicious_rows = issues_report[issues_report["suspicious"]].row_index.values
    keep_mask = ~df.reset_index().index.isin(suspicious_rows)
    dropped_df = df[keep_mask].copy()
    out_path = os.path.join(out_dir, "data_dropped.tsv")
    dropped_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
    return out_path


# ------------------------------
# Simple CV performance snapshot (optional sanity check)
# ------------------------------
def quick_cv_report(X_embed: np.ndarray, y: np.ndarray, cv_folds: int, random_state: int) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    clf = LogisticRegression(max_iter=2000)
    y_pred = cross_val_predict(clf, X_embed, y, cv=cv, method="predict", n_jobs=-1)
    rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
    # Return a small summary
    return {
        "accuracy": rep.get("accuracy", np.nan),
        "precision_weighted": rep["weighted avg"]["precision"],
        "recall_weighted": rep["weighted avg"]["recall"],
        "f1_weighted": rep["weighted avg"]["f1-score"],
    }


# ------------------------------
# Main
# ------------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Automatically find likely label issues with cleanlab.")
    p.add_argument("--tsv", required=True, help="Path to TSV with columns Question and Potentially_Pejorative")
    p.add_argument("--embed_model", default="thenlper/gte-base-zh", help="SentenceTransformer model name")
    p.add_argument("--cv", type=int, default=10, help="Number of CV folds")
    p.add_argument("--out_dir", default="./label_issue_outputs", help="Output directory")
    p.add_argument("--no_cache", action="store_true", help="Do not cache embeddings/predictions")
    args = p.parse_args()

    return Config(
        tsv_path=args.tsv,
        embed_model=args.embed_model,
        cv_folds=args.cv,
        out_dir=args.out_dir,
        cache_embeddings=not args.no_cache,
        cache_pred_probs=not args.no_cache,
    )


def main(cfg: Config) -> None:
    ensure_out_dir(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print(f"[1/6] Loading: {cfg.tsv_path}")
    df = load_tsv(cfg.tsv_path, cfg.text_col, cfg.label_col)

    print(f"[2/6] Normalizing labels (positive values: {cfg.pos_values})")
    y = normalize_labels(df[cfg.label_col], cfg.pos_values)
    texts = df[cfg.text_col].fillna("").astype(str).tolist()

    print(f"[3/6] Computing embeddings with '{cfg.embed_model}' (this may take a few minutes)…")
    X_embed = get_embeddings(texts, cfg.embed_model, cfg.out_dir, cache=cfg.cache_embeddings)
    print(f"    Embedding shape: {X_embed.shape}")

    print(f"[4/6] Getting out-of-fold predicted probabilities via {cfg.cv_folds}-fold CV…")
    pred_probs = get_oof_pred_probs(
        X_embed, y, cfg.cv_folds, cfg.random_state, cfg.out_dir, cache=cfg.cache_pred_probs
    )

    print(f"[5/6] Finding likely label issues with cleanlab…")
    issues_report = find_issues(y, pred_probs, df, cfg.text_col, cfg.label_col, cfg.out_dir, row_min=0, row_max=2651)
    n_suspicious = int(issues_report["suspicious"].sum())
    print(f"    Found {n_suspicious} suspicious rows. Saved: {os.path.join(cfg.out_dir, 'label_issues.csv')}")

    # Save only suspicious rows with their text + original labels
    suspicious_only = issues_report.loc[issues_report["suspicious"], ["row_index", "text", "given_label_raw"]]
    suspicious_only.to_csv(
        os.path.join(cfg.out_dir, "suspicious_only.csv"),
        index=False,
        encoding="utf-8"
    )
    print(f"    Wrote: {os.path.join(cfg.out_dir, 'suspicious_only.csv')}")

    print(f"[6/6] Writing dataset with suspicious rows dropped…")
    dropped_path = save_dropped_dataset(df, issues_report, cfg.text_col, cfg.label_col, cfg.out_dir)
    print(f"    Wrote: {dropped_path}")

    # Optional: quick CV score on original (for sanity)
    metrics = quick_cv_report(X_embed, y, cfg.cv_folds, cfg.random_state)
    print("\nQuick CV snapshot on original labels (for context):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    cfg = parse_args()
    try:
        main(cfg)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
