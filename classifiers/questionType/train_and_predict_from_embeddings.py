#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train on precomputed embeddings (train) and predict on embeddings (infer).

Inputs:
  --train-npz   artifacts/train_embeds/embeddings.npz   (must have y)
  --infer-npz   artifacts/infer_embeds/embeddings.npz
  --infer-meta  artifacts/infer_embeds/meta.jsonl       (same row count as infer X)

Outputs:
  - augmented.jsonl  (meta + predicted_prob + predicted_label)
  - possibly.jsonl   (subset where predicted_label == true)

Model: SVC(C=1, probability=True, random_state=42)
Threshold: 0.5 (change with --threshold)
"""

import argparse, os, json
from typing import Any, Dict, Iterable, List
import numpy as np
from cuml.svm import SVC
from sklearn.base import BaseEstimator
from tqdm import tqdm
import cupy as cp
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from itertools import cycle


class ThresholdClassifier(BaseEstimator):
    def __init__(self, base_clf=None, threshold: float = 0.5):
        self.base_clf = base_clf if base_clf is not None else SVC(C=1, probability=True, random_state=42)
        self.threshold = float(threshold)
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_clf.fit(X, y); return self
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_clf.predict_proba(X)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:,1] >= self.threshold).astype(int)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Train SVC on embeddings and predict.")
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--infer-npz", required=True)
    ap.add_argument("--infer-meta", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    
    cluster = LocalCUDACluster()   # uses all visible GPUs by default
    client = Client(cluster)
    workers = list(client.scheduler_info()["workers"].keys())
    n_workers = len(workers)

    ensure_dir(args.out_dir)

    train = load_npz(args.train_npz)
    Xtr = train["X"]
    has_y = bool(int(train.get("has_y", np.array([0]))[0]))
    if not has_y or "y" not in train or train["y"].shape[0] == 0:
        raise SystemExit("Training NPZ must include labels y. Re-run embedding with --label-col.")
    ytr = train["y"].astype(int)
    if Xtr.shape[0] != ytr.shape[0]:
        raise SystemExit(f"Train shapes mismatch: X {Xtr.shape}, y {ytr.shape}")

    clf = ThresholdClassifier(threshold=args.threshold).fit(Xtr, ytr)

    infer = load_npz(args.infer_npz)
    Xinfer = infer["X"]
    meta_rows = read_jsonl(args.infer_meta)
    if Xinfer.shape[0] != len(meta_rows):
        raise SystemExit(f"Inference shapes mismatch: X {Xinfer.shape[0]} vs meta rows {len(meta_rows)}")
        
        
    model_b = client.scatter(clf.base_clf, broadcast=True)
    


    def _proba_chunk(model, X_host_chunk):
        X_dev = cp.asarray(X_host_chunk,dtype=cp.float32)               # send chunk to this worker's GPU
        p = model.predict_proba(X_dev)[:, 1]           # cuML SVC on GPU
        return cp.asnumpy(p)                           # return to host for concatenation

    # split inference rows roughly evenly across workers
    chunks = np.array_split(Xinfer, n_workers*8)

    from itertools import cycle
    worker_cycle = cycle(workers)

    # submit all chunks, round robin to workers
    futures = [
        client.submit(_proba_chunk, model_b, chunk, workers=[next(worker_cycle)], pure=False)
        for chunk in chunks
    ]

    probs_parts = client.gather(futures)
    probs = np.concatenate(probs_parts)

    # sanity check
    if probs.shape[0] != Xinfer.shape[0]:
        raise SystemExit(f"Probs len {probs.shape[0]} != Xinfer rows {Xinfer.shape[0]}")

    preds = (probs >= args.threshold).astype(int)



    augmented, possibly = [], []
    for row, p, lbl in tqdm(zip(meta_rows, probs, preds), total=len(meta_rows), desc="Writing"):
        out = dict(row)
        out["predicted_prob"] = float(p)
        out["predicted_label"] = bool(lbl)
        augmented.append(out)
        if lbl == 1:
            possibly.append(out)

    aug_path = os.path.join(args.out_dir, "augmented.jsonl")
    pos_path = os.path.join(args.out_dir, "possibly.jsonl")
    write_jsonl(aug_path, augmented)
    write_jsonl(pos_path, possibly)
    print(f"✓ Wrote: {aug_path}")
    print(f"✓ Wrote: {pos_path}")

if __name__ == "__main__":
    main()
