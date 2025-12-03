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

import argparse, os, json, gc
from typing import Any, Dict, Iterable, List

import numpy as np
from cuml.svm import SVC
from sklearn.base import BaseEstimator
from tqdm import tqdm
import cupy as cp
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, as_completed
from dask.distributed import TimeoutError as DaskTimeoutError

class ThresholdClassifier(BaseEstimator):
    def __init__(self, base_clf=None, threshold: float = 0.5):
        # cuML currently ignores random_state for probabilistic SVC, warning is harmless
        self.base_clf = base_clf if base_clf is not None else SVC(
            C=1,
            probability=True,
            random_state=42
        )
        self.threshold = float(threshold)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_npz(path: str) -> Dict[str, Any]:
    # If embeddings are huge, you can add mmap_mode="r" in np.load for lower RAM usage.
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


def _proba_chunk(model, X_host_chunk):
    """
    Run predict_proba on a chunk on a single worker/GPU.
    X_host_chunk: numpy array on host for this chunk.
    """
    # move chunk to this worker's GPU
    X_dev = cp.asarray(X_host_chunk, dtype=cp.float32)
    p = model.predict_proba(X_dev)[:, 1]   # cuML SVC on GPU
    return cp.asnumpy(p)                   # back to host


def main():
    ap = argparse.ArgumentParser(description="Train SVC on embeddings and predict.")
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--infer-npz", required=True)
    ap.add_argument("--infer-meta", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--rows-per-chunk",
        type=int,
        default=10000,
        help="Approx number of inference rows per GPU chunk (tune for speed vs. memory).",
    )
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # ------------------ 1) TRAINING (single GPU) ------------------
    print("🔧 [1/3] Loading training embeddings & fitting SVC...")
    train = load_npz(args.train_npz)
    Xtr = train["X"]
    has_y = bool(int(train.get("has_y", np.array([0]))[0]))
    if not has_y or "y" not in train or train["y"].shape[0] == 0:
        raise SystemExit("Training NPZ must include labels y. Re-run embedding with --label-col.")
    ytr = train["y"].astype(int)
    if Xtr.shape[0] != ytr.shape[0]:
        raise SystemExit(f"Train shapes mismatch: X {Xtr.shape}, y {ytr.shape}")

    clf = ThresholdClassifier(threshold=args.threshold).fit(Xtr, ytr)
    print(f"✅ Trained SVC on {Xtr.shape[0]} samples with dim={Xtr.shape[1]}")

    # free training data from host memory
    del train, Xtr, ytr
    gc.collect()

    # ------------------ 2) INFERENCE (multi-GPU via Dask) ------------------
    print("📦 [2/3] Loading inference embeddings & meta...")
    infer = load_npz(args.infer_npz)
    Xinfer = infer["X"]
    meta_rows = read_jsonl(args.infer_meta)
    if Xinfer.shape[0] != len(meta_rows):
        raise SystemExit(f"Inference shapes mismatch: X {Xinfer.shape[0]} vs meta rows {len(meta_rows)}")

    n_rows = Xinfer.shape[0]
    print(f"   -> Inference rows: {n_rows}, dim={Xinfer.shape[1]}")

    # Start multi-GPU cluster only for inference to avoid extra overhead during training
    print("🚀 Starting LocalCUDACluster for multi-GPU inference ...")

    # nanny=False avoids the distributed.nanny shutdown warnings/timeouts
    cluster = LocalCUDACluster(
        nanny=False,
        threads_per_worker=1,   # good default for GPU workers
    )
    client = Client(cluster)

    try:
        workers = list(client.scheduler_info()["workers"].keys())
        n_workers = len(workers)
        print(f"   -> Using {n_workers} GPU workers: {workers}")

        # Broadcast trained base classifier to all workers
        model_b = client.scatter(clf.base_clf, broadcast=True)

        # Decide chunking
        rows_per_chunk = max(args.rows_per_chunk, 1)
        n_chunks = max((n_rows + rows_per_chunk - 1) // rows_per_chunk, 1)
        print(f"   -> Splitting inference into {n_chunks} chunks (~{rows_per_chunk} rows/chunk)")

        # Create host-side chunks
        chunks = [
            Xinfer[i * rows_per_chunk : (i + 1) * rows_per_chunk]
            for i in range(n_chunks)
        ]

        # Scatter chunks to workers to avoid embedding large arrays in the task graph
        chunk_futures = [client.scatter(ch, broadcast=False) for ch in chunks]

        # Submit inference tasks, track which chunk index each future corresponds to
        submit_map = {
            client.submit(_proba_chunk, model_b, ch_fut, pure=False): i
            for i, ch_fut in enumerate(chunk_futures)
        }

        # Collect results with a progress bar, preserving original order
        probs_parts = [None] * n_chunks
        for fut in tqdm(
            as_completed(submit_map),
            total=n_chunks,
            desc="Inference (multi-GPU)",
        ):
            idx = submit_map[fut]
            probs_parts[idx] = fut.result()

        probs = np.concatenate(probs_parts, axis=0)

    finally:
        # Try to shut down nicely; if it times out, just warn and continue.
        try:
            client.close()
        except Exception as e:
            print(f"⚠️ Warning: error closing Dask client (ignored): {e}")
        try:
            # give workers some time; adjust timeout if needed
            cluster.close(timeout=15)
        except (DaskTimeoutError, TimeoutError, Exception) as e:
            print(f"⚠️ Warning: error closing Dask cluster (ignored): {e}")

        # Cluster & client are closed here → fewer leaked semaphore warnings

        # Sanity check
        if probs.shape[0] != Xinfer.shape[0]:
            raise SystemExit(f"Probs len {probs.shape[0]} != Xinfer rows {Xinfer.shape[0]}")

        # Apply threshold on host
        preds = (probs >= args.threshold).astype(int)

    # ------------------ 3) WRITING OUTPUTS ------------------
    print("📝 [3/3] Writing outputs ...")
    augmented, possibly = [], []
    for row, p, lbl in tqdm(
        zip(meta_rows, probs, preds),
        total=len(meta_rows),
        desc="Writing JSONL",
    ):
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
