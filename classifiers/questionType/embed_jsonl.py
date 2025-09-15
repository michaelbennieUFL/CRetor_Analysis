#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
from typing import Any, Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from itertools import islice
import math, gc
import numpy as np
from sentence_transformers import SentenceTransformer

os.environ.pop("TRANSFORMERS_CACHE", None)  

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def encode_stream_mp(model: SentenceTransformer,
                     texts: list[str],
                     pool,
                     batch_size: int = 128,
                     outer_chunk_size: int = 2000,
                     fp16: bool = True,
                     max_seq_len: int | None = 256,
                     desc: str = "Embedding") -> np.ndarray:
    """Encode in multi-process mode but stream in small slices to avoid OOM,
    showing ONE tqdm bar for chunks."""
    if fp16:
        try: model.half()
        except Exception: pass
    if max_seq_len is not None:
        try: model.max_seq_length = max_seq_len
        except Exception: pass

    total = len(texts)
    n_chunks = math.ceil(total / outer_chunk_size)
    embs, done = [], 0

    with tqdm(total=n_chunks, desc=desc, unit="chunk") as pbar:
        for idx, chunk in enumerate(chunked(texts, outer_chunk_size), 1):
            # ⚠️ disable inner bar so we only see the chunk-level bar
            emb = model.encode(
                chunk,
                pool=pool,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embs.append(emb.astype(np.float32, copy=False))
            done += len(chunk)
            pbar.update(1)
            pbar.set_postfix_str(f"{done}/{total}")

            # free memory between chunks
            del emb
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    if embs:
        return np.vstack(embs)
    # fallback empty array with correct width
    dim = model.get_sentence_embedding_dimension()
    return np.empty((0, dim), dtype=np.float32)

# --- Helper functions---
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sanitize_text(v: Any) -> str:
    if v is None: return ""
    s = str(v)
    return s.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t").strip()

def load_tsv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    # very simple TSV reader (no pandas dependency)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        cols = header.split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            parts = line.split("\t")
            # pad if short
            parts += [""] * max(0, len(cols) - len(parts))
            row = {cols[i]: parts[i] for i in range(len(cols))}
            rows.append(row)
    return rows, cols

def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    # supports .json (array of objects) or .jsonl
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        s = head.lstrip()
        if s.startswith("["):  # JSON array
            rows = json.load(f)
            rows = [r for r in rows if isinstance(r, dict)]
        else:  # JSONL
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict): rows.append(obj)
                except json.JSONDecodeError:
                    continue
    return rows

def default_label_from_tsv(val: Any) -> int:
    # your TSV uses "Potentially" vs "None"/empty
    s = "" if val is None else str(val).strip().lower()
    return int(s not in {"", "none", "nan"})

def truthy_to_label(v: Any) -> int:
    if v is None: return 0
    s = str(v).strip().lower()
    return int(s in {"1", "true", "yes", "y", "potentially", "pos", "positive"})

def main():
    ap = argparse.ArgumentParser(description="Embed TSV or JSON/JSONL with selected columns.")
    ap.add_argument("--input", required=True, help="Path to .tsv / .json / .jsonl")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--embedding-model", default="thenlper/gte-base-zh", help="SentenceTransformer hub name")
    ap.add_argument("--batch-size", type=int, default=512, help="Batch size PER GPU.") # Clarified help text

    # TSV-specific
    ap.add_argument("--tsv-text-col", default="Question", help="Text column for TSV")
    ap.add_argument("--label-col", default=None, help="Optional label column (both TSV and JSON supported)")

    # JSON-specific
    ap.add_argument("--json-text-cols", nargs="+", default=["title", "desc"], help="Ordered text cols for JSON/JSONL")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # --- Data loading logic remains the same ---
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".tsv":
        rows, cols = load_tsv(args.input)
        text_col = args.tsv_text_col
        if text_col not in cols:
            raise SystemExit(f"TSV missing column: {text_col}")
        texts = [sanitize_text(r.get(text_col, "")) for r in rows]
        # label
        y = None; has_y = False
        if args.label_col and args.label_col in cols:
            y = np.asarray([default_label_from_tsv(r.get(args.label_col)) for r in rows], dtype=np.int32)
            has_y = True
        text_schema = [text_col]
    else:
        rows = read_json_or_jsonl(args.input)
        if not rows: raise SystemExit("No rows found.")
        texts = []
        for r in rows:
            parts = [sanitize_text(r.get(c, "")) for c in args.json_text_cols]
            texts.append(" ".join([p for p in parts if p]))
        # label
        y = None; has_y = False
        if args.label_col is not None:
            y = np.asarray([truthy_to_label(r.get(args.label_col)) for r in rows], dtype=np.int32)
            has_y = True
        text_schema = args.json_text_cols

    # --- ✨ NEW: Multi-GPU Embedding Logic ✨ ---
    print("Initializing SentenceTransformer model...")
    model = SentenceTransformer(args.embedding_model)
    try:
        model.half() # Use half precision for speed/memory savings
        print("Using half precision (FP16).")
    except Exception:
        pass

    # 1. Start a pool of worker processes, one for each GPU
    print("Starting multi-GPU encoding pool...")
    pool = model.start_multi_process_pool()

    # 2. Prepare the different text prompts for concatenation
    texts_original = texts
    texts_prompt_eng = [f'Does the input "{text}" talk about race, ethnicity, gender, sex, occupations, or sexuality and hint at discrimination?' for text in texts]
    texts_prompt_zh = [(
        '下面是一些示例： '
        '"那你有哪里来的样本说明贵州人素质低了？"==True， '
        '"你喜欢美国人歧视亚裔的文化吗？你喜欢中国人歧视越南人的文化吗？你不喜欢，仅仅是受益人不是你罢了，人家没违法没犯罪，你为什么讨厌？"==True， '
        '"hub么？我在国外都不用翻墙🐶"==False， '
        '"你搁发廊作市场调研是吧，剩下的一个是不是扫黄的警察？说实话是不是集体蹲墙角刚被放出来？"==False， '
        '"作者你有父母吗？来把匿名打开，你在那装你妈"==False。'
        f'请问以上示例是否涉及敏感话题或含有歧视性内容？ {text}=='
    ) for text in texts]
    
    
        # ORIGINAL
    emb1 = encode_stream_mp(
        model, texts_original, pool,
        batch_size=min(args.batch_size, 512),     # lower if you still OOM
        outer_chunk_size=6000,                   # tune to fit your GPUs
        fp16=True,
        max_seq_len=1024,
        desc="Embedding original texts"
    )

    # EN-PROMPT
    emb2 = encode_stream_mp(
        model, texts_prompt_eng, pool,
        batch_size=min(args.batch_size, 512),
        outer_chunk_size=6000,
        fp16=True,
        max_seq_len=1024,
        desc="Embedding English-prompted texts"
    )

    # ZH-PROMPT
    emb3 = encode_stream_mp(
        model, texts_prompt_zh, pool,
        batch_size=min(args.batch_size, 512),
        outer_chunk_size=6000,
        fp16=True,
        max_seq_len=1024,
        desc="Embedding Chinese-prompted texts"
    )    
    

    # 4. Stop the worker processes
    model.stop_multi_process_pool(pool)
    print("Multi-GPU pool stopped.")

    # 5. Concatenate the results
    X = np.concatenate([emb1, emb2, emb3], axis=1).astype(np.float32)

    # --- File writing logic remains the same ---
    # write meta.jsonl
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in rows:
            r_out = {k: sanitize_text(v) if isinstance(v, (str, int, float, bool)) or v is None else v
                     for k, v in r.items()}
            f.write(json.dumps(r_out, ensure_ascii=False) + "\n")

    # write embeddings.npz
    npz_path = os.path.join(args.out_dir, "embeddings.npz")
    np.savez_compressed(
        npz_path,
        X=X,
        has_y=np.array([int(has_y)], dtype=np.int32),
        y=(y if has_y else np.zeros((0,), dtype=np.int32)),
        text_cols=np.array(text_schema, dtype=object),
        embedding_model=np.array([args.embedding_model], dtype=object),
        source=np.array([args.input], dtype=object),
        format=np.array([ext.replace(".", "")], dtype=object),
    )

    print(f"✓ Saved embeddings: {npz_path}")
    print(f"✓ Saved meta:       {meta_path}")
    print(f"Final embedding shape: {X.shape}")


if __name__ == "__main__":
    main()