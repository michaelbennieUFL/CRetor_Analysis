#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a model with two prompts (basic vs optimized) on the Mandarin bias-topic dataset.

Metrics:
  1. Subset accuracy (exact match)
  2. Macro-averaged F1 (over labels)
  3. Macro-averaged accuracy (mean per-label accuracy)
  4. Hamming loss
  5. For each label: accuracy, precision, recall, F1

Inputs (CLI args):
  --model_name            (required)
  --api_base              (default: https://api.openai.com/v1)
  --api_key               (default: $OPENAI_API_KEY)
  --model_type            (default: chat; only "chat" is supported)
  --max_tokens            (default: 45000)
  --num_threads           (default: 5; parallel threads for inference)

  --project_csv_path      (required; project-4 TSV file)
  --topics_tsv_path       (required; predicted_topics TSV file)
  --optimized_prompt_path (required; file with optimized prompt text)
  --basic_prompt_path     (optional; if omitted, BASIC_PROMPT string is used)

Outputs (in ./results):
  1) {sanitized_model_name}_category_results.tsv
     - two rows: basic, optimized
  2) {sanitized_model_name}_predictions.jsonl
     - one record per (example, prompt_name), including raw output, parsed JSON, errors, etc.
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json_repair
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

# =========================
# 0. Logging
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# =========================
# 1. Category config
# =========================

CATEGORY_ID_TO_NAME: Dict[str, str] = {
    "4": "地域",
    "5": "中国的少数民族",
    "6": "种族文化",
    "7": "性别",
    "8": "性傾向",
    "9": "身心障碍",
    "0": "职业/教育",
    "q": "宗教",
    "w": "其他",
}
ALL_CATEGORY_IDS: List[str] = list(CATEGORY_ID_TO_NAME.keys())

# =========================
# 2. Dataset builder
# =========================


def build_topic_dataframe_from_files(
    project_csv_path: str,
    topics_tsv_path: str,
) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
      - idx   (string)
      - text  (combined question + context)
      - groups: list of topic IDs as strings, e.g. ["4", "7"]

    project_csv_path: project-4 TSV.
      must contain:
        - 'idx'
        - 'q' (main question text)
        - 'c' (optional context; may be NaN)

    topics_tsv_path: predicted_topics TSV, with columns:
        ['uncertainty', 'topics_地域', 'topics_少数民族', ..., 'topics_其他', 'idx']
    """
    df_proj = pd.read_csv(project_csv_path, sep="\t")
    df_topics = pd.read_csv(topics_tsv_path, sep="\t")

    logger.info("Loaded project file with %d rows", len(df_proj))
    logger.info("Loaded topics file with %d rows", len(df_topics))

    topic_col_to_id = {
        "topics_地域": "4",
        "topics_少数民族": "5",
        "topics_种族文化": "6",
        "topics_性别": "7",
        "topics_性傾向": "8",
        "topics_身心障碍": "9",
        "topics_职业教育": "0",
        "topics_宗教": "q",
        "topics_其他": "w",
    }
    topic_cols = list(topic_col_to_id.keys())

    rows: List[Dict[str, Any]] = []

    # ensure idx is string in both
    df_proj["idx"] = df_proj["idx"].astype(str)
    df_topics["idx"] = df_topics["idx"].astype(str)

    # iterate topics in order and merge with project
    for _, r in tqdm(
        df_topics.iterrows(),
        total=len(df_topics),
        desc="Merging topics with project text",
        unit="rows",
    ):
        idx = str(r["idx"])

        matches = df_proj[df_proj["idx"] == idx]
        if matches.empty:
            logger.debug("No matching text row for idx=%s; skipping.", idx)
            continue

        # take the first matching row if there are duplicates
        proj_row = matches.iloc[0]

        q_text = str(proj_row.get("q", ""))
        c_raw = proj_row.get("c", None)

        if c_raw is None or (isinstance(c_raw, float) and pd.isna(c_raw)):
            c_text = ""
        else:
            c_text = str(c_raw)

        text = q_text + c_text

        groups: List[str] = []
        for col in topic_cols:
            if col not in r:
                continue
            val = r[col]
            try:
                v_int = int(val)
            except Exception:
                v_int = 0
            if v_int == 1:
                groups.append(topic_col_to_id[col])

        rows.append({"idx": idx, "text": text, "groups": groups})

    df_data = pd.DataFrame(rows)
    logger.info("Built merged data with %d rows", len(df_data))
    return df_data


# =========================
# 3. Output parsing helpers
# =========================


def strip_final_answer_tags(pred_text: str) -> str:
    """Extract inner JSON from <final_answer> ... </final_answer>."""
    if pred_text is None:
        return ""
    s = str(pred_text)
    start_token = "<final_answer>"
    end_token = "</final_answer>"
    start = s.find(start_token)
    end = s.rfind(end_token)
    if start == -1 or end == -1:
        # No tags: assume whole string is JSON-ish
        return s.strip()
    inner = s[start + len(start_token) : end]
    return inner.strip()


def parse_groups_from_output(output_str: str) -> Tuple[List[str], Any, str]:
    """
    Parse the 'groups' field from the model's output.
    Uses json_repair to robustly handle slightly malformed JSON.

    Returns:
      groups:   list of valid category IDs (subset of ALL_CATEGORY_IDS)
      parsed:   parsed JSON object (dict / list / None)
      err_msg:  "" if no error, otherwise human-readable error description
    """
    if not isinstance(output_str, str) or not output_str.strip():
        return [], None, "empty or non-string output"

    json_str = strip_final_answer_tags(output_str).strip()

    try:
        obj = json_repair.repair_json(
            json_str,
            return_objects=True,
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning("Failed to repair/parse JSON: %s", e)
        return [], None, f"Failed to repair/parse JSON: {e}"

    # If it's not a dict, we still record obj, but note an error for 'groups'.
    if not isinstance(obj, dict):
        logger.warning("Parsed JSON is not an object: %r", obj)
        return [], obj, "parsed JSON is not an object (no 'groups' field)"

    groups = obj.get("groups", [])
    if not isinstance(groups, list):
        logger.warning("'groups' is not a list in parsed JSON: %r", obj)
        return [], obj, "'groups' field is not a list"

    cleaned: List[str] = []
    for g in groups:
        g_str = str(g).strip()
        if g_str in ALL_CATEGORY_IDS:
            cleaned.append(g_str)
        else:
            # silently drop unknown labels (not treated as error)
            logger.debug("Dropping unknown group label from output: %r", g_str)

    # Successful parse
    return cleaned, obj, ""


# =========================
# 4. Multi-label metrics
# =========================


def to_multi_hot(
    list_of_label_lists: List[List[str]],
    label_order: List[str],
) -> np.ndarray:
    """Convert list of sets/lists of labels into multi-hot matrix (n_samples x n_labels)."""
    n_samples = len(list_of_label_lists)
    n_labels = len(label_order)
    y = np.zeros((n_samples, n_labels), dtype=int)
    label_to_idx = {lab: i for i, lab in enumerate(label_order)}

    for i, groups in enumerate(list_of_label_lists):
        for g in groups:
            if g in label_to_idx:
                y[i, label_to_idx[g]] = 1
    return y


def compute_all_metrics(
    gold_groups: List[List[str]],
    pred_groups: List[List[str]],
    label_ids: List[str],
) -> Dict[str, float]:
    """
    Compute:
      - subset_accuracy
      - macro_f1
      - macro_accuracy
      - hamming_loss
      - per-label accuracy/precision/recall/f1
    """
    assert len(gold_groups) == len(pred_groups), "gold and pred length mismatch"
    y_true = to_multi_hot(gold_groups, label_ids)
    y_pred = to_multi_hot(pred_groups, label_ids)

    # 1. Subset accuracy (exact match)
    subset_acc = accuracy_score(y_true, y_pred)

    # 2. Macro-averaged F1 over labels
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 3. Macro-averaged accuracy (mean per-label accuracy)
    per_label_accs = []
    per_label_precisions = []
    per_label_recalls = []
    per_label_f1s = []

    n_labels = len(label_ids)
    for j in range(n_labels):
        y_true_j = y_true[:, j]
        y_pred_j = y_pred[:, j]

        acc_j = accuracy_score(y_true_j, y_pred_j)
        prec_j = precision_score(y_true_j, y_pred_j, zero_division=0)
        rec_j = recall_score(y_true_j, y_pred_j, zero_division=0)
        f1_j = f1_score(y_true_j, y_pred_j, zero_division=0)

        per_label_accs.append(acc_j)
        per_label_precisions.append(prec_j)
        per_label_recalls.append(rec_j)
        per_label_f1s.append(f1_j)

    macro_accuracy = float(np.mean(per_label_accs))

    # 4. Hamming loss
    hloss = hamming_loss(y_true, y_pred)

    metrics: Dict[str, float] = {
        "subset_accuracy": float(subset_acc),
        "macro_f1": float(macro_f1),
        "macro_accuracy": float(macro_accuracy),
        "hamming_loss": float(hloss),
    }

    # 5. Per-label metrics
    for idx, lab in enumerate(label_ids):
        name = CATEGORY_ID_TO_NAME.get(lab, lab)
        suffix = f"{lab}_{name}"
        metrics[f"accuracy_{suffix}"] = float(per_label_accs[idx])
        metrics[f"precision_{suffix}"] = float(per_label_precisions[idx])
        metrics[f"recall_{suffix}"] = float(per_label_recalls[idx])
        metrics[f"f1_{suffix}"] = float(per_label_f1s[idx])

    return metrics


# =========================
# 5. LLM inference (parallel)
# =========================

def extract_message_text(message) -> str:
    """
    Extract text from a ChatCompletionMessage, trying `content` first,
    then `reasoning_content`, then any extra fields.
    """
    # New OpenAI client uses a pydantic model
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content

    # Qwen / reasoning models often put text here
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content

    # Some servers may stash extra fields in model_extra/additional_kwargs
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        rc = extra.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            return rc

    # Fallback to empty string
    return ""




def run_model_on_dataset(
    client: OpenAI,
    model_name: str,
    prompt_text: str,
    texts: List[str],
    idxs: List[str],
    max_tokens: int,
    progress_label: str,
    prompt_name: str,
    num_threads: int,
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    """
    For each input text, call the model with:
      system: prompt_text
      user:   "文本: <text>"

    Runs in parallel using a ThreadPoolExecutor.

    Returns:
      - predictions: list of predicted group ID lists, one per text (same order as input).
      - logs: list of JSON-serializable dicts for JSONL logging (same order as input).
    """
    n = len(texts)
    assert n == len(idxs), "texts and idxs must have same length"

    predictions: List[List[str]] = [None] * n  # type: ignore
    logs: List[Dict[str, Any]] = [None] * n    # type: ignore

    def worker(i: int) -> Tuple[int, List[str], Dict[str, Any]]:
        raw_text = texts[i]
        idx = idxs[i]
        user_content = f"文本: {raw_text}"

        output = ""
        parsed = None
        parse_err_msg = ""
        groups: List[str] = []

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
            )

            msg = resp.choices[0].message
            output = extract_message_text(msg)

            # If you still want to debug, uncomment:
            # print(resp)
            # print("OUTPUT:", repr(output))

            groups, parsed, parse_err_msg = parse_groups_from_output(output)
        except Exception as e:
            logger.error(
                "Error during completion at index %d (idx=%s): %s. Treating as empty prediction.",
                i,
                idx,
                e,
            )
            parse_err_msg = f"completion error: {e}"
            groups = []
            parsed = None
            output = output or ""

        log_record: Dict[str, Any] = {
            "idx": idx,
            "prompt_name": prompt_name,
            "text": raw_text,
            "raw_output": output,
            "parsed": parsed,
            "groups": groups,
            "error": bool(parse_err_msg),
            "error_msg": parse_err_msg,
        }
        return i, groups, log_record

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(n)]
        with tqdm(total=n, desc=progress_label, unit="ex", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                i, groups, log_record = fut.result()
                predictions[i] = groups
                logs[i] = log_record
                pbar.update(1)

    return predictions, logs


# =========================
# 6. Main
# =========================

# You can leave this blank and/or edit it by hand,
# OR pass --basic_prompt_path to load from a file instead.
BASIC_PROMPT = ""  # baseline / unoptimized prompt (if not loaded from file)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model with basic vs optimized prompt on bias-topic dataset."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g. openai/inclusionAI/Ring-flash-2.0)",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://api.openai.com/v1",
        help="API base URL (e.g. http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key (default: $OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="chat",
        help="Model type (only 'chat' is supported)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=45000,
        help="Max tokens for completion",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=5,
        help="Number of parallel threads for inference (default: 5)",
    )

    parser.add_argument(
        "--project_csv_path",
        type=str,
        required=True,
        help="Path to project-4 TSV file",
    )
    parser.add_argument(
        "--topics_tsv_path",
        type=str,
        required=True,
        help="Path to predicted_topics TSV file",
    )
    parser.add_argument(
        "--optimized_prompt_path",
        type=str,
        required=True,
        help="Path to optimized prompt text file",
    )
    parser.add_argument(
        "--basic_prompt_path",
        type=str,
        default=None,
        help="Optional path to basic/unoptimized prompt text file",
    )

    args = parser.parse_args()

    if args.model_type != "chat":
        raise ValueError(f"Only model_type='chat' is supported, got {args.model_type}")

    # Load dataset (entire dataset, no splits)
    df_data = build_topic_dataframe_from_files(
        args.project_csv_path, args.topics_tsv_path
    )
    
    #df_data=df_data[:7]
    
    if df_data.empty:
        raise RuntimeError("Merged dataset is empty; check your input files.")

    texts = df_data["text"].tolist()
    gold_groups = df_data["groups"].tolist()
    idxs = df_data["idx"].tolist()

    # Load prompts
    optimized_prompt_text = Path(args.optimized_prompt_path).read_text(
        encoding="utf-8"
    )

    if args.basic_prompt_path is not None:
        basic_prompt_text = Path(args.basic_prompt_path).read_text(encoding="utf-8")
    else:
        basic_prompt_text = BASIC_PROMPT

    logger.info(
        "Using basic prompt length: %d characters", len(basic_prompt_text or "")
    )
    logger.info(
        "Using optimized prompt length: %d characters", len(optimized_prompt_text or "")
    )

    # Configure OpenAI-compatible client
    client = OpenAI(
        api_key=args.api_key if args.api_key else "dummy",
        base_url=args.api_base,
    )

    # Evaluate basic prompt
    logger.info("Running model with BASIC prompt...")
    basic_pred_groups, basic_logs = run_model_on_dataset(
        client=client,
        model_name=args.model_name,
        prompt_text=basic_prompt_text,
        texts=texts,
        idxs=idxs,
        max_tokens=args.max_tokens,
        progress_label="Inference (basic prompt)",
        prompt_name="basic",
        num_threads=args.num_threads,
    )
    basic_metrics = compute_all_metrics(
        gold_groups, basic_pred_groups, ALL_CATEGORY_IDS
    )
    basic_metrics["prompt_name"] = "basic"

    # Evaluate optimized prompt
    logger.info("Running model with OPTIMIZED prompt...")
    optimized_pred_groups, optimized_logs = run_model_on_dataset(
        client=client,
        model_name=args.model_name,
        prompt_text=optimized_prompt_text,
        texts=texts,
        idxs=idxs,
        max_tokens=args.max_tokens,
        progress_label="Inference (optimized prompt)",
        prompt_name="optimized",
        num_threads=args.num_threads,
    )
    optimized_metrics = compute_all_metrics(
        gold_groups, optimized_pred_groups, ALL_CATEGORY_IDS
    )
    optimized_metrics["prompt_name"] = "optimized"

    # Save results as TSV + JSONL
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = args.model_name.replace("/", "_").replace(":", "_")
    tsv_path = results_dir / f"{safe_model_name}_category_results.tsv"
    jsonl_path = results_dir / f"{safe_model_name}_predictions.jsonl"

    # TSV metrics
    df_results = pd.DataFrame([basic_metrics, optimized_metrics])
    cols = ["prompt_name"] + [c for c in df_results.columns if c != "prompt_name"]
    df_results = df_results[cols]
    df_results.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
    logger.info("Saved metrics to %s", tsv_path)

    # JSONL predictions (basic + optimized)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in basic_logs + optimized_logs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved predictions JSONL to %s", jsonl_path)


if __name__ == "__main__":
    main()
