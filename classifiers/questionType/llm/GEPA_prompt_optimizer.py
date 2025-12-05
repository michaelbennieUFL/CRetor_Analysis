#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GEPA + DSPy prompt optimization for Mandarin bias-topic extraction.

Inputs:
  - project-4 CSV: contains the text (column 'a') and an 'idx' key.
  - predicted_topics TSV: contains gold multi-label topic columns and the same 'idx'.

We build a dataset where each example is:
  text  = "文本: <a>"
  gold groups = list of topic IDs, e.g. ["4", "7"]

Categories (ID -> description):

  [4] 地域
  [5] 中国的少数民族
  [6] 种族文化
  [7] 性别
  [8] 性傾向
  [9] 身心障碍
  [0] 职业/教育
  [q] 宗教
  [w] 其他

The model’s output format:

  <final_answer>
  {
    "groups": ["4", "7"],
    "tuples": [
      ["地域", "河北女孩", "都不漂亮"],
      ["性别", "河北女孩", "都不漂亮"]
    ]
  }
  </final_answer>

Only "groups" is scored. "tuples" is for analysis.

Weighted Jaccard-like score:
  +1 for each true positive group
  -3 for each false positive (except "w")
  0 for missed groups
then normalized to [0, 1] for GEPA.
"""

import json
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import dspy
from dspy import GEPA
from pathlib import Path

# =========================
# 0. Debug / logging config
# =========================


GEPA_LOG_DIR = Path("./gepa_logs/bias_gepa_run_v1")
GEPA_LOG_DIR.mkdir(parents=True, exist_ok=True)


DEBUG = True  # <<< flip this to False to silence debug noise

LOG_LEVEL =logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
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
# 2. DSPy Signature & Program
# =========================

class BiasJSONSignature(dspy.Signature):
    """
    你是一个中文偏见检测与抽取助手。

    输入格式：
      文本: <TEXT>

    任务：
      1. 识别文本中是否针对以下任一“受保护群体”进行评价（尤其是负面评价或质疑）：
         [4] 地域: 针对某个地区的人（“中国人”“美国人”“广东人”“上海人”“欧洲人”“日本人”“小日本”等），
             注意必须是“人”，不是国家/政府/政策本身。
         [5] 中国的少数民族: 回族、白族、藏族、维吾尔族、壮族、苗族等。
         [6] 种族文化: 非中国范围的种族或族群，如“白人”“黑人”“原住民”等。
         [7] 性别: 男人、女人、男性、女性、女孩、男孩、非二元性别者等。
         [8] 性傾向: LGBTQ+ 或性取向相关群体。
         [9] 身心障碍: 身体或精神残障人群。
         [0] 职业/教育: 以职业、教育程度为依据攻击的人群。
         [q] 宗教: 以宗教信仰划分的人群。
         [w] 其他: 仅攻击国家、政府、政策、抽象概念、事件，或无法归入以上任一类时使用。

      2. 输出 JSON（只保留两个字段）：
         {
           "groups": [...],   // 例如 ["4", "7"]
           "tuples": [
             ["类别中文名", "被指向的群体短语", "对该群体的负面或质疑性描述"],
             ...
           ]
         }

         示例1：
           文本: 为什么我们河北女孩都不漂亮，人家湘妹子还有川妹子都长的好可爱，是?
           期望：
           {
             "groups": ["4", "7"],
             "tuples": [
               ["地域", "河北女孩", "都不漂亮"],
               ["性别", "河北女孩", "都不漂亮"]
             ]
           }

         示例2：
           文本: 男人最{会]看不起哪种女人{女孩]？
           理想 tuples 中包含：
           [
             ["性别", "男人", "最{会]看不起哪种女人{女孩]"],
             ["性别", "女人{女孩]", "男人最{会]看不起"]
           ]

      3. 严格输出格式：
         - 第一行输出：<final_answer>
         - 中间输出合法 JSON，仅包含 "groups" 和 "tuples" 字段
         - 最后一行输出：</final_answer>

      注意：
        - "groups" 中只放编号字符串，如 ["4", "7"]。
        - "tuples" 用于分析，不会参与评分，但请尽量合理。
    """

    text: str = dspy.InputField(desc="带前缀 '文本: ' 的中文句子。")
    raw_output: str = dspy.OutputField(
        desc="必须是 <final_answer> 包裹的 JSON 字符串，包含 'groups' 与 'tuples'。"
    )


class BiasProgram(dspy.Module):
    """A small module: one ChainOfThought call from text -> JSON string."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(BiasJSONSignature)

    def forward(self, text: str):
        result = self.predictor(text=text)
        return dspy.Prediction(raw_output=result.raw_output)


# =========================
# 3. Utils: parsing + scoring
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
        # No tags: assume whole string is JSON
        return s.strip()

    inner = s[start + len(start_token):end]
    return inner.strip()


def parse_predicted_groups(pred: dspy.Prediction) -> Tuple[List[str], bool, str]:
    """
    Given a prediction with raw_output, return:
      groups: list[str],
      ok: bool,
      error_msg: str
    """
    output_str = getattr(pred, "raw_output", "")

    if not isinstance(output_str, str) or not output_str.strip():
        return [], False, "raw_output is empty or not a string."

    json_str = strip_final_answer_tags(output_str)

    try:
        obj = json.loads(json_str)
    except Exception as e:
        return [], False, f"Failed to parse JSON: {e}"

    if not isinstance(obj, dict):
        return [], False, "Parsed JSON is not an object."

    groups = obj.get("groups", [])
    if not isinstance(groups, list):
        return [], False, "'groups' is not a list."

    groups = [str(g).strip() for g in groups if str(g).strip()]
    return groups, True, ""

def remap_score(
    s: float,
    baseline: float,
    s_min: float = 0.0,
) -> float:
    """
    Remap old score s ∈ [s_min, 1] to new score in [0, 1] so that:
      - s = s_min   → 0
      - s = baseline → 0.5
      - s = 1       → 1
    """
    # Safety
    if baseline <= s_min:
        raise ValueError("baseline must be > s_min")
    if baseline >= 1.0:
        raise ValueError("baseline must be < 1.0")

    if s <= s_min:
        new = 0.0
    elif s < baseline:
        # map [s_min, baseline] → [0, 0.5]
        new = 0.5 * (s - s_min) / (baseline - s_min)
    elif s <= 1.0:
        # map [baseline, 1] → [0.5, 1]
        new = 0.5 + 0.5 * (s - baseline) / (1.0 - baseline)
    else:
        new = 1.0

    # clamp
    if new < 0.0:
        new = 0.0
    elif new > 1.0:
        new = 1.0

    return new

def weighted_jaccard_score(gold_groups: List[str], pred_groups: List[str]) -> float:
    """
    Weighted Jaccard-like score:

      +1 for each correctly predicted group (gold ∩ pred)
      -3 for each false positive group (pred - gold), EXCEPT 'w'
      0 for missed true groups

    Then normalized to [0, 1] using theoretical min/max bounds.
    """
    gold_set = set(gold_groups)
    pred_set = set(pred_groups)

    correct = len(gold_set & pred_set)
    fp_non_w = len([g for g in (pred_set - gold_set) if g != "w"])

    raw = correct - 3 * fp_non_w

    # Theoretical bounds:
    n_penalized_labels = len([g for g in ALL_CATEGORY_IDS if g != "w"])
    max_raw = max(len(gold_set), 1)  # if no gold labels, treat best as 1 for normalization
    min_raw = -3 * n_penalized_labels

    if max_raw == min_raw:
        return 1.0 if raw >= max_raw else 0.0

    norm = (raw - min_raw) / (max_raw - min_raw)
    norm = max(0.0, min(1.0, norm))
    
    # Baseline = score for raw=0 (i.e. predicting nothing)
    if max_raw == min_raw:
        baseline_norm = base_norm  # degenerate case, just reuse
    else:
        baseline_norm = (0.0 - min_raw) / (max_raw - min_raw)
        baseline_norm = max(0.0, min(1.0, baseline_norm))
    
    remapped = remap_score(norm, baseline=baseline_norm, s_min=0.0)
    return remapped


def metric_with_feedback(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
):
    """
    GEPA-compatible metric:

    - example["answer"] is a JSON like {"groups": ["4","7"]}.
    - pred.raw_output is <final_answer>...JSON...<final_answer>.
    - Returns scalar score in [0,1] when pred_name is None.
    - Returns dspy.Prediction(score, feedback) when pred_name is not None.
    """
    # Parse gold groups
    try:
        gold_obj = json.loads(example["answer"])
        gold_groups = [str(g) for g in gold_obj.get("groups", [])]
    except Exception:
        logger.warning("Gold standard parse issue: %s", example["answer"])
        gold_groups = []

    pred_groups, ok, err = parse_predicted_groups(pred)

    if not ok:
        logger.warning("Prediction parse failed. Example: %s | Pred: %s", example, pred)
        score = 0.0
        feedback_text = (
            f"输出格式错误，无法解析 JSON：{err}。\n"
            "请严格按照以下格式输出：\n"
            "<final_answer>\n"
            "{\n"
            '  \"groups\": [\"7\"],\n'
            '  \"tuples\": [[\"性别\", \"男人\", \"最{会]看不起哪种女人{女孩]\"]]\n'
            "}\n"
            "</final_answer>\n"
        )
    else:
        score = weighted_jaccard_score(gold_groups, pred_groups)
        gold_set = set(gold_groups)
        pred_set = set(pred_groups)

        correct = sorted(list(gold_set & pred_set))
        missing = sorted(list(gold_set - pred_set))
        extra = sorted(list(pred_set - gold_set))

        # Only dump the really verbose stuff when debugging, and only for low scores
        if DEBUG and score < 0.4:
            logger.warning(
                "Low score %.3f\nExample: %s\nPred: %s\nMissing: %s\nExtra: %s",
                score,
                example,
                pred,
                missing,
                extra,
            )

        lines = [
            f"Gold groups: {gold_groups}",
            f"Predicted groups: {pred_groups}",
            f"Correct: {correct}",
        ]
        if missing:
            lines.append(f"Missing groups (should be present): {missing}.")
        if extra:
            fp_non_w = [g for g in extra if g != 'w']
            if fp_non_w:
                lines.append(
                    f"False positives (penalized -3 each): {fp_non_w}."
                )
            fp_w = [g for g in extra if g == 'w']
            if fp_w:
                lines.append(
                    f"Extra 'w' labels (not penalized, but unnecessary): {fp_w}."
                )

        lines.append(f"Weighted-Jaccard-like score: {score:.3f}")
        feedback_text = "\n".join(lines)

    # If called without pred_name (e.g. dspy.Evaluate), just return score
    if pred_name is None:
        return score

    # If called with pred_name (GEPA), return score + feedback
    return dspy.Prediction(score=score, feedback=feedback_text)


# =========================
# 4. Dataset builder from project + topics files
# =========================

@dataclass
class DatasetSplit:
    train: List[dspy.Example]
    val: List[dspy.Example]
    test: List[dspy.Example]


def build_topic_dataframe_from_files(
    project_csv_path: str,
    topics_tsv_path: str,
) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
      - idx
      - text  (from project 'q')
      - groups: list of topic IDs as strings, e.g. ["4","7"]

    project_csv_path: project-4 CSV, tab-separated.
    topics_tsv_path: predicted_topics TSV, tab-separated, with columns:
        ['uncertainty', 'topics_地域', 'topics_少数民族', ..., 'topics_其他', 'idx']
    """
    # Load project file (it's actually TSV with quoted strings)
    df_proj = pd.read_csv(project_csv_path, sep="\t")
    df_topics = pd.read_csv(topics_tsv_path, sep="\t")

    logger.info("Loaded project file with %d rows", len(df_proj))
    logger.info("Loaded topics file with %d rows", len(df_topics))

    # Mapping from topics_* columns to our category IDs
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

    for _, r in df_topics.iterrows():
        idx = str(r["idx"])
        # Grab first matching text row in project file
        matches = df_proj[df_proj["idx"].astype(str) == idx]
        if matches.empty:
            logger.debug("No matching text row for idx=%s; skipping.", idx)
            continue

        q_text = str(matches.iloc[0]["q"])

        c_text = str(matches.iloc[0]["c"])
        if not len(q_text):
            logger.warning("missing question text:",q_text,c_text)
        
        text=q_text+c_text
        
        groups: List[str] = []
        for col in topic_cols:
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


def build_dataset_from_project_and_topics(
    project_csv_path: str,
    topics_tsv_path: str,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> DatasetSplit:
    """
    Build train/val/test splits of dspy.Example from the project + topics files.

    Each Example has:
      - text="文本: <text>"
      - answer='{"groups": [...]}'
    with inputs("text").
    """
    df_data = build_topic_dataframe_from_files(project_csv_path, topics_tsv_path)

    examples: List[dspy.Example] = []
    for _, row in df_data.iterrows():
        raw_text = str(row["text"])
        gold_groups = [str(g) for g in row["groups"]]

        gold_answer_str = json.dumps({"groups": gold_groups}, ensure_ascii=False)
        ex = dspy.Example(
            text=f"文本: {raw_text}",
            answer=gold_answer_str,
        ).with_inputs("text")

        examples.append(ex)

    rng = random.Random(seed)
    rng.shuffle(examples)

    n = len(examples)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    logger.info(
        "Split dataset into train=%d, val=%d, test=%d",
        len(train), len(val), len(test),
    )

    return DatasetSplit(train=train, val=val, test=test)


# =========================
# 5. Main
# =========================

def main():
    # -----------------
    # 5.1 Paths & basic config
    # -----------------
    PROJECT_CSV_PATH = "./project-4-at-2025-12-05-03-06-66edb22a.csv"
    TOPICS_TSV_PATH = "./predicted_topics(1).tsv"

    # GEPA budget: approximate "rollout" / search budget.
    GEPA_MAX_METRIC_CALLS = 512*4  # tweak this for heavier or lighter runs

    # Number of threads for evaluation (parallelism)
    NUM_THREADS = 7

    logger.info("DEBUG=%s", DEBUG)

    # -----------------
    # 5.2 Configure local LLM (your Qwen instance)
    # -----------------
    lm = dspy.LM(
        model="openai/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        api_base="http://localhost:8080/v1",
        api_key="dummy",
        model_type="chat",
        max_tokens=13000,
    )
    dspy.configure(lm=lm)

    # Use the same LM for reflection (you could also choose a different, stronger one)
    reflection_lm = lm

    # -----------------
    # 5.3 Build dataset
    # -----------------
    dataset = build_dataset_from_project_and_topics(
        project_csv_path=PROJECT_CSV_PATH,
        topics_tsv_path=TOPICS_TSV_PATH,
        val_ratio=0.3,
        test_ratio=0.3,
        seed=0,
    )

    logger.info(
        "Dataset sizes: train=%d, val=%d, test=%d",
        len(dataset.train), len(dataset.val), len(dataset.test),
    )

    # -----------------
    # 5.4 Baseline program & evaluation
    # -----------------
    program = BiasProgram()

    evaluate = dspy.Evaluate(
        devset=dataset.test,
        metric=lambda ex, pred, trace=None: metric_with_feedback(ex, pred, trace),
        num_threads=NUM_THREADS,
        display_table=False,
        display_progress=True,
    )

    logger.info("[Baseline] Evaluating unoptimized program...")
    baseline_result = evaluate(program)
    logger.info("[Baseline] Average metric: %.4f", baseline_result.score)

    # -----------------
    # 5.5 GEPA optimization
    # -----------------
    optimizer = GEPA(
        metric=metric_with_feedback,
        max_metric_calls=GEPA_MAX_METRIC_CALLS,
        reflection_lm=reflection_lm,
        num_threads=NUM_THREADS,
        failure_score=0.0,
        perfect_score=1.0,
        track_stats=True,          # <-- get detailed_results
        track_best_outputs=True,   # <-- keep best outputs on valset
        use_merge=True,
        log_dir=str(GEPA_LOG_DIR), # <-- checkpoint + resume
    )

    logger.info("[GEPA] Starting optimization...")
    optimized_program = optimizer.compile(
        program,
        trainset=dataset.train,
        valset=dataset.val,
    )
    logger.info("[GEPA] Optimization complete.")

    # Evaluate optimized program
    logger.info("[GEPA] Evaluating optimized program...")
    optimized_result = evaluate(optimized_program)
    logger.info("[GEPA] Optimized average metric: %.4f", optimized_result.score)

    # -----------------
    # 5.6 Demo example
    # -----------------
    demo_text = "男人最{会]看不起哪种女人{女孩]？"
    demo_input = f"文本: {demo_text}"

    logger.info("=== Demo Example ===")
    logger.info("Input: %s", demo_input)

    demo_pred = optimized_program(text=demo_input)

    logger.info("Raw LLM output:\n%s", demo_pred.raw_output)

    inner_json = strip_final_answer_tags(demo_pred.raw_output)
    try:
        obj = json.loads(inner_json)
        logger.info("Parsed JSON:\n%s", json.dumps(obj, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning("Failed to parse JSON from demo output: %s", e)

    # Ideal target-style tuples for reference (not used in scoring):
    ideal_tuples = [
        ["性别", "男人", "最{会]看不起哪种女人{女孩]"],
        ["性别", "女人{女孩]", "男人最{会]看不起"],
    ]
    logger.info(
        "Ideal tuples for this sentence (target style):\n%s",
        json.dumps(ideal_tuples, ensure_ascii=False, indent=2),
    )


if __name__ == "__main__":
    main()
