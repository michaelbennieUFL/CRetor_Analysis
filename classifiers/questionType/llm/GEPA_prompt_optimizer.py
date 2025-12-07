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
import json_repair
import pandas as pd
import dspy
from dspy import GEPA
from pathlib import Path

# =========================
# 0. Debug / logging config
# =========================


GEPA_LOG_DIR = Path("./gepa_logs/bias_gepa_run_v4")
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
    你是一个中文偏见检测与抽取助手，需严格按以下规则处理输入文本：

    ### 输入格式
    文本: <TEXT>

    ### 任务
    1. **识别所有受保护群体提及**（无论是否含评价描述）：
       - 对每个提及的群体短语，判断其所属类别（仅限以下编号类别），并将对应编号加入`groups`。
       - **关键规则**：群体提及即标记，即使无评价描述也必须加入`groups`。

    2. **生成评价元组**（仅当存在直接评价描述时）：
       - 仅当文本包含**直接针对群体特征的评价描述**（如“皮肤黑”“都不漂亮”“水平差”）时，生成tuple。
       - 无评价描述时`tuples`为空列表。

    3. **输出JSON格式**（仅含两个字段）：
       ```json
       {
         "groups": ["类别编号字符串列表"],
         "tuples": [
           ["类别中文名", "被指向的群体短语", "评价描述"],
           ...
         ]
       }
       ```

    ---

    ### 类别定义与识别规则（必须严格遵守）
    #### [4] 地域（仅限“人”）
    - **定义**：针对某个地区的人群（如“中国人”“广东人”“上海人”“小日本”），必须是具体人群（非国家/政府/政策/抽象概念）。
    - **关键规则**：
      - 地区名称（如“浙江”“三亚”）**仅当上下文隐含指向人群时**才归类为[4]（例：“浙江会下落到第二梯队”指浙江人 → [4]；“浙江风景美”不归类）。
      - **中国范围内的地区人群**（如“中国人”“日本人”）属于[4]；**非中国范围的种族**（如“亚裔”“白人”“黑人”）属于[6]。
      - **多成分短语处理**：若短语包含多个属性（如“中国男子足球队”），需拆解为：
        - “中国” → [4]（地域）
        - “男子” → [7]（性别）
        - “足球队” → [0]（职业）
        - 所有适用类别均需加入`groups`。

    #### [5] 中国少数民族
    - **定义**：回族、白族、藏族、维吾尔族、壮族、苗族等中国法定少数民族。
    - **示例**：“回族姑娘” → [5]（回族）和[7]（姑娘）。

    #### [6] 种族文化（非中国范围）
    - **定义**：非中国范围的种族或族群（如“白人”“黑人”“原住民”“亚裔”）。
    - **关键规则**：
      - “亚裔”属于[6]（即使在中国语境中）。
      - “中国人”“日本人”属于[4]，**非[6]**。

    #### [7] 性别
    - **定义**：男人、女人、男性、女性、女孩、男孩、非二元性别者、男士、女士、姑娘、妇人等。
    - **关键规则**：
      - “处女”“非处女”**不属于任何类别**（仅性行为状态属性，非群体）。
      - “老太”“老头”属于[7]（性别）和[w]（其他）。

    #### [8] 性倾向
    - **定义**：LGBTQ+或性取向相关群体（如“女同性恋”“同性恋”“双性恋”）。

    #### [9] 身心障碍
    - **定义**：身体或精神残障人群（如“残疾人”“自杀者”“盲人”）。
    - **关键规则**：
      - “自杀者”**必须归类为[9]**（即使字面无“残障”）。
      - “残疾人比赛”中的“残疾人”属于[9]。

    #### [0] 职业/教育
    - **定义**：以职业、教育程度为依据评价的人群（如“大学生”“农民工”“护士”“足球队”“教师”）。
    - **关键规则**：
      - **仅提及职业群体时**（如“大学生”“足球队”），必须将[0]加入`groups`（无论是否有评价）。
      - **仅当存在直接评价描述时**才生成tuple（例：“大学生懒惰” → tuple；“大学生”仅提及 → 无tuple）。

    #### [q] 宗教
    - **定义**：以宗教信仰划分的人群（如“穆斯林”“基督徒”“佛教徒”）。

    #### [w] 其他
    - **定义**：仅攻击国家/政府/政策/抽象概念/事件（如“中国政策”“经济下滑”“日本社会”“政府腐败”），或无法归入以上类别。
    - **关键规则**：
      - “日本社会”“中国政策”“经济下滑” → [w]。
      - “星座”（如“处女座”“天蝎座”）**不属于任何类别**。

    ---

    ### 评价描述生成规则
    1. **直接关联性**：评价必须直接描述群体特征（例）：
       - “三亚女人皮肤黑” → 评价“皮肤黑”针对地域[4]（非性别[7]）。
       - “河北女孩都不漂亮” → 评价“都不漂亮”同时针对地域[4]和性别[7]。
       - “中国男子足球队水平差” → 评价“水平差”针对职业[0]（足球队）。
       - “护士工作辛苦” → 评价“辛苦”针对职业[0]。

    2. **疑问句/陈述句处理**：
       - 仅提问（如“日本人好相处吗？”）**无明确评价描述** → 不生成tuple。
       - 隐含评价（如“这样应该能赢了吧”）**需判断是否直接描述群体特征**：
         - 若隐含“足球队能力差” → 生成tuple ["职业", "足球队", "能力差"]。
         - 若仅为建议（如“纳入残疾人比赛”）无直接评价 → 不生成tuple。

    3. **多群体处理**：
       - 若群体短语同时属于多个类别（如“河北女孩”=地域[4]+性别[7]），为每个适用类别生成独立tuple。
       - 例：“河北女孩都不漂亮” → 
         ```json
         ["地域", "河北女孩", "都不漂亮"],
         ["性别", "河北女孩", "都不漂亮"]
         ```

    ---

    ### 特殊注意事项
    - **“沪漂”**：指从其他地区迁入上海的人群，属于地域[4]（因隐含“上海人”相关群体）。
    - **“中国男子足球队”**：
      - “中国” → [4]
      - “男子” → [7]
      - “足球队” → [0]
      - 所有适用类别均需加入`groups`。
    - **“残疾人比赛”**：
      - “残疾人” → [9]
      - “比赛”本身不触发任何类别。
    - **“老太”“老头”**：同时属于[7]（性别）和[w]（其他），但[w]仅在无其他类别时使用（实际[7]已覆盖）。

    ---

    ### 输出示例（关键场景）
    #### 示例1：多成分短语
    - 文本: "建议国际足联把中国男子足球队纳入残疾人比赛。"
    - 输出:
      ```json
      {
        "groups": ["4", "7", "0", "9"],
        "tuples": []
      }
      ```
      - 理由：  
        - "中国" → [4]  
        - "男子" → [7]  
        - "足球队" → [0]  
        - "残疾人" → [9]  
        - 无直接评价描述（"纳入残疾人比赛"是建议，非评价）→ tuples为空。

    #### 示例2：抽象概念
    - 文本: "为什么日本社会很压抑？"
    - 输出:
      ```json
      {
        "groups": ["w"],
        "tuples": [["其他", "日本社会", "很压抑"]]
      }
      ```
      - 理由：  
        - "日本社会"是抽象概念 → [w]  
        - 评价"很压抑"直接关联 → 生成tuple。

    #### 示例3：性别+职业
    - 文本: "护士工作辛苦，但医生更累。"
    - 输出:
      ```json
      {
        "groups": ["0"],
        "tuples": [
          ["职业", "护士", "工作辛苦"],
          ["职业", "医生", "更累"]
        ]
      }
      ```

    #### 示例4：无评价描述
    - 文本: "大学生和农民工有什么区别？"
    - 输出:
      ```json
      {
        "groups": ["0"],
        "tuples": []
      }
      ```

    #### 示例5：星座不归类
    - 文本: "处女座的人很完美主义。"
    - 输出:
      ```json
      {
        "groups": [],
        "tuples": []
      }
      ```

    ---

    ### 严格禁止
    - 生成JSON外的任何文本（包括`<final_answer>`外的说明）。
    - 将“处女”“星座”“国家/政府/政策”误归类为受保护群体。
    - 仅提及职业/教育群体时生成tuple（必须有评价描述）。
    - 忽略多成分短语的拆解（如“中国男子足球队”必须拆解为[4][7][0]）。
    - 对“自杀者”归类为其他类别（必须为[9]）。
    - 对“亚裔”归类为[4]（必须为[6]）。
    - 对“日本社会”归类为[4]（必须为[w]）。
    - 对“浙江”仅提及地区名称（无人群隐含）时归类为[4]（必须无类别）。

    <final_answer>
    {"groups": [], "tuples": []}
    </final_answer>
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

    # Strip <final_answer> ... </final_answer> wrapper
    json_str = strip_final_answer_tags(output_str).strip()

    try:
        # Option 1: directly get Python object (fastest)
        obj = json_repair.repair_json(
            json_str,
            return_objects=True,   # return Python object instead of string
            ensure_ascii=False,    # keep Chinese characters as-is
        )
    except Exception as e:
        # json_repair couldn't fix it either
        return [], False, f"Failed to repair/parse JSON: {e}"

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
        baseline_norm = 0.95  # degenerate case, just reuse
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
        logger.warning(
            "Prediction parse failed: %s\nExample: %s\nPred: %s",
            err, example, pred
        )
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
        if DEBUG and score < 0.42:
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
        c_raw = matches.iloc[0]["c"]
        c_text = "" if pd.isna(c_raw) else str(c_raw)
        text = q_text + c_text
        if not len(q_text):
            logger.warning("missing question text:",q_text,c_text)
                
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
    TOPICS_TSV_PATH = "./predicted_topics_corrected.tsv"

    # GEPA budget: approximate "rollout" / search budget.
    GEPA_MAX_METRIC_CALLS = 512*4  # tweak this for heavier or lighter runs

    # Number of threads for evaluation (parallelism)
    NUM_THREADS = 10

    logger.info("DEBUG=%s", DEBUG)

    # -----------------
    # 5.2 Configure local LLM (your Qwen instance)
    # -----------------
    lm = dspy.LM(
        model="openai/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        api_base="http://localhost:8080/v1",
        api_key="dummy",
        model_type="chat",
        max_tokens=40000,
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
        seed=3,
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
        auto="heavy",   # or "heavy"
        reflection_lm=reflection_lm,
        num_threads=NUM_THREADS,
        track_stats=True,
        track_best_outputs=True,
        use_merge=True,
        log_dir=str(GEPA_LOG_DIR),
        reflection_minibatch_size=10,
    )


    logger.info("[GEPA] Starting optimization...")
    optimized_program = optimizer.compile(
        program,
        trainset=dataset.train,
        valset=dataset.val,
    )
    logger.info("[GEPA] Optimization complete.")
    
    print("-----------1-1-1-1-1-1-1-1--1-1----------")


    # Grab best program (BiasProgram) from detailed_results if available
    if hasattr(optimized_program, "detailed_results"):
        res = optimized_program.detailed_results
        best_program = res.best_candidate
    else:
        best_program = optimized_program

    # Optional: see full structure
    print(best_program)

    GEPA_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Get the optimized prompt for the predictor.predict primitive
    prompt_text = best_program.predictor.predict.signature.instructions

    prompt_path = GEPA_LOG_DIR / "optimized_prompt_predictor.predict.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    logger.info("Saved optimized prompt to %s", prompt_path)

    print("\n==== OPTIMIZED PROMPT (predictor.predict) ====\n")
    print(prompt_text)
    print("\n=============================================\n")

    logger.info("Saved optimized prompt to %s", prompt_path)
    

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
