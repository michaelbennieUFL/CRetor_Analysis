#!/usr/bin/env python3
import json
import re
import sys
from typing import Dict, Any, List

import pandas as pd
from pathlib import Path

def detect_potentially_pejorative(item: Dict[str, Any]) -> str:
    """
    Return "Potentially" if ANY field matching regex r'^Q\\d+\\s*hateful$' has value 'hate' (case-insensitive),
    otherwise return "None".
    """
    for k, v in item.items():
        if re.fullmatch(r"Q\d+\s*hateful", k):
            if isinstance(v, str) and v.strip().lower() == "hate":
                return "Potentially"
    return "None"

def qualifies_prune(item: Dict[str, Any]) -> bool:
    """
    Keep an item if:
      1. It does not contain 'NULL' in ANY of its values
      2. There exists SOME Q# such that:
         - Q# hateful == 'hate'
         - Q# Group exists AND DOES NOT contain the token 'others'
    """
    # 🔴 Step 1: Drop items with 'NULL' anywhere in their values
    for val in item.values():
        if isinstance(val, str) and val.strip().upper() == "NULL":
            return False

    # 🔴 Step 2: Apply the original logic
    for qnum in (1, 2, 3):
        hateful_key = f"Q{qnum} hateful"
        group_key = f"Q{qnum} Group"
        hateful_val = item.get(hateful_key)
        group_val = item.get(group_key)

        if isinstance(hateful_val, str) and hateful_val.strip().lower() == "hate" and isinstance(group_val, str):
            tokens = [t.strip().lower() for t in group_val.split(",")]
            if "others" not in tokens:
                return True
    return False


def load_json_records(path: str) -> List[Dict[str, Any]]:
    """
    Load records from a JSON file that may be either:
      - a JSON array of objects
      - newline-delimited JSON (JSONL)
    Returns [] if the path doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        return []

    with p.open("r", encoding="utf-8") as f:
        text = f.read().strip()
        # Try JSON array first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass

    # Fallback: JSONL
    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records

def records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in records:
        question = item.get("content", "")
        label = detect_potentially_pejorative(item)
        rows.append({"Question": question, "Potentially_Pejorative": label})
    return pd.DataFrame(rows, columns=["Question", "Potentially_Pejorative"])

def main():
    test_path, train_path = "test.json", "train.json"

    test_records = load_json_records(test_path)
    train_records = load_json_records(train_path)

    # NEW: prune by the requested rule
    pruned_test = [r for r in test_records if qualifies_prune(r)]
    pruned_train = [r for r in train_records if qualifies_prune(r)]

    df_test = records_to_df(pruned_test)
    df_train = records_to_df(pruned_train)

    df = pd.concat([df_train, df_test], ignore_index=True)
    output_path = "STATE-Combined.tsv"
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {output_path}")

if __name__ == "__main__":
    main()
