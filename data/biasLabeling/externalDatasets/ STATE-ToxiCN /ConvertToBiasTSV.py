#!/usr/bin/env python3
import json
import re
import sys
from typing import Dict, Any, List

import pandas as pd

def detect_potentially_pejorative(item: Dict[str, Any]) -> str:
    """
    Return "Potentially" if ANY field matching regex r'^Q\\d+\\s*hateful$' has value 'hate' (case-insensitive),
    otherwise return "None".
    """
    for k, v in item.items():
        if "hateful" in k:
            if isinstance(v, str) and v.strip().lower() == "hate":
                return "Potentially"
    return "None"

def load_json_records(path: str) -> List[Dict[str, Any]]:
    """
    Load records from a JSON file that may be either:
      - a JSON array of objects
      - newline-delimited JSON (JSONL)
    """
    with open(path, "r", encoding="utf-8") as f:
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

    # Fallback: JSON Lines
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                # Skip malformed lines rather than failing the whole conversion
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

    df_test = records_to_df(test_records)
    df_train = records_to_df(train_records)

    df = pd.concat([df_train, df_test], ignore_index=True)
    output_path = "STATE-Combined.tsv"
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {output_path}")

if __name__ == "__main__":
    main()