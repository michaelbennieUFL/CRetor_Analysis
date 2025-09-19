#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

def load_records(path: Path):
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except Exception as e:
            sys.exit(f"failed to parse json array: {e}")
    rows = []
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            print(f"skip line {i}: {e}", file=sys.stderr)
    return rows

def main():
    ap = argparse.ArgumentParser(description="Make TSV for Non-harmful==0")
    ap.add_argument("--input", required=True, help="path to json or jsonl")
    ap.add_argument("--output", required=True, help="path to tsv")
    args = ap.parse_args()

    recs = load_records(Path(args.input))
    if not isinstance(recs, list):
        sys.exit("input must decode to a list of records")

    rows = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        if r.get("label", None) == 0:
            q = r.get("text", "")
            if not isinstance(q, str):
                q = str(q)
            rows.append({"Question": q, "Potentially_Pejorative": "None"})

    df = pd.DataFrame(rows, columns=["Question", "Potentially_Pejorative"])
    df.to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()
