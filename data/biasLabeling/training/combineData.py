#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge combined_fix6.tsv with a new TSV (e.g., baike predictions).

- New file has: Question, desc, Potentially_Pejorative (TRUE/FALSE), self_confidence
- Output: combined_fix_baike_1.tsv
    Columns: Question, Potentially_Pejorative
    - "Question" is Question + " " + desc (trimmed)
    - Potentially_Pejorative: TRUE -> Potentially, FALSE -> None
"""

import pandas as pd
import argparse
import os


def load_train(path: str) -> pd.DataFrame:
    """Load the existing combined_fix6.tsv"""
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "Question" not in df or "Potentially_Pejorative" not in df:
        raise ValueError("combined_fix6.tsv must have columns: Question, Potentially_Pejorative")
    return df[["Question", "Potentially_Pejorative"]]


def load_new(path: str) -> pd.DataFrame:
    """Load the new TSV and convert to the training format"""
    df = pd.read_csv(path, sep="\t", dtype=str)

    # Build combined Question
    q = df["Question"].fillna("").astype(str).str.strip()
    d = df["desc"].fillna("").astype(str).str.strip()
    df["Question"] = (q + " " + d).str.strip()

    # Map labels TRUE/FALSE -> Potentially/None
    df["Potentially_Pejorative"] = (
        df["Potentially_Pejorative"].str.upper().map({"TRUE": "Potentially", "FALSE": "None"})
    )

    return df[["Question", "Potentially_Pejorative"]]


def main():
    parser = argparse.ArgumentParser(description="Merge baike TSV into combined_fix6.tsv")
    parser.add_argument("--train", default="combined_fix6.tsv", help="Path to combined_fix6.tsv")
    parser.add_argument("--new", default="baike_valid_fix_1.tsv", help="Path to baike TSV")
    parser.add_argument("--out", default="combined_fix_baike_1.tsv", help="Output file path")
    args = parser.parse_args()

    df_train = load_train(args.train)
    df_new = load_new(args.new)

    combined = pd.concat([df_train, df_new], ignore_index=True)

    combined["Potentially_Pejorative"] = (
        combined["Potentially_Pejorative"].fillna("None").replace(r"^\s*$", "None", regex=True)
    )
    combined.to_csv(args.out, sep="\t", index=False, encoding="utf-8", na_rep="None")

    print(f"Wrote {args.out} with {len(combined)} rows.")


if __name__ == "__main__":
    main()
