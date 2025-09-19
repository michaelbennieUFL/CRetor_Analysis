import json
import pandas as pd
import re
import unicodedata
from pathlib import Path

# ========= Paths =========
LEXICON_PATH = Path("annotated lexicon.json")
TSV_PATH     = Path("combined_fix_baike_STATE_3_ChineseHarm_ToxiCN_MM.tsv")
OUTPUT_JSON  = Path("filtered_lexicon.json")

# ========= Load data =========
with LEXICON_PATH.open("r", encoding="utf-8") as f:
    lexicon = json.load(f)

# TSV has two columns: sentence, Potentially_Pejorative
tsv = pd.read_csv(
    TSV_PATH,
    sep="\t",
    header=None,
    names=["sentence", "Potentially_Pejorative"],
    dtype=str,
    keep_default_na=False,  # keep "None" as a literal string
)

# Drop any accidental header-like first row
mask_headerish = (
    (tsv["sentence"].str.strip().str.lower() == "question")
    | (tsv["Potentially_Pejorative"].str.strip() == "Potentially_Pejorative")
)
tsv = tsv[~mask_headerish].reset_index(drop=True)

# Normalize whitespace just a bit
tsv["sentence"] = tsv["sentence"].astype(str).str.replace("\u3000", " ", regex=False).str.strip()
tsv["Potentially_Pejorative"] = tsv["Potentially_Pejorative"].astype(str).str.strip()

terms = lexicon.get("terms", [])

# ========= Utilities =========
_SHORT_LATIN_RE = re.compile(r"^[A-Za-z]{1,2}$")

def is_short_latin_only(term: str) -> bool:
    """
    True if the term is composed ONLY of 1–2 ASCII letters (A–Z/a–z).
    This excludes any Chinese/mixed strings and anything length >= 4.
    """
    term = (term or "").strip()
    return bool(_SHORT_LATIN_RE.fullmatch(term))

def nfkc_lower(s: str) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", str(s)).lower()

def definition_has_zaimouxie(t: dict) -> bool:
    """
    Return True if the term's definition contains '在某些' (after NFKC normalization).
    """
    return "在某些" in nfkc_lower(t.get("definition", "")) or "原本" in nfkc_lower(t.get("definition", "")) or "本意" in nfkc_lower(t.get("definition", "")) or "本为" in nfkc_lower(t.get("definition", ""))  or "多数" in nfkc_lower(t.get("definition", ""))  or "通称" in nfkc_lower(t.get("definition", "")) or "调侃" in nfkc_lower(t.get("definition", ""))

def in_any_sentence(term: str, sentences: pd.Series) -> bool:
    """Check if a term appears as a substring in any sentence (literal match, not regex)."""
    if not term:
        return False
    return sentences.astype(str).str.contains(term, na=False, regex=False).any()

def contains_lexicon_term(sentence: str, term_list) -> bool:
    s = sentence or ""
    # simple literal substring check
    return any(t["term"] in s for t in term_list)

# ========= Step 1: Remove terms that occur in any sentence labeled "None" =========
none_sentences = tsv.loc[tsv["Potentially_Pejorative"] == "None", "sentence"]
terms_step1 = [
    t for t in terms
    if not in_any_sentence(t.get("term", ""), none_sentences)
]

# ========= Step 2: Remove single-character terms =========
terms_step2 = [
    t for t in terms_step1
    if len((t.get("term") or "").strip()) > 1
]

# ========= Step 2b: Remove terms that are ONLY 1–2 Latin letters (eg, YP, abc) =========
# (Keeps terms with Chinese/mixed chars or length >= 3)
terms_step2b = [
    t for t in terms_step2
    if not is_short_latin_only(t.get("term", ""))
]

# ========= Step 2c: Remove terms whose definition contains "在某些" =========
terms_step2c = [
    t for t in terms_step2b
    if not definition_has_zaimouxie(t)
]

# ========= Step 3: must appear in at least X TSV sentences =========
MIN_SENTENCE_HITS = 3

all_sentences = tsv["sentence"].astype(str)

def sentence_hits(term: str, sentences: pd.Series) -> int:
    if not term:
        return 0
    return int(sentences.str.contains(term, na=False, regex=False).sum())

# 預先把每個 term 的句子命中數算好
term2hits = {
    t.get("term", ""): sentence_hits(t.get("term", ""), all_sentences)
    for t in terms_step2c
}

terms_step3 = [
    t for t in terms_step2c
    if term2hits.get(t.get("term", ""), 0) >= MIN_SENTENCE_HITS
]

# ========= Step 4: remove  other items (category contains 'other') =========
terms_step4 = [
    t for t in terms_step3
    if "other" not in str(t.get("category", "")).lower()
]

# Update lexicon object
cleaned_lexicon = {
    **{k: v for k, v in lexicon.items() if k != "terms"},
    "terms": terms_step4,
}

# ========= Save cleaned lexicon =========
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(cleaned_lexicon, f, ensure_ascii=False, indent=2)

# ========= Step 5: Percentages for "None" and "Potentially" lines that contain >=1 term =========
none_df = tsv[tsv["Potentially_Pejorative"] == "None"]
pot_df  = tsv[tsv["Potentially_Pejorative"] == "Potentially"]

def pct_with_term(df: pd.DataFrame, term_list) -> float:
    if df.empty:
        return 0.0
    hits = df["sentence"].astype(str).apply(lambda s: contains_lexicon_term(s, term_list)).sum()
    return 100.0 * hits / len(df)

none_pct = pct_with_term(none_df, terms_step4)
pot_pct  = pct_with_term(pot_df, terms_step4)

# ========= Console report =========
print("=== Lexicon Cleaning Report ===")
print(f"Original terms: {len(terms)}")
print(f"After Step 1 (remove if occurs in 'None' lines)    : {len(terms_step1)}")
print(f"After Step 2 (remove single-character terms)       : {len(terms_step2)}")
print(f"After Step 2b (remove <=3-letter Latin only)       : {len(terms_step2b)}")
print(f"After Step 2c (remove defs containing '在某些/本')     : {len(terms_step2c)}")
print(f"After Step 3 (must appear in TSV sentences)        : {len(terms_step3)}")
print(f"After Step 4 (drop 'other' categories)             : {len(terms_step4)}")
print()
print("=== Coverage (contains ≥1 term) ===")
print(f"'None' lines: {none_pct:.2f}%")
print(f"'Potentially' lines: {pot_pct:.2f}%")
print()
print(f"Cleaned lexicon saved to: {OUTPUT_JSON}")
