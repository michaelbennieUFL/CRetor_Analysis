import csv
import os
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

# --- I/O paths ---
input_file = "../../data/biasLabeling/training/combined_fix_baike_STATE_3_ChineseHarm_ToxiCN_MM.tsv"
output_file = "../../data/biasLabeling/training/out/combined_fix_baike_STATE_3_ChineseHarm_ToxiCN_MM_omni_scores.tsv"

# --- Moderation categories to extract ---
category_fields = [
    "sexual", "sexual/minors", "harassment", "harassment/threatening",
    "hate", "hate/threatening", "illicit", "illicit/violent",
    "self-harm", "self-harm/intent", "self-harm/instructions",
    "violence", "violence/graphic"
]

# --- Threading + retry config ---
MAX_WORKERS = 6
MAX_RETRIES = 5
RETRY_SLEEP_SECONDS = 15


def get_text_from_row(row: dict) -> str:
    """
    Pick the text to moderate from a row.
    Adjust if your TSV has a specific column name (e.g., 'Question', 'text', etc.).
    """
    return row.get("text") or row.get("content") or row.get("Question") or next(iter(row.values()), "")


def moderate_row(index: int, row: dict) -> tuple[int, dict]:
    """
    Worker that calls the moderation API for one row with retries.
    Returns (index, updated_row_dict).
    If all retries fail, returns the row with None scores populated.
    """
    client = OpenAI(api_key=API_KEY)

    text = get_text_from_row(row)
    updated = deepcopy(row)

    # Pre-fill with None so the output schema is consistent even on failure/empty text
    for cat in category_fields:
        updated[cat] = None

    if not text:
        return index, updated  # nothing to do
    current_model="omni-moderation-latest"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if current_model == "omni-moderation-latest":
                current_model="omni-moderation-2024-09-26"
            elif current_model == "omni-moderation-2024-09-26":
                current_model="omni-moderation-latest"
            resp = client.moderations.create(
                model=current_model,
                input=text
            )
            result = resp.results[0]
            scores = result.category_scores.model_dump()

            for cat in category_fields:
                updated[cat] = scores.get(cat, None)

            return index, updated

        except Exception as e:
            if attempt < MAX_RETRIES:
                # wait then retry
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                # All retries failed; keep None scores and attach an error note if you want
                # updated["moderation_error"] = str(e)  # uncomment if you'd like an error column
                return index, updated


def main():
    # --- Read input TSV ---
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        base_fields = reader.fieldnames or []

    print(f"Loaded {len(rows)} rows from {input_file}")

    # --- Prepare output schema ---
    output_fields = base_fields + [c for c in category_fields if c not in base_fields]

    # --- Dispatch threaded work ---
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(moderate_row, i, row): i for i, row in enumerate(rows)}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Moderating"):
            idx, updated_row = fut.result()
            results[idx] = updated_row

    # --- Write results in the original order ---
    with open(output_file, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fields, delimiter="\t")
        writer.writeheader()
        for i in range(len(rows)):
            writer.writerow(results[i])

    print(f"✅ Moderation results saved to {output_file}")


if __name__ == "__main__":
    main()
