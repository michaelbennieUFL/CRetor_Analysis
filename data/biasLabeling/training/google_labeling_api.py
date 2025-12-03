import csv
import json
import subprocess
import sys
import time
from typing import Dict, List
from tqdm import tqdm
API_URL = "https://language.googleapis.com/v1/documents:moderateText"

def get_token() -> str:
    try:
        out = subprocess.check_output(
            ["gcloud", "auth", "application-default", "print-access-token"],
            text=True
        ).strip()
        if not out:
            raise RuntimeError("empty token")
        return out
    except Exception as e:
        print(f"error getting access token: {e}", file=sys.stderr)
        sys.exit(1)

def norm_col(name: str) -> str:
    s = name.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")
    return f"{s}_confidence"

def call_api(token: str, project: str, text: str) -> Dict[str, float]:
    import urllib.request
    req = urllib.request.Request(API_URL, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    req.add_header("x-goog-user-project", project)
    body = {"document": {"type": "PLAIN_TEXT", "content": text}}
    data = json.dumps(body).encode("utf-8")
    with urllib.request.urlopen(req, data=data, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    out: Dict[str, float] = {}
    for c in payload.get("moderationCategories", []) or []:
        out[norm_col(c.get("name", "Unknown"))] = c.get("confidence", 0.0)
    return out

def run_moderation_pipeline(
    input_path: str,
    output_path: str,
    project_id: str,
    sleep_seconds: float = 0.0
) -> None:
    token = get_token()

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows: List[Dict[str, str]] = list(reader)

    if not rows:
        print("no rows", file=sys.stderr)
        sys.exit(1)

    base_cols = ["Question", "Potentially_Pejorative"]
    for r in tqdm(rows):
        if "Question" not in r:
            print("missing column Question", file=sys.stderr)
            sys.exit(1)
        val = r.get("Potentially_Pejorative", "")
        if val is None or str(val).strip() == "" or str(val).strip().lower() == "nan":
            r["Potentially_Pejorative"] = "None"
        else:
            r["Potentially_Pejorative"] = str(val)

    cat_cols_order: List[str] = []
    processed: List[Dict[str, str]] = []

    for idx, r in tqdm(enumerate(rows, 1), total=len(rows)):
        q = r["Question"]
        try:
            cats = call_api(token, project_id, q)
        except Exception as e:
            print(f"row {idx} api failed: {e}", file=sys.stderr)
            cats = {}

        for k in cats.keys():
            if k not in cat_cols_order:
                cat_cols_order.append(k)

        out_row: Dict[str, str] = {"Question": q, "Potentially_Pejorative": r["Potentially_Pejorative"]}
        for k in cat_cols_order:
            out_row[k] = ""
        for k, v in cats.items():
            out_row[k] = str(v)
        processed.append(out_row)

        if sleep_seconds and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    header = base_cols + cat_cols_order
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in processed:
            for k in cat_cols_order:
                if k not in r:
                    r[k] = ""
            writer.writerow(r)

if __name__ == "__main__":
    # 在此填入參數
    INPUT_PATH = "combined_fix_baike_STATE_3.tsv"
    OUTPUT_PATH = "combined_fix_baike_STATE_3_moderated.tsv"
    PROJECT_ID = "gen-lang-client-0949876192"
    SLEEP_SECONDS = 0.0

    run_moderation_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        project_id=PROJECT_ID,
        sleep_seconds=SLEEP_SECONDS
    )
