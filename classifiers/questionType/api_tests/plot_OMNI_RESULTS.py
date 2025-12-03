import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# === 1) Load ===
file_path = "../../../data/biasLabeling/training/out/combined_fix_baike_STATE_3_ChineseHarm_ToxiCN_MM_omni_scores.tsv"
df = pd.read_csv(file_path, sep="\t")

label_col = "Potentially_Pejorative"   # your ground-truth column
drop_cols = {"Question"}                # any non-score text columns to exclude from plotting

# === 2) Build binary labels from string rules ===
# False if blank/whitespace or "none" (case-insensitive), True otherwise
lab = df[label_col].astype(str).str.strip()
y_true = (~lab.eq("") & ~lab.str.lower().eq("none")).astype(int)

# === 3) Detect score columns (everything except the label and explicit drops) ===
score_cols = [c for c in df.columns if c not in {label_col, *drop_cols}]

print("Score columns detected:", score_cols)

# === 4) Plot ROC curves and compute AUC ===
plt.figure(figsize=(10, 8))
auc_results = {}

for col in score_cols:
    try:
        # coerce scores to float (errors='coerce' turns bad values into NaN)
        y_score = pd.to_numeric(df[col], errors="coerce")

        # drop rows with NaN in score or label
        mask = y_score.notna() & y_true.notna()
        if mask.sum() == 0:
            print(f"Skipping {col}: no valid rows after NaN filtering.")
            continue

        y_t = y_true[mask]
        y_s = y_score[mask]

        # need both classes present for ROC
        if y_t.nunique() < 2:
            print(f"Skipping {col}: y_true has only one class after filtering.")
            continue

        fpr, tpr, _ = roc_curve(y_t, y_s)
        auc = roc_auc_score(y_t, y_s)
        auc_results[col] = auc

        plt.plot(fpr, tpr, label=f"{col} (AUC={auc:.3f})")

    except Exception as e:
        print(f"Skipping {col} due to error: {e}")

# === 5) Formatting ===
plt.plot([0, 1], [0, 1], linestyle="--")  # no explicit colors
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Moderation Categories")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 6) AUC summary ===
print("\n=== AUC Summary ===")
for col, auc in sorted(auc_results.items(), key=lambda x: -x[1]):
    print(f"{col:<35} {auc:.4f}")
