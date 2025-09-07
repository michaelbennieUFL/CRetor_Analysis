import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

from typing import List, Dict
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score

import dspy
from dspy import Example, Prediction
from dspy.teleprompt import COPRO, MIPROv2, GEPA
from dspy.adapters import JSONAdapter, ChatAdapter
from dspy.utils.exceptions import AdapterParseError

# ---------------------------
# Configuration (UFL proxy)
# ---------------------------
UFL_BASE_URL = "https://api.ai.it.ufl.edu"
TASK_MODEL_NAME = "openai/gpt-oss-20b"       # cheaper for task eval
PROMPT_MODEL_NAME = "openai/gpt-oss-120b"    # user requested for proposal/reflection

GROUPS_ZH = [
    "地域", "少数民族", "种族文化", "性别", "性傾向", "身心障碍",
    "职业/教育", "宗教"
]

# ---------------------------
# DSPy signature
# ---------------------------
class BiasPotentialSignature(dspy.Signature):
    """Return exactly 'Potentially' or 'None'."""
    text = dspy.InputField(desc=f"User text (Chinese or mixed). Consider: {', '.join(GROUPS_ZH)}.")
    label = dspy.OutputField(desc="Exactly 'Potentially' or 'None'.")

def configure_lms():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY (env or .env) with your UFL LiteLLM user key.")

    # Task LM: small answers, deterministic
    task_lm = dspy.LM(
        TASK_MODEL_NAME,
        api_key=api_key,
        api_base=UFL_BASE_URL,
        temperature=0.0,
        max_tokens=100,          # short answers for label-only tasks
    )

    # Prompt/Proposal/Reflection LM: roomier budget
    prompt_lm = dspy.LM(
        PROMPT_MODEL_NAME,
        api_key=api_key,
        api_base=UFL_BASE_URL,
        temperature=0.0,         # keep strict for JSON compliance
        max_tokens=2000,         # ample for JSON blobs and reflection
    )

    # Prefer structured outputs globally; we’ll fall back to ChatAdapter when needed.
    dspy.configure(lm=task_lm, adapter=JSONAdapter())

    return task_lm, prompt_lm

# ---------------------------
# Data loading / prep
# ---------------------------
def load_dataset(path: str, text_col="Question", label_col="Potentially_Pejorative"):
    df = pd.read_csv(path, sep="\t")
    df[text_col] = df[text_col].astype(str)

    def map_label(v):
        v_str = str(v).strip()
        if v_str and v_str.lower() not in {"none", "nan"}:
            return "Potentially"
        return "None"

    df["label"] = df[label_col].apply(map_label)
    return df[[text_col, "label"]].rename(columns={text_col: "text"})

def split_train_dev_test(df, test_size=0.2, dev_size=0.2, seed=42):
    train_dev, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    dev_ratio = dev_size / (1.0 - test_size)
    train, dev = train_test_split(train_dev, test_size=dev_ratio, random_state=seed, stratify=train_dev["label"])
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)

def to_dspy_examples(df) -> List[Example]:
    exs: List[Example] = []
    for _, row in df.iterrows():
        exs.append(Example(text=str(row["text"]), label=str(row["label"])).with_inputs("text"))
    return exs

# ---------------------------
# Metrics & helpers
# ---------------------------
def normalize_label(s: str) -> str:
    return "Potentially" if str(s).strip().lower().startswith("potential") else "None"

def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    to_bin = lambda x: 1 if x == "Potentially" else 0
    y_true_b = list(map(to_bin, y_true))
    y_pred_b = list(map(to_bin, y_pred))
    return {
        "accuracy": accuracy_score(y_true_b, y_pred_b),
        "f1": f1_score(y_true_b, y_pred_b, zero_division=0),
        "precision": precision_score(y_true_b, y_pred_b, zero_division=0),
    }

def gold_label(ex) -> str:
    return (getattr(ex, "label", None) or ex["label"]).strip().lower()

def acc_metric(ex, pred, trace=None):
    gold = gold_label(ex)
    got = normalize_label(getattr(pred, "label", "None"))
    return 1.0 if got.lower() == gold else 0.0

# GEPA metric: float at program level; Prediction(score, feedback) at predictor level
def gepa_metric(ex, pred, trace=None, pred_name=None, pred_trace=None):
    gold = gold_label(ex)
    got_raw = getattr(pred, "label", "None")
    got = normalize_label(got_raw)
    score = 1.0 if got.lower() == gold else 0.0

    if pred_name is None:
        return float(score)

    parts = [f"Gold: {gold}. Pred: {got}.", f"Predictor: {pred_name}."]
    if got.lower() != gold:
        parts += [
            "Return exactly 'Potentially' or 'None'.",
            "Mocking/derogatory/stereotypes/discussing bias → 'Potentially'; otherwise 'None'."
        ]
    return Prediction(score=float(score), feedback=" ".join(parts))

# ---------------------------
# Safe wrappers
# ---------------------------
def compile_with_fallback(optimizer, module, *, trainset, valset, is_copro=False, max_attempts=3):
    """
    Try compiling with JSONAdapter; if parse issues occur, switch to ChatAdapter and retry.
    """
    # helper to actually call compile with the right signature
    def _compile():
        if is_copro:
            return optimizer.compile(module, trainset=valset, eval_kwargs={})
        elif isinstance(optimizer, MIPROv2):
            return optimizer.compile(module, trainset=trainset, valset=valset)
        elif isinstance(optimizer, GEPA):
            return optimizer.compile(module, trainset=trainset, valset=valset)
        else:
            # fallback for unknown teleprompters
            return optimizer.compile(module, trainset=trainset, valset=valset)

    # first attempts under current adapter (likely JSON)
    last_err = None
    for k in range(1, max_attempts + 1):
        try:
            return _compile()
        except AdapterParseError as e:
            print(f"[WARN] AdapterParseError on attempt {k}/{max_attempts}: {e}")
            last_err = e
        except Exception as e:
            print(f"[WARN] Compile failed (attempt {k}/{max_attempts}): {e}")
            last_err = e

    # switch adapter to Chat and try a couple more times
    print("[INFO] Switching DSPy adapter to ChatAdapter and retrying…")
    dspy.configure(adapter=ChatAdapter())
    for k in range(1, 1 + max_attempts):
        try:
            return _compile()
        except Exception as e:
            print(f"[WARN] Compile failed after adapter switch (attempt {k}/{max_attempts}): {e}")
            last_err = e

    print("[ERROR] All compile attempts failed. Proceeding with unoptimized base module.")
    # return the original module so downstream code still runs
    return module

def predict_batch_safe(optimized_module, texts: List[str]) -> List[str]:
    preds: List[str] = []
    for t in texts:
        try:
            out = optimized_module(text=t)
            preds.append(normalize_label(getattr(out, "label", "None")))
        except AdapterParseError as e:
            # try one more time with ChatAdapter for this call
            print(f"[WARN] AdapterParseError during predict; switching adapter for this call: {e}")
            dspy.configure(adapter=ChatAdapter())
            try:
                out = optimized_module(text=t)
                preds.append(normalize_label(getattr(out, "label", "None")))
            except Exception as e2:
                print(f"[WARN] Predict still failed; defaulting to 'None': {e2}")
                preds.append("None")
            finally:
                dspy.configure(adapter=JSONAdapter())
        except Exception as e:
            print(f"[WARN] Predict failed; defaulting to 'None': {e}")
            preds.append("None")
    return preds

# ---------------------------
# Optimizer runner
# ---------------------------
def run_optimizer(name: str, optimizer, module, trainset, devset, test_df):
    print(f"\n=== Optimizing with {name} ===")

    # GEPA: use smaller val slice for better exploration within budget
    val_for_opt = devset[: min(200, len(devset))] if isinstance(optimizer, GEPA) else devset

    is_copro = isinstance(optimizer, COPRO)
    optimized = compile_with_fallback(
        optimizer,
        module,
        trainset=trainset,
        valset=val_for_opt,
        is_copro=is_copro,
        max_attempts=3
    )

    # Predict on test, safely
    preds = predict_batch_safe(optimized, test_df["text"].astype(str).tolist())
    metrics = evaluate_predictions(test_df["label"].tolist(), preds)
    print(f"{name} -> Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Prec: {metrics['precision']:.4f}")
    return metrics

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/biasLabeling/training/combined_fix6.tsv")
    ap.add_argument("--train_size", type=int, default=1000)
    ap.add_argument("--dev_size", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="dspy_optimization_results.tsv")
    args = ap.parse_args()

    task_lm, prompt_lm = configure_lms()

    # Load data
    df = load_dataset(args.data)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train_all, dev_all, test = split_train_dev_test(df, test_size=0.2, dev_size=0.2, seed=args.seed)
    train = train_all.sample(n=min(args.train_size, len(train_all)), random_state=args.seed)
    dev   = dev_all.sample(n=min(args.dev_size, len(dev_all)), random_state=args.seed)

    trainset = to_dspy_examples(train)
    devset   = to_dspy_examples(dev)
    base_module = dspy.Predict(signature=BiasPotentialSignature)

    # Teleprompters (robust but conservative budgets)
    copro = COPRO(
        metric=acc_metric,
        prompt_model=prompt_lm,
        max_iters=3,
        breadth=3,
        depth=2,
        verbose=True
    )

    mipro = MIPROv2(
        metric=acc_metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto="light",
        verbose=True
    )

    gepa = GEPA(
        metric=gepa_metric,
        auto="light",
        reflection_lm=prompt_lm,
        track_stats=False
    )

    # Run optimizers; never crash on one failure
    results = {}
    for name, opt in [("GEPA", gepa), ("MIPROv2", mipro), ("COPRO", copro)]:
        try:
            results[name] = run_optimizer(name, opt, base_module, trainset, devset, test)
        except Exception as e:
            print(f"[ERROR] {name} run failed entirely: {e}")
            results[name] = {"accuracy": float("nan"), "f1": float("nan"), "precision": float("nan")}

    # Save results
    try:
        out_df = pd.DataFrame.from_dict(results, orient="index")[["accuracy", "f1", "precision"]]
        out_df.to_csv(args.out, sep="\t")
        print(f"\nSaved results to {args.out}")
    except Exception as e:
        print(f"[WARN] Could not write results TSV: {e}")

    # Quick probe (safe)
    example = "女权怼了女拳？这事儿女拳知道吗？"
    for name, opt in [("COPRO", copro), ("MIPROv2", mipro), ("GEPA", gepa)]:
        try:
            compiled = compile_with_fallback(
                opt,
                base_module,
                trainset=trainset,
                valset=devset[: min(200, len(devset))],
                is_copro=isinstance(opt, COPRO),
                max_attempts=2
            )
            pred = predict_batch_safe(compiled, [example])[0]
            print(f'{name} example → "{example}" => {pred}')
        except Exception as e:
            print(f"[WARN] {name} probe failed: {e}")

if __name__ == "__main__":
    main()
