"""
Evaluate classification performance of the two-stage IsPlanktonBIO pipeline.

Usage:
    python -m tests.evaluate_metrics --json_path output/inference_results_cs_test.json

Evaluation methodology
----------------------
Performance is reported at three levels, each isolating a different aspect
of the system.

Pipeline Acceptance / Rejection Analysis summarises the pipeline_status field
for every image: fraction accepted (matched), rejected as OOD at each stage,
rejected for border-touching specimens, and unmatched cases where Stage-1 and
Stage-2 predictions disagreed.  This section characterises the operating point
of the OOD gates independently of classification quality.

Global Classifier Cascade uses the best classification prediction produced at
any stage: Stage-2 when available, Stage-1 otherwise.  Images rejected by
the border filter before any meaningful classification are counted as
"Rejected/Failed".  This metric evaluates the discriminative power of the
classifier cascade independently of the OOD gating decision, and is the
primary end-to-end accuracy reported for in-distribution test sets.

Stage Metrics report the raw classifier accuracy of each backbone without
OOD gating.  Stage-1 accuracy covers all images using the Stage-1 argmax
prediction.  Stage-2 accuracy covers all images that reached Stage-2
(including those subsequently rejected as OOD there), isolating the
discriminative quality of each backbone.

All metrics (Accuracy, Macro-Precision, Macro-Recall, Macro-F1) are computed
with sklearn using macro averaging across all classes; zero-division is set
to 0.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _derive_output_path(json_path: str) -> Path:
    """Derive the output .txt path from the inference JSON filename.

    'output/inference_results_vos_test.json'  →  'tests/output/vos_acc.txt'
    'output/inference_results_cs_ood.json'    →  'tests/output/cs_acc.txt'
    'output/inference_results.json'           →  'tests/output/inference_results_acc.txt'
    """
    stem = Path(json_path).stem
    prefix = "inference_results_"
    if stem.startswith(prefix):
        remainder = stem[len(prefix):]          # e.g. 'vos_test' or 'cs_ood'
        parts = remainder.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("ood", "test"):
            method = parts[0]                   # 'vos', 'cs', 'le', …
        else:
            method = remainder
    else:
        method = stem
    return Path("tests/output") / f"{method}_acc.txt"


def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc * 100, prec * 100, rec * 100, f1 * 100


def evaluate(json_path: str) -> None:
    with open(json_path, "r") as f:
        data = json.load(f)

    y_true_global, y_pred_global = [], []
    y_true_s1, y_pred_s1 = [], []
    y_true_s2, y_pred_s2 = [], []

    status_counts: dict[str, int] = {}
    total = 0

    for ground_truth, items in data.items():
        for item in items:
            total += 1
            status = item["pipeline_status"]
            status_counts[status] = status_counts.get(status, 0) + 1

            # Stage-1: raw classifier accuracy (no OOD gating)
            y_true_s1.append(ground_truth)
            y_pred_s1.append(item["stage_1_prediction"])

            # Stage-2: raw classifier accuracy on images that reached Stage-2
            if item["stage_2_prediction"] is not None:
                y_true_s2.append(ground_truth)
                y_pred_s2.append(item["stage_2_prediction"])

            # Global (End-to-End): best prediction the pipeline produced at any stage.
            # touches-border is rejected before meaningful classification, so it is
            # the only status treated as "Rejected/Failed".
            y_true_global.append(ground_truth)
            if item["stage_2_prediction"] is not None:
                y_pred_global.append(item["stage_2_prediction"])
            elif status != "touches-border":
                y_pred_global.append(item["stage_1_prediction"])
            else:
                y_pred_global.append("Rejected/Failed")

    # --- PIPELINE REJECTION ANALYSIS ---
    accepted = status_counts.get("matched", 0)
    ood_s1 = status_counts.get("ood_sample_stage1", 0)
    ood_s2 = status_counts.get("ood_sample_stage2", 0)
    border = status_counts.get("touches-border", 0)
    unmatched = status_counts.get("unmatched", 0)

    print("--- PIPELINE ACCEPTANCE / REJECTION ANALYSIS ---")
    print(f"Total Images:              {total}")
    print(f"  Accepted (matched):      {accepted}  ({accepted/total*100:.1f}%)")
    print(f"  Rejected OOD @ Stage-1:  {ood_s1}  ({ood_s1/total*100:.1f}%)")
    print(f"  Rejected OOD @ Stage-2:  {ood_s2}  ({ood_s2/total*100:.1f}%)")
    print(f"  Rejected (border):       {border}  ({border/total*100:.1f}%)")
    print(f"  Unmatched (S1≠S2):       {unmatched}  ({unmatched/total*100:.1f}%)")
    print()

    # --- GLOBAL ---
    acc, prec, rec, f1 = calculate_metrics(y_true_global, y_pred_global)
    print("--- GLOBAL METRICS (CLASSIFIER CASCADE, END-TO-END) ---")
    print("  Uses Stage-2 prediction when available, else Stage-1.")
    print(f"Total Images: {len(y_true_global)}")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall:    {rec:.2f}%")
    print(f"F1-Score:  {f1:.2f}%")
    print()

    # --- STAGE METRICS ---
    print("--- STAGE METRICS (RAW CLASSIFIER, NO OOD GATING) ---")

    s1_acc, s1_prec, s1_rec, s1_f1 = calculate_metrics(y_true_s1, y_pred_s1)
    print(f"Stage-1 Evaluated Images: {len(y_true_s1)}")
    print(f"Stage-1 Accuracy:  {s1_acc:.2f}%")
    print(f"Stage-1 Precision: {s1_prec:.2f}%")
    print(f"Stage-1 Recall:    {s1_rec:.2f}%")
    print(f"Stage-1 F1-Score:  {s1_f1:.2f}%")
    print()

    s2_acc, s2_prec, s2_rec, s2_f1 = calculate_metrics(y_true_s2, y_pred_s2)
    print(f"Stage-2 Evaluated Images: {len(y_true_s2)} (images that reached Stage-2)")
    print(f"Stage-2 Accuracy:  {s2_acc:.2f}%")
    print(f"Stage-2 Precision: {s2_prec:.2f}%")
    print(f"Stage-2 Recall:    {s2_rec:.2f}%")
    print(f"Stage-2 F1-Score:  {s2_f1:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate inference results from the new JSON format."
    )
    parser.add_argument(
        "--json_path",
        required=True,
        help="Path to the inference results JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the output .txt file. "
            "Auto-derived from --json_path when omitted "
            "(e.g. output/vos_acc.txt)."
        ),
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else _derive_output_path(args.json_path)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluate(args.json_path)

    text = buf.getvalue()
    sys.stdout.write(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"\nResults saved → {output_path}")
