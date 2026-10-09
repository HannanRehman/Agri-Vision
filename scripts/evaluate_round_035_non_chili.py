"""Evaluate existing round-35 weights and report the 24 non-chili classes."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from model import class_names, create_model


def main():
    torch.set_num_threads(4)
    checkpoint = torch.load(ROOT / "saved_models/global_round_035.pt", map_location="cpu", weights_only=True)
    assert checkpoint["round"] == 35
    assert checkpoint["class_names"] == class_names()
    model = create_model()
    model.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = model.val(
        data=str(ROOT / "original_dataset/data.yaml"), split="test",
        device="cpu", imgsz=640, batch=8, workers=0, plots=True,
        project=str(ROOT / "saved_models"), name="round_035_non_chili_analysis",
    )
    rows = []
    for i, index in enumerate(metrics.box.ap_class_index):
        index = int(index)
        precision, recall, map50, map95 = map(float, metrics.box.class_result(i))
        rows.append({"class": metrics.names[index], "class_id": index,
                     "images": int(metrics.nt_per_image[index]),
                     "instances": int(metrics.nt_per_class[index]),
                     "precision": precision, "recall": recall,
                     "mAP50": map50, "mAP50_95": map95})
    kept = [row for row in rows if not row["class"].startswith("Chili_")]
    chili = [row for row in rows if row["class"].startswith("Chili_")]
    assert len(rows) == 30 and len(kept) == 24 and len(chili) == 6
    keys = ["precision", "recall", "mAP50", "mAP50_95"]
    average = lambda group: {key: sum(row[key] for row in group) / len(group) for key in keys}
    groups = {"all_30_classes": rows, "non_chili_24_classes": kept, "chili_6_classes": chili}
    summaries = {name: {"classes": len(group), "instances": sum(row["instances"] for row in group), **average(group)} for name, group in groups.items()}
    output = ROOT / "reports"
    for filename, group in [("round_035_test_per_class.csv", rows), ("round_035_test_non_chili_per_class.csv", kept)]:
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(group)
    payload = {"round": 35, "split": "test", "imgsz": 640, "save_dir": str(metrics.save_dir),
               "method": "Macro averages of existing model's per-class scores; all original test images retained. Precision/recall use the evaluation operating point selected over all 30 classes.",
               "groups": summaries}
    (output / "round_035_non_chili_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Round 35: Existing Model Performance Excluding Chili Classes", "",
             "The existing 30-class checkpoint was evaluated at image size 640 on the held-out original test split. No weights or datasets were modified. Chili classes are excluded from the reported 24-class average, not from the model or test images.", "",
             "| Group | Classes | Labelled instances | Precision | Recall | mAP50 | mAP50–95 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        lines.append(f"| {name} | {summary['classes']} | {summary['instances']} | " + " | ".join(f"{summary[key]*100:.2f}%" for key in keys) + " |")
    lines += ["", "These are macro class averages: every included class has equal weight. Precision and recall use the operating point selected for the original 30-class model, rather than a newly optimized non-chili threshold. Conventional classification accuracy is not computed for this detection task.", "",
              "All test images remain included. Predictions of non-chili classes on chili images can count as false positives. This reports how the existing model performs on non-chili classes; it does not predict performance after retraining without chili.", "",
              "| Crop group | Classes | Precision | Recall | mAP50 | mAP50–95 |", "|---|---:|---:|---:|---:|---:|"]
    for crop in ["Banana", "Cauliflower", "Groundnut", "Radish"]:
        group = [row for row in kept if row["class"].startswith(crop + "_")]
        scores = average(group)
        lines.append(f"| {crop} | {len(group)} | " + " | ".join(f"{scores[key]*100:.2f}%" for key in keys) + " |")
    old = list(csv.DictReader((output / "round_025_test_metrics.csv").open(encoding="utf-8")))
    old_kept = [row for row in old if row["class"] != "all" and not row["class"].startswith("Chili_")]
    old_scores = {key: sum(float(row[key]) for row in old_kept) / len(old_kept) for key in keys}
    lines += ["", "## Comparison with round 25 on the same held-out test split", "",
              "Round-25 values are recomputed from its rounded per-class CSV, so they are approximate. Precision/recall operating points can differ between checkpoints.", "",
              "| Non-chili metric | Round 25 | Round 35 | Change (percentage points) |", "|---|---:|---:|---:|"]
    for key in keys:
        lines.append(f"| {key} | {old_scores[key]*100:.2f}% | {summaries['non_chili_24_classes'][key]*100:.2f}% | {(summaries['non_chili_24_classes'][key]-old_scores[key])*100:+.2f} |")
    lines += ["", "## Non-chili per-class scores", "",
              "| Class | Instances | Precision | Recall | mAP50 | mAP50–95 |", "|---|---:|---:|---:|---:|---:|"]
    for row in kept:
        lines.append(f"| {row['class']} | {row['instances']} | " + " | ".join(f"{row[key]*100:.2f}%" for key in keys) + " |")
    lines += ["", f"Evaluation plots and prediction samples: `{metrics.save_dir}`.", ""]
    (output / "round_035_non_chili_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("NON_CHILI_RESULT", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
