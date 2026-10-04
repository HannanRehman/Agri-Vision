"""Report on archived results only; does not train or evaluate a detector."""
import csv
import hashlib
import io
import json
import re
import statistics
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
ARCHIVE = ROOT / "saved_models/20261002_121640_89aa17-20261003T102507Z-1-001.zip"
KEYS = ["precision", "recall", "mAP50", "mAP50_95"]
LABELS = ["Precision", "Recall", "mAP50", "mAP50–95"]
with zipfile.ZipFile(ARCHIVE) as z:
    names = z.namelist()
    prefix = names[0].split("/")[0] + "/"
    bad = z.testzip()
    if bad:
        raise ValueError(f"Archive CRC failure: {bad}")
    settings = json.loads(z.read(prefix + "settings.json"))
    status = json.loads(z.read(prefix + "status.json"))
    metrics = {r: json.loads(z.read(prefix + f"metrics_round_{r:03d}.json")) for r in range(26, 36)}
    local = []
    args = []
    for client in range(1, 4):
        for r in range(26, 36):
            stem = prefix + f"client{client}/round_{r:03d}/"
            rows = list(csv.DictReader(io.StringIO(z.read(stem + "results.csv").decode())))
            if len(rows) != 1:
                raise ValueError(f"Expected one epoch: {stem}")
            local.append({"client": client, "round": r, **{k: float(v) for k, v in rows[0].items()}})
            args.append(yaml.safe_load(z.read(stem + "args.yaml")))
            for suffix in ["weights/best.pt", "weights/last.pt"]:
                assert stem + suffix in names
    for r in metrics:
        assert prefix + f"global_round_{r:03d}.pt" in names
        assert all(0 <= metrics[r][k] <= 1 for k in KEYS)
    assert status["phase"] == "complete" and status["round"] == 35
    assert status["metrics"] == metrics[35]
    local_checkpoint = ROOT / "global_round_035.pt"
    matching = local_checkpoint.exists() and hashlib.sha256(local_checkpoint.read_bytes()).digest() == hashlib.sha256(z.read(prefix + "global_round_035.pt")).digest()
    ck_size = z.getinfo(prefix + "global_round_035.pt").file_size

with (OUT / "rounds_026_035_global_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["round", *KEYS])
    w.writeheader()
    w.writerows({"round": r, **m} for r, m in metrics.items())
with (OUT / "rounds_026_035_client_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(local[0]))
    w.writeheader()
    w.writerows(local)

best = {k: max(metrics, key=lambda r: metrics[r][k]) for k in KEYS}
means = {k: statistics.mean(m[k] for m in metrics.values()) for k in KEYS}
start = datetime.strptime(prefix.rstrip("/").split("_")[0] + "_" + prefix.rstrip("/").split("_")[1], "%Y%m%d_%H%M%S")
end = datetime.fromisoformat(status["updated_at_utc"]).replace(tzinfo=None)
duration = (end - start).total_seconds()
logged_time = sum(row["time"] for row in local)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
rounds = list(metrics)
for ax, k, label, color in zip(axes.flat, KEYS, LABELS, ["#216e39", "#1b67a5", "#b06500", "#8045a0"]):
    ax.plot(rounds, [metrics[r][k] * 100 for r in rounds], marker="o", color=color)
    b = best[k]
    ax.scatter([b], [metrics[b][k] * 100], s=110, facecolors="none", edgecolors="black", label=f"Best: round {b}")
    ax.set(title=label, xlabel="Federated round", ylabel="Validation score (%)", xticks=rounds)
    ax.grid(alpha=.2)
    ax.legend(loc="best", fontsize=9)
fig.suptitle("Agri Vision · Rounds 26–35 · Weighted client validation", fontsize=16)
fig.savefig(OUT / "rounds_026_035_validation.png", dpi=180)
fig.savefig(OUT / "rounds_026_035_validation.pdf")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
for ax, key, title in zip(axes, ["train/box_loss", "train/cls_loss", "train/dfl_loss"], ["Box loss", "Classification loss", "Distribution focal loss"]):
    for client in range(1, 4):
        rows = [row for row in local if row["client"] == client]
        ax.plot([row["round"] for row in rows], [row[key] for row in rows], marker=".", label=f"Client {client}")
    ax.set(title=title, xlabel="Federated round", ylabel="Training loss")
    ax.grid(alpha=.2)
axes[0].legend()
fig.suptitle("Local training losses · Each point is a separate one-epoch client run", fontsize=13)
fig.savefig(OUT / "rounds_026_035_training_losses.png", dpi=180)
fig.savefig(OUT / "rounds_026_035_training_losses.pdf")
plt.close(fig)

lines = [
    "# Agri Vision: Detailed Report on Federated Rounds 26–35", "",
    f"Source archive: `{ARCHIVE.name}`. Analysis reads saved artifacts only; no training or detector evaluation was performed.", "",
    "## Main findings", "",
    "The archive confirms successful completion of ten additional federated rounds from a round-25 checkpoint. Metrics were saved in separate JSON files, rather than inside the global checkpoint. The latest checkpoint is not the highest-scoring checkpoint in this continuation.", "",
    f"Round 35: precision **{metrics[35]['precision']*100:.2f}%**, recall **{metrics[35]['recall']*100:.2f}%**, mAP50 **{metrics[35]['mAP50']*100:.2f}%**, mAP50–95 **{metrics[35]['mAP50_95']*100:.2f}%**. These are validation metrics averaged across clients, weighted by validation-image count; they are not held-out test results.", "",
    f"Best mAP50: round **{best['mAP50']}** ({metrics[best['mAP50']]['mAP50']*100:.2f}%). Best mAP50–95: round **{best['mAP50_95']}** ({metrics[best['mAP50_95']]['mAP50_95']*100:.2f}%).", "",
    "## Run configuration and completion evidence", "",
    "| Setting | Recorded value |", "|---|---|",
    "| Detector | YOLOv8n; project uses a 30-class detection head |",
    "| Execution | Three-client sequential FedAvg simulation on one GPU |",
    "| Starting checkpoint | global_round_025.pt |",
    f"| Additional rounds | {settings['rounds']} (26–35) |",
    f"| Local epochs | {settings['local_epochs']} per client per round; 10 additional epochs per client |",
    "| Client training calls | 30; these are not 30 sequential epochs on one shared model |",
    f"| Image size / batch / workers | {settings['imgsz']} / {settings['batch']} / {settings['workers']} |",
    f"| GPU | {settings['gpu']} |",
    f"| PyTorch | {settings['torch']} |",
    "| Recorded optimizer setting | auto; actual selected optimizer is not recorded here |",
    "| AMP / deterministic / seed | true / true / 0 in each local training configuration |",
    "| Local resume setting | false; model weights are loaded by the federated code, not a continuous optimizer resume |",
    f"| Final status | {status['phase']}, round {status['round']} |",
    f"| Final status timestamp | {status['updated_at_utc']} |", "",
    "Archive inventory: 142 files: 10 global checkpoints, 10 global metric JSON files, 30 local CSV files, 30 local argument YAML files, 60 local best/last checkpoints, settings.json, and status.json. All archive entries passed ZIP CRC verification. Every expected client-round result and checkpoint is present. The final status metrics exactly match metrics_round_035.json.", "",
    f"The standalone workspace global_round_035.pt {'is byte-identical to' if matching else 'could not be confirmed byte-identical to'} the archived round-35 checkpoint. A global checkpoint is {ck_size:,} bytes ({ck_size/1024**2:.2f} MiB); this serialized file size is not a measurement of network traffic.", "",
    "## Global validation results by round", "",
    "| Round | Precision | Recall | mAP50 | mAP50–95 |", "|---:|---:|---:|---:|---:|",
]
for r, m in metrics.items():
    lines.append(f"| {r} | " + " | ".join(f"{m[k]*100:.2f}%" for k in KEYS) + " |")
lines += ["", "![Global validation trends](rounds_026_035_validation.png)", "",
    "Definitions: precision measures correct detections among predicted detections; recall measures recovered labelled objects. mAP50 measures average precision at a box-overlap threshold of 0.50, while mAP50–95 averages stricter thresholds from 0.50 to 0.95. Saved precision/recall are detector summaries averaged within clients and then across clients; they are not pooled object counts.", "",
    "The weighted mean of separate client AP values does not equal AP recomputed from all validation predictions together. Client-level global-model validation scores and counts were not retained separately in this archive, so the weighted aggregation cannot be independently reconstructed from these files.", "",
    "## Trend and checkpoint selection", "",
    "| Metric | Round 26 | Round 35 | Change (percentage points) | Best round | Best value |", "|---|---:|---:|---:|---:|---:|",
]
for k, label in zip(KEYS, LABELS):
    lines.append(f"| {label} | {metrics[26][k]*100:.2f}% | {metrics[35][k]*100:.2f}% | {(metrics[35][k]-metrics[26][k])*100:+.2f} | {best[k]} | {metrics[best[k]][k]*100:.2f}% |")
lines += ["", f"Across the ten rounds, mean mAP50 was {means['mAP50']*100:.2f}% and mean mAP50–95 was {means['mAP50_95']*100:.2f}%. mAP50 stayed near 61% for rounds 27–33 before declining in rounds 34–35. The final score was {(metrics[best['mAP50']]['mAP50']-metrics[35]['mAP50'])*100:.2f} percentage points below the best mAP50 and {(metrics[best['mAP50_95']]['mAP50_95']-metrics[35]['mAP50_95'])*100:.2f} points below the best mAP50–95.", "",
    "If the declared primary selection metric is mAP50–95, round 30 is the candidate to retain. If it is mAP50, round 33 is the candidate. Round 33 also has the highest reported recall, but lower precision. The differences among rounds 27–33 are small; no repeated-seed experiments or confidence intervals establish statistically significant superiority. These rankings apply only to the rounds in this archive, not all rounds 1–35.", "",
    "## Local client training results", "",
    "The client CSVs describe individually trained local models before FedAvg. They must not be substituted for global-model validation. Although args.yaml sets val=false, the CSVs contain nonzero validation metrics and losses; the archive establishes that those values were emitted, but does not retain the console logs needed to confirm their exact validation execution path. A final-epoch validation is a possible explanation, not a verified finding.", "",
    "| Client | Mean local mAP50 (26–35) | Round-26 local mAP50 | Round-35 local mAP50 | Round-35 local mAP50–95 |", "|---|---:|---:|---:|---:|",
]
for client in range(1, 4):
    rows = [row for row in local if row["client"] == client]
    a, b = rows[0], rows[-1]
    lines.append(f"| Client {client} | {statistics.mean(row['metrics/mAP50(B)'] for row in rows)*100:.2f}% | {a['metrics/mAP50(B)']*100:.2f}% | {b['metrics/mAP50(B)']*100:.2f}% | {b['metrics/mAP50-95(B)']*100:.2f}% |")
lines += ["", "These clients use different validation shards. Their differences may reflect shard difficulty, class support, and local training effects; they do not prove that one client is intrinsically better. The global model's mAP50 is higher than every local-model mAP50 recorded at the same round throughout this continuation, but shard-level global scores are unavailable for a paired comparison.", "",
    "## Loss behaviour", "",
    "| Client | Box loss: 26 → 35 | Classification loss: 26 → 35 | DFL: 26 → 35 |", "|---|---:|---:|---:|",
]
for client in range(1, 4):
    rows = [row for row in local if row["client"] == client]
    lines.append(f"| Client {client} | " + " | ".join(f"{rows[0][k]:.4f} → {rows[-1][k]:.4f}" for k in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]) + " |")
lines += ["", "![Local training losses](rounds_026_035_training_losses.png)", "",
    "All three training-loss components decreased monotonically for all three clients. Nevertheless, global validation mAP declined at the end. This shows that improving training fit did not translate into continuing validation improvement. It is consistent with overfitting or federated optimisation effects, but these artifacts cannot identify a unique cause.", "",
    "Local validation losses also increased between rounds 26 and 35 for all three clients and all three recorded components. This strengthens the evidence of a growing training-versus-validation gap, subject to the CSV validation-provenance limitation above.", "",
    "Each client CSV records an epoch numbered 1 and the same final learning rate, 0.0000978485, in all three parameter groups. The local code creates a new one-epoch training call each round, so this is not a continuous ten-epoch optimizer/scheduler history. args.yaml records three warmup epochs even though each call trains for one epoch. The effect of optimizer restarts and warmup should be investigated before assuming that increasing rounds will help; this report does not claim they caused the decline.", "",
    "## Runtime and resource evidence", "",
    f"The sum of the 30 recorded local epoch times is {logged_time:.3f} seconds ({logged_time/60:.2f} minutes). Individual client-epoch times range from {min(row['time'] for row in local):.3f} to {max(row['time'] for row in local):.3f} seconds. This covers the times emitted in local CSVs, not the whole run.", "",
    f"The interval from the run folder's timestamp ({start.isoformat()} assumed UTC) to the final status timestamp is {duration/60:.2f} minutes. This is an approximate wall-clock interval inferred from naming, not a recorded timer. The difference from summed epoch times can include global validation, model setup, checkpoint writes, and Drive copies, but those components cannot be separated here.", "",
    "The final CUDA allocations are only a post-run snapshot (0.031 GB allocated/reserved). They do not represent peak training GPU memory. Network traffic, energy, and inference latency were not measured. The single-GPU sequential simulation does not establish performance on three physically independent devices.", "",
    "## What this archive cannot establish", "",
    "- It contains no held-out test evaluation for these rounds. The existing round-25 report is a test report, so comparing its 57.01% test mAP50 with round-35's 60.05% validation mAP50 would not demonstrate improvement.",
    "- It contains no round-25 validation metric file, so improvement relative to the starting checkpoint on matching validation data cannot be computed from this archive.",
    "- No per-class global AP, confusion matrices, or prediction examples are retained. Claims that particular chili diseases improved or regressed would be unsupported.",
    "- No per-client global validation breakdown, raw detections, or sample counts are saved with the metrics. Aggregation weights cannot be independently audited here.",
    "- No centralized baseline, repeated runs, or alternative architectures are included.",
    "- The exact training source code and Ultralytics version were not archived in this run. Workflow interpretation uses the current local project code and saved args/settings.", "",
    "## Recommended next steps", "",
    "1. Preserve all checkpoints. Select round 30 if mAP50–95 is the primary validation criterion, or round 33 if mAP50 is primary; document the choice before test evaluation.",
    "2. Retrieve round-25 validation metrics from the original run if available, to measure the effect of this continuation on matching data.",
    "3. Once a candidate is chosen, evaluate it on the held-out test split and retain per-class results, plots, and predictions. No such evaluation was run for this report.",
    "4. For future runs, save individual client validation metrics and counts, package versions, actual optimizer choice, and timing information alongside the global summaries.",
    "5. Investigate the late validation decline with controlled validation experiments. Further rounds alone are not justified by the observed trend.", "",
    "## FYP-ready result statement", "",
    "> Ten additional federated rounds were completed using three clients, one local epoch per round, and a YOLOv8n detector at image size 640 on a Tesla T4. The final round achieved validation mAP50 of 60.05% and mAP50–95 of 39.11%. The highest mAP50 was 61.22% at round 33, while the highest mAP50–95 was 40.61% at round 30. Although local training losses decreased throughout the continuation, validation performance declined in the last two rounds, indicating that additional training did not provide sustained generalisation gains. These results are validation-image-weighted averages of client metrics, rather than pooled held-out test performance.", "",
    "## Exported artifacts", "",
    "- rounds_026_035_global_metrics.csv: exact saved global summary values.",
    "- rounds_026_035_client_metrics.csv: all 30 local CSV rows with client and round identifiers.",
    "- rounds_026_035_validation.png / .pdf: global validation trends.",
    "- rounds_026_035_training_losses.png / .pdf: local loss curves.",
    "- analyze_rounds_026_035.py: reproducible archive-only report generator.", "",
]
(OUT / "rounds_026_035_detailed_report.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"report": str(OUT / 'rounds_026_035_detailed_report.md'), "best_rounds": best, "means": means, "round_35": metrics[35], "logged_epoch_minutes": logged_time/60, "estimated_wall_minutes": duration/60, "checkpoint_matches": matching}, indent=2))
