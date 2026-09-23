# Agri Vision: Federated Crop-Disease Detection

This FYP prototype trains a **30-class YOLOv8n object detector** with [Flower](https://flower.ai/docs/framework/). Three local clients each train on a different image shard; a FedAvg server combines their model parameters after every round. Images stay on their client, but model parameters are exchanged. This is a research prototype, not a privacy guarantee or a deployed diagnosis system.

## What is in the project

| File | Purpose |
| --- | --- |
| `model.py` | Creates the common 30-class YOLOv8n model and converts its state to/from Flower arrays. |
| `federated_client.py`, `client1.py`–`client3.py` | Train and validate each client; the three launchers only select the dataset. |
| `server.py` | Runs FedAvg, sends epoch/batch/image-size settings, aggregates validation metrics, and saves global checkpoints. |
| `split_dataset.py` | Checks for data leakage or rebuilds deterministic, non-overlapping shards. |
| `evaluate.py`, `predict.py` | Test a global checkpoint and run image/video inference. |
| `original_dataset/` | Source images and YOLO-format labels; its `test/` split is held out from training. |
| `dataset/client1/`–`client3/` | Client-specific `train`, `valid`, and `test` splits plus `data.yaml`. |
| `runs/`, `saved_models/` | Local training logs and aggregated global checkpoints. |

The old Faster R-CNN function in `model.py` was unrelated to the YOLO clients and has been replaced. `yolov8n.pt` is the common pretrained starting point, not the final trained FYP model.

## Requirements

### Run on Google Colab

Open [AgriVision Colab](https://colab.research.google.com/github/HannanRehman/HannanRehman4/blob/main/AgriVision_Colab.ipynb).
Upload `original_dataset_colab.zip` and `agrivision_colab_code.zip` to
`MyDrive/Agri Vision/`, select a GPU runtime,
then choose **Run all** and authorize Drive. Edit the ZIP path in the first cell if needed.
Keep existing Drive files; the small code bundle avoids requiring GitHub credentials.
It extracts the dataset locally and rebuilds disjoint client shards with seed 42.

`colab_train.py` runs all three clients sequentially on one GPU as a local FedAvg
simulation, without network workers. Every client starts from the same global weights;
updates are weighted by training image count. This preserves the three-client round
structure but is not a deployment across three independent devices.
The Colab notebook defaults to ten rounds, one epoch per client, batch 16, image size
320, and two data-loader workers. Per-client training validation is disabled because
the aggregated global model is already validated on all three clients after each round.
Completed global checkpoints, local client outputs, metrics and logs go to
`MyDrive/Agri Vision/colab_results`. To continue, set `INITIAL_CHECKPOINT` to a saved
global checkpoint and select the number of additional rounds. An interrupted partial
round must restart. During training, a heartbeat is printed every minute and the current
round, client, phase, and CUDA memory are written to `status.json` in the run folder.
GPU execution must be verified in Colab; local unit tests alone
do not establish Colab compatibility or training accuracy.

The current 30-class names were reconstructed from label/file evidence; confirm them
against the original export before reporting class-specific results. Numeric label
validation cannot verify disease-name semantics.

- A working Python 3.11 or 3.12 installation, enough disk space for a second dataset copy, and preferably an NVIDIA GPU with a compatible PyTorch install. CPU training is possible but slow.
- Run all commands from the project root. On Windows, open PowerShell in `F:\Agri Vision`.
- The committed `venv/` was created against an `E:\python.exe` installation and is broken if that Python standard library is unavailable. Create a fresh environment instead of relying on it.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch, flwr, ultralytics; print(torch.cuda.is_available())"
```

If you need CUDA, select the appropriate PyTorch installation command from the [official PyTorch installer](https://pytorch.org/get-started/locally/) before installing the remaining requirements. `False` in the final command means the code will default to CPU.

## Prepare and check the data

**Do this before training.** The committed client shards are already non-overlapping, but run the checker after copying the project or changing any dataset files. The server refuses to train if leakage or invalid labels are detected.

```powershell
python split_dataset.py
# Only rebuild when the checker reports a partition problem:
python split_dataset.py --rebuild --seed 42
```

The check should show no overlap, 30 valid classes, and totals of 15,310 train, 4,374 validation, and 2,191 test images. `--rebuild` copies the original dataset into a new, verified partition, swaps it into `dataset/`, and **keeps the previous `dataset/` as a timestamped `dataset_backup_*` directory**. It does not delete the original or the backup. Ensure sufficient free space before rebuilding. The same seed produces the same assignment.

If copying is interrupted, the previous `dataset/` remains in place and a partial `dataset_staging_*` directory may remain. Inspect it before removing it; the script never automatically deletes a dataset or backup.

Each client `data.yaml` must have the same 30 class IDs and names. YOLO labels use `class_id x_center y_center width height` with normalized coordinates. The checker rejects malformed coordinates and class IDs outside `0`–`29` before training.

## Train

Start the server, then start **all three clients** in separate terminals (activate `.venv` in each):

```powershell
# Terminal 1
python server.py --rounds 5 --local-epochs 3 --batch 2 --imgsz 320

# Terminal 2
python client1.py

# Terminal 3
python client2.py

# Terminal 4
python client3.py
```

One round means every client receives the current global model, trains for `--local-epochs`, and sends updated parameters back. In the example, **each client can train for up to 15 local epochs** (5 rounds × 3 epochs), with aggregation after every 3. For a quick functional check, use `--rounds 1 --local-epochs 1`. You can raise `--rounds`, `--local-epochs`, `--batch`, or `--imgsz` when hardware allows; these settings do not guarantee better accuracy. Compare validation and test mAP and watch for overfitting. The old CPU run took roughly 1.4 hours for one client's single epoch, so plan runtime accordingly.

By default the server listens on `0.0.0.0:8080`, while clients connect to `127.0.0.1:8080`. On separate machines, pass `--server HOST:8080` to each client and ensure the port is reachable. Pass `--device cpu` to clients without CUDA; on one GPU, three concurrent clients may exceed available VRAM, so lower batch size or use separate devices. `--batch` and `--imgsz` are set by the server for all clients.

Local YOLO logs and best/last checkpoints go under `runs/federated/clientN/round_XXX/`. The server saves the **aggregated global model** after each successful round as `saved_models/global_round_XXX.pt`. This is a project checkpoint containing a PyTorch state dictionary, not a native Ultralytics `.pt` checkpoint; load it with the project scripts below. To continue from a completed global round in a new run, use `python server.py --initial-checkpoint saved_models/global_round_005.pt --rounds 5 ...` and restart the three clients. New output round numbers start at 1 for that run, so choose a new `--checkpoint-dir` to preserve previous files.

Flower weights aggregation by the *actual number of client training images*. The server also averages client validation mAP50, mAP50-95, precision, and recall weighted by validation-image count. Flower's reported scalar “loss” is `1 - mAP50`, an error proxy, **not** YOLO's box/classification loss. Use the YOLO run logs for training losses.

## Evaluate and predict

After training, evaluate the final global model once on the held-out original test split:

```powershell
python evaluate.py --checkpoint saved_models/global_round_005.pt --device cpu
python predict.py --checkpoint saved_models/global_round_005.pt --source path\to\image.jpg --device cpu
```

`evaluate.py` prints test mAP50 and mAP50-95. `predict.py` saves annotated outputs under `runs/predictions/`. Keep the test split for final reporting; tune epochs and other choices using validation results, not repeated test-set trials. For your report, record the split seed, Python/package versions, model settings, hardware, per-round validation results, and final held-out test metrics. A centralized YOLOv8n baseline trained on the same training images would make the federated comparison much stronger.

## Verification and limitations

`python split_dataset.py` checks client overlap, coverage of the original images, matching label files, class mappings, and every label value. It blocks the server when the shards are invalid. The code can be syntax-checked without expensive training, but a full multi-client run still depends on a working Python environment and sufficient hardware. Better accuracy is an empirical outcome, not something the epoch count alone can promise.

The partition tests can be run with `python -m unittest discover -s tests -v`. The existing `runs/detect/` files are from earlier local training, not proof that the updated federated workflow has completed.
