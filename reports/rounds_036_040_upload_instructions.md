# Colab upload instructions: rounds 36–40

This continues the existing round-35, 30-class YOLOv8n model for five additional federated rounds. It includes all six chili classes. Each round trains three clients sequentially on the Colab GPU, starting from the same global weights, then combines updates and validates the global model.

## Upload to Google Drive

Use the folder **MyDrive/Agri Vision/**:

| Local file | Drive filename | Action |
|---|---|---|
| `agrivision_colab_code.zip` | `agrivision_colab_code.zip` | Replace the earlier code ZIP with this newly rebuilt bundle. |
| `saved_models/global_round_035.pt` | `global_round_035.pt` | Upload directly inside Agri Vision, without a saved_models subfolder. |
| `original_dataset_colab.zip` | `original_dataset_colab.zip` | Reuse the existing copy if unchanged; otherwise upload this file. |

Upload **AgriVision_Colab.ipynb** through Colab's **File → Upload notebook**. Use this updated local notebook, rather than an older notebook configured for rounds 26–35. The dataset ZIP is approximately 984 MB; the checkpoint is approximately 12.2 MB. No virtual environment, previous result ZIP, or report files are needed for training.

## Run the notebook

1. Select **Runtime → Change runtime type → GPU**.
2. Select **Runtime → Run all** and approve Google Drive access.
3. The notebook verifies SHA-256 checksums for the dataset, code, and checkpoint; verifies round 35; prepares the same seed-42 client partitions; and runs rounds 36–40.
4. Watch the live log and heartbeat. Normal completion includes **ROUND 40 validation**, **COMPLETE**, and a final table covering rounds 36, 37, 38, 39, and 40.

The settings are five additional rounds, one local epoch per client per round, image size 640, batch 8, and two data-loader workers. Model, data, aggregation, and training hyperparameters are kept the same as the previous continuation. This setup does not change optimizer restart or warmup behaviour. Validation outcomes determine whether further rounds helped; round 40 is not automatically the best checkpoint.

## Results to download afterwards

A unique folder is created under **MyDrive/Agri Vision/colab_results/**. Keep the earlier folders. Download the entire new folder, plus its sibling `.log` and `_input_sha256.json` files.

The folder contains:

- `global_round_036.pt` through `global_round_040.pt`.
- `metrics_round_036.json` through `metrics_round_040.json`: weighted global validation summaries.
- `metrics_clients_round_036.json` through `metrics_clients_round_040.json`: each client's validation-image count and global-model scores.
- `global_validation_metrics.csv`: one row per completed validation round.
- `client1/`, `client2/`, and `client3/`: local training CSVs, configurations, and weights.
- `settings.json`: configuration, package versions, class names, and round range.
- `status.json`: latest progress and completion state.

The global metric JSON shape is retained for compatibility with the earlier reports. These scores remain weighted client validation summaries, not pooled test metrics. Test evaluation is not run by this notebook.

## If Colab disconnects

A checkpoint is written before validation. Check the matching metric JSON and status to establish whether validation finished. Resume only from a completed global checkpoint, using a new run folder. Update the starting round, checkpoint path, checkpoint checksum, and remaining round count; the default notebook configuration is specifically for round 35 through round 40.

The verified upload checksums are recorded in `reports/rounds_036_040_upload_manifest.json` and embedded in the notebook. If project code changes, rebuild with `python scripts/prepare_colab_rounds_036_040.py` and re-upload the code ZIP and notebook together.
