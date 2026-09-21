"""One Flower client implementation reused by all three dataset owners."""

import argparse

import flwr as fl
import torch

from model import PROJECT_ROOT, class_names, create_model, get_parameters, set_parameters


class AgriVisionClient(fl.client.NumPyClient):
    def __init__(self, client_id, device):
        self.client_id = client_id
        self.data_yaml = PROJECT_ROOT / "dataset" / client_id / "data.yaml"
        if class_names(self.data_yaml) != class_names():
            raise ValueError(f"Class IDs/names differ from the other clients: {self.data_yaml}")
        self.train_count = self._count_images("train")
        self.val_count = self._count_images("valid")
        self.device = device
        self.model = create_model()
        self.run_dir = PROJECT_ROOT / "runs" / "federated" / client_id

    def _count_images(self, split):
        folder = self.data_yaml.parent / split / "images"
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image directory: {folder}")
        count = sum(1 for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if count == 0:
            raise ValueError(f"No images in {folder}")
        return count

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        epochs = int(config.get("local_epochs", 1))
        batch = int(config.get("batch", 2))
        imgsz = int(config.get("imgsz", 320))
        if min(epochs, batch, imgsz) < 1:
            raise ValueError("local_epochs, batch, and imgsz must be positive")
        server_round = int(config.get("server_round", 0))
        self.model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=str(self.run_dir),
            name=f"round_{server_round:03d}",
            exist_ok=False,  # never overwrite an earlier local training run
            workers=0,  # reliable on Windows when running multiple client processes
        )
        return get_parameters(self.model), self.train_count, {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        metrics = self.model.val(
            data=str(self.data_yaml),
            split="val",
            imgsz=int(config.get("imgsz", 320)),
            batch=int(config.get("batch", 2)),
            device=self.device,
            workers=0,
        )
        map50 = float(metrics.box.map50)
        # Flower requires a scalar loss; this is an error proxy, not YOLO's training loss.
        return 1.0 - map50, self.val_count, {
            "mAP50": map50,
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }


def start_client(client_id):
    parser = argparse.ArgumentParser(description=f"Start the {client_id} Flower client")
    parser.add_argument("--server", default="127.0.0.1:8080", help="Flower server host:port")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="Ultralytics device, e.g. 0 or cpu")
    args = parser.parse_args()
    fl.client.start_numpy_client(
        server_address=args.server,
        client=AgriVisionClient(client_id, args.device),
    )
