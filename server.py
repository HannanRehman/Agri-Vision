"""Start federated YOLO training and save the aggregated global model."""

import argparse
from pathlib import Path

import flwr as fl
import torch

from model import PROJECT_ROOT, class_names, create_model, get_parameters, set_parameters
from split_dataset import check_partition


def weighted_metrics(results):
    total = sum(count for count, _ in results)
    if not total:
        return {}
    keys = ("mAP50", "mAP50_95", "precision", "recall")
    return {
        key: sum(count * float(metrics[key]) for count, metrics in results) / total
        for key in keys
    }


class CheckpointingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model, checkpoint_dir, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.checkpoint_dir = checkpoint_dir

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            set_parameters(self.model, fl.common.parameters_to_ndarrays(parameters))
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = self.checkpoint_dir / f"global_round_{server_round:03d}.pt"
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(
                {
                    "model_state_dict": self.model.model.state_dict(),
                    "class_names": class_names(),
                    "round": server_round,
                },
                temporary,
            )
            temporary.replace(checkpoint)
            print(f"Saved aggregated global model: {checkpoint}")
        return parameters, metrics


def main():
    parser = argparse.ArgumentParser(description="Start the Agri Vision Flower server")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Epochs per client in each round")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "saved_models")
    parser.add_argument("--initial-checkpoint", type=Path, help="Start another run from a saved global checkpoint")
    args = parser.parse_args()
    if min(args.rounds, args.local_epochs, args.batch, args.imgsz) < 1:
        parser.error("rounds, local-epochs, batch, and imgsz must be positive")
    if args.checkpoint_dir.exists() and any(args.checkpoint_dir.glob("global_round_*.pt")):
        parser.error("checkpoint directory already contains global rounds; choose a new --checkpoint-dir")

    # This local FYP setup has all three shards on one machine. Refuse data leakage.
    for line in check_partition():
        print(line)
    for client_id in ("client2", "client3"):
        data_yaml = PROJECT_ROOT / "dataset" / client_id / "data.yaml"
        if class_names(data_yaml) != class_names():
            raise ValueError(f"Class IDs/names differ across clients: {data_yaml}")

    model = create_model()
    if args.initial_checkpoint:
        checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["class_names"] != class_names():
            raise ValueError("Checkpoint classes do not match the dataset")
        model.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    strategy = CheckpointingFedAvg(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        accept_failures=False,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model)),
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "local_epochs": args.local_epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
        },
        on_evaluate_config_fn=lambda server_round: {
            "batch": args.batch,
            "imgsz": args.imgsz,
        },
        evaluate_metrics_aggregation_fn=weighted_metrics,
    )
    fl.server.start_server(
        server_address=args.address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    main()
