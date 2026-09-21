"""Run inference with an aggregated global checkpoint."""

import argparse

import torch

from model import PROJECT_ROOT, class_names, create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="saved_models/global_round_XXX.pt")
    parser.add_argument("--source", required=True, help="Image, directory, or video path")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != class_names():
        raise ValueError("Checkpoint classes do not match the current dataset")
    model = create_model()
    model.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.predict(
        source=args.source,
        device=args.device,
        conf=args.conf,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "predictions"),
    )


if __name__ == "__main__":
    main()
