"""Evaluate a saved global checkpoint on the held-out original test split."""

import argparse

import torch

from model import PROJECT_ROOT, class_names, create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    if args.batch < 1 or args.imgsz < 1:
        parser.error("batch and imgsz must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != class_names():
        raise ValueError("Checkpoint classes do not match the current dataset")
    model = create_model()
    model.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = model.val(
        data=str(PROJECT_ROOT / "original_dataset" / "data.yaml"),
        split="test",
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
    )
    print(f"Test mAP50: {metrics.box.map50:.4f}")
    print(f"Test mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
