"""Check or reproducibly rebuild the three non-overlapping client datasets."""

import argparse
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "original_dataset"
DESTINATION = ROOT / "dataset"
CLIENTS = ("client1", "client2", "client3")
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_names(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    return {path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}


def class_names(config):
    """Read consecutive YOLO class IDs from a dataset configuration."""
    with Path(config).open(encoding="utf-8") as stream:
        raw_names = yaml.safe_load(stream)["names"]
    names = {int(index): str(name) for index, name in raw_names.items()}
    if sorted(names) != list(range(len(names))):
        raise ValueError(f"Class IDs must be consecutive and start at zero: {config}")
    if len(set(names.values())) != len(names):
        raise ValueError(f"Class names must be unique: {config}")
    return names


def validate_label(label_path, class_count):
    """Reject labels that Ultralytics would otherwise silently ignore."""
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 values, found {len(values)}")
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: label values must be numeric") from error
        if not 0 <= class_id < class_count:
            raise ValueError(
                f"{label_path}:{line_number}: class ID {class_id} is outside 0-{class_count - 1}"
            )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: coordinates must be finite")
        x_center, y_center, width, height = coordinates
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            raise ValueError(f"{label_path}:{line_number}: box center must be within 0-1")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{label_path}:{line_number}: box size must be within 0-1")


def check_partition(destination=DESTINATION, source=SOURCE):
    """Raise on overlap, missing data, inconsistent classes, or invalid labels."""
    expected_names = class_names(source / "data.yaml")
    report = []
    validated_labels = set()
    for client in CLIENTS:
        client_names = class_names(destination / client / "data.yaml")
        if client_names != expected_names:
            raise ValueError(f"{client}: class IDs/names do not match {source / 'data.yaml'}")
    for split in SPLITS:
        original = image_names(source / split / "images")
        seen = set()
        counts = []
        for client in CLIENTS:
            client_root = destination / client / split
            names = image_names(client_root / "images")
            overlap = seen & names
            if overlap:
                raise ValueError(f"{split}: {len(overlap)} images appear in multiple clients; rebuild the dataset")
            seen.update(names)
            counts.append(len(names))
            labels = client_root / "labels"
            if not labels.is_dir():
                raise FileNotFoundError(f"Missing labels directory: {labels}")
            missing_labels = [name for name in names if not (labels / f"{Path(name).stem}.txt").is_file()]
            if missing_labels:
                raise ValueError(f"{client}/{split}: {len(missing_labels)} images have no label file")
            for name in names:
                label_path = labels / f"{Path(name).stem}.txt"
                if label_path not in validated_labels:
                    validate_label(label_path, len(expected_names))
                    validated_labels.add(label_path)
        if seen != original:
            raise ValueError(f"{split}: client images do not match the original set ({len(seen)} vs {len(original)})")
        report.append(f"{split}: {len(original)} distinct images; client counts {counts}")
    report.append(f"classes: {len(expected_names)} consecutive IDs; all labels valid")
    return report


def rebuild(seed=42, destination=DESTINATION, source=SOURCE):
    """Build separately, verify, then swap in; keep the old data as a backup."""
    if not (destination.parent.resolve() == ROOT.resolve() and source.parent.resolve() == ROOT.resolve()):
        raise ValueError("Dataset paths must stay directly under the project root")
    source_config = source / "data.yaml"
    template_path = source_config if source_config.is_file() else ROOT / "data.yaml"
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Class configuration missing: expected {source_config} or {ROOT / 'data.yaml'}"
        )
    yaml_text = "\n".join(
        line for line in template_path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("path:")
    ).strip() + "\n"
    source_config.parent.mkdir(parents=True, exist_ok=True)
    source_config.write_text(yaml_text, encoding="utf-8")
    stage = ROOT / f"dataset_staging_{uuid4().hex[:8]}"
    backup = ROOT / f"dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if stage.exists() or backup.exists():
        raise FileExistsError("Staging or backup target already exists")

    for split in SPLITS:
        original = sorted(image_names(source / split / "images"))
        if not original:
            raise ValueError(f"No source images for {split}")
        random.Random(f"{seed}:{split}").shuffle(original)
        for index, client in enumerate(CLIENTS):
            images_dir = stage / client / split / "images"
            labels_dir = stage / client / split / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            for name in original[index::len(CLIENTS)]:
                label = f"{Path(name).stem}.txt"
                source_label = source / split / "labels" / label
                if not source_label.is_file():
                    raise FileNotFoundError(f"Missing source label: {source_label}")
                shutil.copy2(source / split / "images" / name, images_dir / name)
                shutil.copy2(source_label, labels_dir / label)
    for client in CLIENTS:
        (stage / client / "data.yaml").write_text(yaml_text, encoding="utf-8")
    report = check_partition(stage, source)

    destination.rename(backup)
    try:
        stage.rename(destination)
    except OSError:
        backup.rename(destination)
        raise
    return backup, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Create clean shards and keep the current dataset as a backup")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed")
    args = parser.parse_args()
    if args.rebuild:
        backup, report = rebuild(args.seed)
        print(f"Old dataset preserved at {backup}")
    else:
        report = check_partition()
    for line in report:
        print(line)


if __name__ == "__main__":
    main()
