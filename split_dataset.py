"""Check or reproducibly rebuild the three non-overlapping client datasets."""

import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4


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


def check_partition(destination=DESTINATION, source=SOURCE):
    """Raise on cross-client overlap, missing images, or missing labels."""
    report = []
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
        if seen != original:
            raise ValueError(f"{split}: client images do not match the original set ({len(seen)} vs {len(original)})")
        report.append(f"{split}: {len(original)} distinct images; client counts {counts}")
    return report


def rebuild(seed=42, destination=DESTINATION, source=SOURCE):
    """Build separately, verify, then swap in; keep the old data as a backup."""
    if not (destination.parent.resolve() == ROOT.resolve() and source.parent.resolve() == ROOT.resolve()):
        raise ValueError("Dataset paths must stay directly under the project root")
    template_path = destination / "client1" / "data.yaml"
    if not template_path.is_file():
        raise FileNotFoundError(f"Class template missing: {template_path}")
    yaml_text = "\n".join(
        line for line in template_path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("path:")
    ).strip() + "\n"
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
