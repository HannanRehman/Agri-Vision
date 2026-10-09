"""Safely merge the vetted Roboflow chili dataset into AgriVision training data.

The Roboflow export contains three rotated variants for most training images.  This
script keeps one variant per source image, maps the six source classes onto the
existing AgriVision class IDs, removes exact duplicates, and appends only to the
training split.  Existing validation and test images are never modified.
"""

import argparse
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from split_dataset import CLIENTS, IMAGE_SUFFIXES, check_partition, class_names, validate_label


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "archive/imported_datasets/PENYAKIT PADA CABAI.v1i.yolov8"
ORIGINAL = ROOT / "original_dataset"
CLIENT_DATA = ROOT / "dataset"
SOURCE_SPLITS = ("train", "valid", "test")
SOURCE_NAMES = {
    0: "Cabai-Sehat",
    1: "Cabai-Terkena-Daun-Berkerut",
    2: "Cabai-Terkena-Penyakit-Anthacnose",
    3: "Cabai-Terkena-Penyakit-Kutu-Putih",
    4: "Cabai-Terkena-Penyakit-Leaf-Spot",
    5: "Cabai-Terkena-Penyakit-Yellowish",
}
TARGET_NAMES = {
    0: "Chili_healthy",
    1: "Chili_leaf_curl",
    2: "Chili_anthracnose",
    3: "Chili_whitefly",
    4: "Chili_leaf_spot",
    5: "Chili_yellowish",
}
SOURCE_ID_PATTERN = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


def source_identity(image_path):
    """Return the pre-augmentation Roboflow source name."""
    return SOURCE_ID_PATTERN.sub("", image_path.stem)


def read_source_names(source):
    with (source / "data.yaml").open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    names = raw.get("names")
    if isinstance(names, list):
        names = {index: str(name) for index, name in enumerate(names)}
    else:
        names = {int(index): str(name) for index, name in names.items()}
    if names != SOURCE_NAMES:
        raise ValueError(f"Unexpected source class mapping: {names}")
    return names


def image_for_label(images_dir, label_path):
    matches = [
        path
        for path in images_dir.glob(f"{label_path.stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {label_path}, found {len(matches)}")
    return matches[0]


def border_black_score(image_path):
    """Prefer the rotated variant with the smallest black border."""
    with Image.open(image_path) as image:
        image = image.convert("L")
        width, height = image.size
        edge = max(1, min(width, height) // 20)
        regions = (
            (0, 0, width, edge),
            (0, height - edge, width, height),
            (0, edge, edge, height - edge),
            (width - edge, edge, width, height - edge),
        )
        black = 0
        total = 0
        for box in regions:
            crop = image.crop(box)
            histogram = crop.histogram()
            black += sum(histogram[:9])
            total += crop.width * crop.height
        return black / total


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remapped_label_text(label_path, target_ids):
    lines = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = line.split()
        source_id = int(values[0])
        values[0] = str(target_ids[source_id])
        lines.append(" ".join(values))
    return "\n".join(lines) + "\n"


def collect_candidates(source, target_ids):
    variants = defaultdict(list)
    processed_count = 0
    for split in SOURCE_SPLITS:
        images_dir = source / split / "images"
        labels_dir = source / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing YOLO split directories under {source / split}")
        labels = sorted(labels_dir.glob("*.txt"))
        images = [
            path for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(images) != len(labels):
            raise ValueError(f"{split}: {len(images)} images but {len(labels)} labels")
        for label_path in labels:
            validate_label(label_path, len(SOURCE_NAMES))
            image_path = image_for_label(images_dir, label_path)
            variants[source_identity(image_path)].append((image_path, label_path))
            processed_count += 1

    source_count = len(variants)
    empty_source_count = 0
    selected = []
    for identity, choices in sorted(variants.items()):
        image_path, label_path = min(
            choices,
            key=lambda pair: (border_black_score(pair[0]), str(pair[0])),
        )
        if not label_path.read_text(encoding="utf-8").strip():
            empty_source_count += 1
            continue
        selected.append((identity, image_path, label_path))

    return processed_count, source_count, empty_source_count, variants, selected


def existing_hashes(candidate_sizes):
    """Hash only existing files that could be byte-identical to a candidate."""
    hashes = defaultdict(list)
    for split in SOURCE_SPLITS:
        images_dir = ORIGINAL / split / "images"
        for image_path in images_dir.iterdir():
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_SUFFIXES
                and image_path.stat().st_size in candidate_sizes
            ):
                hashes[file_digest(image_path)].append(image_path)
    return hashes


def label_class_ids(label_path):
    return {
        int(line.split()[0])
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def target_filename(image_path, digest):
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.stem)
    return f"chili_rf_v1_{safe_stem}_{digest[:10]}{image_path.suffix.lower()}"


def audit(source):
    read_source_names(source)
    destination_names = class_names(ORIGINAL / "data.yaml")
    name_to_id = {name: class_id for class_id, name in destination_names.items()}
    missing = sorted(set(TARGET_NAMES.values()) - set(name_to_id))
    if missing:
        raise ValueError(f"Target dataset is missing classes: {missing}")
    target_ids = {source_id: name_to_id[name] for source_id, name in TARGET_NAMES.items()}
    (
        processed_count,
        source_count,
        empty_source_count,
        variants,
        selected,
    ) = collect_candidates(source, target_ids)

    all_with_hashes = [
        (identity, image_path, label_path, file_digest(image_path))
        for identity, choices in variants.items()
        for image_path, label_path in choices
    ]
    candidate_sizes = {
        image_path.stat().st_size for _, image_path, _, _ in all_with_hashes
    }
    known_hashes = existing_hashes(candidate_sizes)

    duplicate_identities = set()
    conflicting_identities = set()
    for identity, _, label_path, digest in all_with_hashes:
        if digest in known_hashes:
            expected_ids = {target_ids[class_id] for class_id in label_class_ids(label_path)}
            for existing_image in known_hashes[digest]:
                existing_label = (
                    existing_image.parent.parent / "labels" / f"{existing_image.stem}.txt"
                )
                if label_class_ids(existing_label) != expected_ids:
                    conflicting_identities.add(identity)
            duplicate_identities.add(identity)

    annotated_identities = {identity for identity, _, _ in selected}
    duplicate_identities &= annotated_identities
    conflicting_identities &= annotated_identities

    accepted = []
    accepted_hashes = {}
    internal_duplicate_count = 0
    for identity, image_path, label_path in selected:
        if identity in duplicate_identities or identity in conflicting_identities:
            continue
        digest = file_digest(image_path)
        expected_ids = {target_ids[class_id] for class_id in label_class_ids(label_path)}
        if digest in accepted_hashes:
            previous_ids = accepted_hashes[digest]
            if previous_ids != expected_ids:
                raise ValueError(
                    f"Byte-identical new images have conflicting labels: {image_path.name}"
                )
            internal_duplicate_count += 1
            continue
        accepted_hashes[digest] = expected_ids
        accepted.append((identity, image_path, label_path, digest))

    image_counts = Counter()
    box_counts = Counter()
    for _, _, label_path, _ in accepted:
        ids = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                target_id = target_ids[int(line.split()[0])]
                ids.append(target_id)
                box_counts[target_id] += 1
        image_counts.update(set(ids))
    return (
        target_ids,
        processed_count,
        source_count,
        empty_source_count,
        selected,
        accepted,
        len(duplicate_identities - conflicting_identities),
        len(conflicting_identities),
        internal_duplicate_count,
        image_counts,
        box_counts,
    )


def append_to_clients(added_names):
    client_counts = {
        client: len([
            path for path in (CLIENT_DATA / client / "train" / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ])
        for client in CLIENTS
    }
    assignments = Counter()
    copied = []
    try:
        for image_name in sorted(added_names):
            client = min(CLIENTS, key=lambda item: (client_counts[item], item))
            source_image = ORIGINAL / "train" / "images" / image_name
            source_label = ORIGINAL / "train" / "labels" / f"{Path(image_name).stem}.txt"
            target_image = CLIENT_DATA / client / "train" / "images" / image_name
            target_label = CLIENT_DATA / client / "train" / "labels" / source_label.name
            shutil.copy2(source_image, target_image)
            shutil.copy2(source_label, target_label)
            copied.extend((target_image, target_label))
            client_counts[client] += 1
            assignments[client] += 1
    except Exception:
        for path in reversed(copied):
            path.unlink(missing_ok=True)
        raise
    return assignments, copied


def apply_merge(source, result):
    (
        target_ids,
        processed_count,
        source_count,
        empty_source_count,
        selected,
        accepted,
        duplicate_count,
        conflict_count,
        internal_duplicate_count,
        _,
        _,
    ) = result
    train_images = ORIGINAL / "train" / "images"
    train_labels = ORIGINAL / "train" / "labels"
    added_paths = []
    added_names = []
    per_class = Counter()
    try:
        for _, image_path, label_path, digest in accepted:
            image_name = target_filename(image_path, digest)
            target_image = train_images / image_name
            target_label = train_labels / f"{Path(image_name).stem}.txt"
            if target_image.exists() or target_label.exists():
                raise FileExistsError(f"Merge target already exists: {target_image}")
            shutil.copy2(image_path, target_image)
            target_label.write_text(remapped_label_text(label_path, target_ids), encoding="utf-8")
            validate_label(target_label, len(class_names(ORIGINAL / "data.yaml")))
            added_paths.extend((target_image, target_label))
            added_names.append(image_name)
            class_ids = {
                int(line.split()[0])
                for line in target_label.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            per_class.update(class_ids)

        assignments, client_paths = append_to_clients(added_names)
        added_paths.extend(client_paths)
        check_partition(CLIENT_DATA, ORIGINAL)
    except Exception:
        for path in reversed(added_paths):
            path.unlink(missing_ok=True)
        raise

    return {
        "processed": processed_count,
        "source_images": source_count,
        "empty_sources": empty_source_count,
        "exact_duplicates": duplicate_count,
        "conflicting_sources": conflict_count,
        "internal_duplicates": internal_duplicate_count,
        "added": len(accepted),
        "per_class": per_class,
        "client_assignments": assignments,
    }


def print_audit(result, target_names):
    (
        _,
        processed_count,
        source_count,
        empty_source_count,
        selected,
        accepted,
        duplicate_count,
        conflict_count,
        internal_duplicate_count,
        image_counts,
        box_counts,
    ) = result
    print(f"Roboflow processed images: {processed_count}")
    print(f"Independent source images: {source_count}")
    print(f"Unannotated source images skipped: {empty_source_count}")
    print(f"Sources already present and skipped: {duplicate_count}")
    print(f"Conflicting-label sources skipped: {conflict_count}")
    print(f"Repeated new image files skipped: {internal_duplicate_count}")
    print(f"Images ready to add: {len(accepted)}")
    for class_id in sorted(image_counts):
        print(
            f"  {class_id:2d} {target_names[class_id]}: "
            f"{image_counts[class_id]} images, {box_counts[class_id]} boxes"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--apply", action="store_true", help="Perform the audited merge")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        parser.error(
            "Historical chili source folder is unavailable. Its unused extraction was removed "
            "during workspace cleanup. To reproduce the old merge, extract the retained ZIP "
            "from archive/imported_datasets and pass the restored folder with --source."
        )
    result = audit(source)
    destination_names = class_names(ORIGINAL / "data.yaml")
    print_audit(result, destination_names)
    if not args.apply:
        print("Dry run only; pass --apply to merge and update client training shards.")
        return

    summary = apply_merge(source, result)
    print(f"Added {summary['added']} images to original_dataset/train")
    print("Client additions: " + ", ".join(
        f"{client}={summary['client_assignments'][client]}" for client in CLIENTS
    ))
    print("Validation and test splits were unchanged.")


if __name__ == "__main__":
    main()
