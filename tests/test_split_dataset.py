"""Small, dependency-free checks for the dataset partitioner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import split_dataset


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "original_dataset"
        self.destination = self.root / "dataset"
        for split in split_dataset.SPLITS:
            for index in range(6):
                name = f"{split}_{index}"
                images = self.source / split / "images"
                labels = self.source / split / "labels"
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                (images / f"{name}.jpg").write_bytes(b"image")
                (labels / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (self.destination / "client1").mkdir(parents=True)
        (self.destination / "client1" / "data.yaml").write_text(
            "path: dataset/client1\ntrain: train/images\nval: valid/images\nnames:\n  0: disease\n"
        )

    def test_rebuild_creates_disjoint_complete_shards_and_backup(self):
        with patch.object(split_dataset, "ROOT", self.root):
            backup, report = split_dataset.rebuild(
                seed=42, destination=self.destination, source=self.source
            )
        self.assertTrue(backup.exists())
        self.assertEqual(len(report), 3)
        self.assertEqual(
            report,
            split_dataset.check_partition(self.destination, self.source),
        )
        self.assertNotIn(
            "path:", (self.destination / "client1" / "data.yaml").read_text()
        )

    def test_overlap_is_rejected(self):
        for client in split_dataset.CLIENTS:
            for split in split_dataset.SPLITS:
                images = self.destination / client / split / "images"
                labels = self.destination / client / split / "labels"
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                (images / f"{split}_0.jpg").write_bytes(b"image")
                (labels / f"{split}_0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        with self.assertRaisesRegex(ValueError, "multiple clients"):
            split_dataset.check_partition(self.destination, self.source)


if __name__ == "__main__":
    unittest.main()
