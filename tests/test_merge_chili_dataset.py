import tempfile
import unittest
from pathlib import Path

from PIL import Image

import merge_chili_dataset


class ChiliMergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_source_identity_ignores_roboflow_variant_hash(self):
        path = Path("leaf_jpg.rf.0123456789abcdef.jpg")
        self.assertEqual(merge_chili_dataset.source_identity(path), "leaf_jpg")

    def test_label_ids_are_remapped_without_changing_boxes(self):
        label = self.root / "sample.txt"
        label.write_text("2 0.5 0.4 0.3 0.2\n5 0.1 0.2 0.3 0.4\n", encoding="utf-8")
        text = merge_chili_dataset.remapped_label_text(label, {2: 13, 5: 18})
        self.assertEqual(text, "13 0.5 0.4 0.3 0.2\n18 0.1 0.2 0.3 0.4\n")

    def test_variant_with_smaller_rotation_border_is_preferred(self):
        source = self.root / "source"
        for split in merge_chili_dataset.SOURCE_SPLITS:
            (source / split / "images").mkdir(parents=True)
            (source / split / "labels").mkdir(parents=True)

        black_border = source / "train" / "images" / "leaf_jpg.rf.aaaaaaaa.jpg"
        clean = source / "train" / "images" / "leaf_jpg.rf.bbbbbbbb.jpg"
        Image.new("RGB", (40, 40), "black").save(black_border)
        Image.new("RGB", (40, 40), "white").save(clean)
        for image in (black_border, clean):
            (source / "train" / "labels" / f"{image.stem}.txt").write_text(
                "2 0.5 0.5 0.25 0.25\n", encoding="utf-8"
            )

        result = merge_chili_dataset.collect_candidates(
            source,
            {0: 14, 1: 15, 2: 13, 3: 17, 4: 16, 5: 18},
        )
        processed, source_count, empty_count, _, selected = result
        self.assertEqual((processed, source_count, empty_count), (2, 1, 0))
        self.assertEqual(selected[0][1], clean)


if __name__ == "__main__":
    unittest.main()
