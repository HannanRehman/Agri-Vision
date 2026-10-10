import unittest
import tempfile
from pathlib import Path

from PIL import Image

from scripts.prepare_non_chili_dataset import build, choose_records, parse_labels, remap_text, source_identity, save_audit, sha_file
from scripts.verify_non_chili_dataset import verify


def record(key, split, client, pixels, source, boxes=None):
    return dict(key=key, split=split, original_client=client, client=client,
                image_sha256=pixels, decoded_sha256=pixels, source_identity=source,
                boxes=boxes or parse_labels('19 0.5 0.5 0.4 0.4', 30), reason=None)


class DatasetPreparationTests(unittest.TestCase):
    def test_remapping_preserves_coordinate_tokens(self):
        rows = parse_labels('19 0.50000 0.5 0.40 0.400\n29 0.25 0.25 0.1 0.1', 30)
        self.assertEqual(remap_text(rows, {19: 13, 29: 23}),
                         '13 0.50000 0.5 0.40 0.400\n23 0.25 0.25 0.1 0.1\n')

    def test_invalid_boxes_rejected(self):
        for line in ['30 0.5 0.5 0.1 0.1', '0 nan 0.5 0.1 0.1',
                     '0 0.1 0.5 0.8 0.1', '0 0.5 0.5 0 0.1']:
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_labels(line, 30)

    def test_source_identity_removes_export_hash_and_keeps_crop(self):
        self.assertEqual(source_identity('leaf.jpg.rf.AB123.jpg', {'Banana'}),
                         'Banana:leaf.jpg')
        self.assertNotEqual(source_identity('leaf.jpg', {'Banana'}),
                            source_identity('leaf.jpg', {'Radish'}))

    def test_holdout_priority_never_moves_training_into_test(self):
        rows = [record('train/a', 'train', 'client1', 'a', 'leaf'),
                record('valid/b', 'valid', 'client2', 'b', 'leaf'),
                record('test/c', 'test', 'client3', 'c', 'leaf')]
        choose_records(rows)
        self.assertEqual([r['key'] for r in rows if r['reason'] is None], ['test/c'])
        self.assertEqual(rows[0]['split'], 'train')

    def test_visually_confirmed_pairs_group_different_filenames_and_pixels(self):
        rows = [record('train/a', 'train', 'client1', 'a', 'name_a'),
                record('test/b', 'test', 'client2', 'b', 'name_b')]
        choose_records(rows, [('train/a','test/b')])
        self.assertEqual(rows[0]['reason'], 'source_group_cross_split_overlap')
        self.assertIsNone(rows[1]['reason'])
        self.assertEqual(rows[0]['group_id'], rows[1]['group_id'])

    def test_conflicting_identical_pixels_quarantine_source_group(self):
        rows = [record('train/a', 'train', 'client1', 'same', 'leaf'),
                record('valid/b', 'valid', 'client2', 'same', 'leaf',
                       parse_labels('20 0.5 0.5 0.4 0.4', 30)),
                record('test/c', 'test', 'client3', 'different', 'leaf')]
        _, conflicts = choose_records(rows)
        self.assertEqual(len(conflicts), 1)
        self.assertTrue(all(r['reason'] == 'conflicting_annotations_on_identical_image' for r in rows))

    def test_source_variants_have_one_client_and_exact_duplicates_removed(self):
        rows = [record('train/a', 'train', 'client2', 'a', 'leaf'),
                record('train/b', 'train', 'client2', 'b', 'leaf'),
                record('train/c', 'train', 'client1', 'c', 'leaf'),
                record('train/d', 'train', 'client1', 'a', 'other')]
        changes, _ = choose_records(rows)
        kept = [r for r in rows if r['reason'] is None]
        self.assertEqual(len(kept), 3)
        self.assertEqual({r['client'] for r in kept}, {'client2'})
        self.assertEqual(len({r['group_id'] for r in kept}), 1)
        self.assertEqual(len(changes), 1)

    def test_build_portable_manifests_and_detect_output_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = root / 'manifests/non_chili_v1'
            old_ids = [i for i in range(30) if i not in range(13, 19)]
            mapping = dict(zip(old_ids, range(24)))
            records = []
            for split in ['train', 'valid', 'test']:
                for index, client in enumerate(['client1', 'client2', 'client3']):
                    path = root / 'source' / split / f'{client}.png'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (16, 16), (index * 60, len(records) * 20, 0)).save(path)
                    label = path.with_suffix('.txt')
                    text = ''.join(f'{i} 0.5000 0.5 0.1 0.1\n' for i in old_ids)
                    label.write_text(text)
                    row = record(f'{split}/{path.name}', split, client, sha_file(path), f'{split}-{client}', parse_labels(text, 30))
                    row.update(filename=path.name, source_image=path.relative_to(root).as_posix(),
                               source_label=label.relative_to(root).as_posix(), label_sha256=sha_file(label),
                               width=16, height=16, group_id=row['key'])
                    records.append(row)
            summary = dict(class_map=mapping, new_classes={v: f'Crop_class{k}' for k,v in mapping.items()},
                           source_classes={i: f'Crop_class{i}' for i in range(30)},
                           missing_retained_classes={s: [] for s in ['train','valid','test']},
                           near_duplicate_candidate_pairs=[])
            save_audit(manifests, records, summary)
            build(root, manifests, records, summary, 2)
            self.assertEqual(verify(root)['images'], 9)
            self.assertFalse((root/'dataset_staging_non_chili_v1').exists())
            first_label = root/'original_dataset_non_chili/train/labels/client1.txt'
            first_label.write_text('0 0.5 0.5 0.2 0.2\n')
            with self.assertRaises(AssertionError):
                verify(root)


if __name__ == '__main__':
    unittest.main()
