"""Create draft experiment scaffolds without loading models or training."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return  # Preserve resolved configs and records in an existing workspace.
    path.write_text(value, encoding='utf-8')

def main():
    shared = {
        'status': 'draft_not_consumed_by_existing_training_code',
        'task': 'object_detection', 'dataset_version': 'non_chili_v1',
        'manifest_directory': 'manifests/non_chili_v1', 'dataset_ready': False,
        'foreground_classes': 24, 'excluded_source_class_ids': list(range(13,19)),
        'regime': 'sequential_three_client_fedavg', 'seed': 42,
        'clients': ['client1','client2','client3'],
        'initial_round_budget': 10, 'local_epochs': 1,
        'aggregation_weight': 'client_training_image_count',
        'checkpoint_selection': 'pooled_validation_map_50_95',
        'evaluation': {'primary':'coco_box_map_50_95', 'also':['map50','per_class_ap','precision','recall','f1'],
                       'precision_recall_iou': .5, 'precision_recall_confidence': .25,
                       'max_detections_per_image': 100},
        'hardware': 'same_GPU_type_for_all_runs_to_be_recorded',
        'optimizer_and_normalization_policy': 'finalize_after_pilots',
        'runtime_profile_required': True, 'test_set_tuning': False,
        'thesis_writing': 'after_2026-10-17', 'submission_deadline': None,
    }
    models = {
        'yolo_v8n': {'display_name':'YOLOv8n', 'implementation':'ultralytics_yolov8n',
                     'initial_weights':'yolov8n.pt', 'input_size_candidate':640,
                     'head':'24_class_yolo_detection_head'},
        'efficientdet_d0': {'display_name':'EfficientDet-D0', 'implementation':'effdet_adapter_to_be_verified',
                            'initial_weights':'COCO_pretrained_D0_source_and_hash_to_record',
                            'input_size_candidate':512, 'head':'EfficientDet_24_class_detection_head'},
        'ssdlite': {'display_name':'SSDLite320 MobileNetV3-Large',
                    'implementation':'torchvision_ssdlite320_mobilenet_v3_large',
                    'initial_weights':'COCO_pretrained_then_replace_class_head',
                    'input_size_candidate':320, 'head':'lightweight_depthwise_separable_SSD_head',
                    'output_classes_including_background':25},
        'ssd_mobilenetv3': {'display_name':'SSD MobileNetV3-Large (standard head)',
                            'implementation':'custom_torchvision_SSD_configuration_to_be_verified',
                            'initial_weights':'backbone_source_to_record_standard_head_has_no_assumed_pretrained_weights',
                            'input_size_candidate':320, 'head':'standard_convolutional_SSD_head',
                            'output_classes_including_background':25,
                            'limitation':'Head initialization differs from pretrained full-detector alternatives; disclose in comparison'},
    }
    write_new(ROOT/'configs/comparison/protocol.json', json.dumps(shared,indent=2)+'\n')
    for key, spec in models.items():
        run_id = f'{key}_non_chili_seed42_run01'
        config = {'status':'planned_not_implemented', 'model_id':key,
                  'shared_protocol':'configs/comparison/protocol.json',
                  'run_directory':f'experiments/{run_id}',
                  'results_directory':f'experiments/{run_id}/results',
                  'checkpoints_directory':f'experiments/{run_id}/checkpoints',
                  'logs_directory':f'experiments/{run_id}/logs', **spec}
        write_new(ROOT/f'configs/comparison/{key}.json', json.dumps(config,indent=2)+'\n')
        folder = ROOT/'experiments'/run_id
        for name in ['checkpoints','logs','results/metrics','results/figures','results/predictions','results/evaluations']:
            (folder/name).mkdir(parents=True, exist_ok=True)
            (folder/name/'.gitkeep').touch(exist_ok=True)
        write_new(folder/'README.md',
            f'# {spec["display_name"]}: non-chili comparison\n\n'
            f'Run ID: `{run_id}`. **Status: planned; no model trained.**\n\n'
            f'Model specification: [configuration](../../configs/comparison/{key}.json). '
            'Shared settings: [protocol](../../configs/comparison/protocol.json). '
            'Dataset: [non_chili_v1](../../manifests/non_chili_v1/README.md), not yet prepared.\n\n'
            'Use checkpoints/ for best and last global weights and logs/ for progress. '
            'Each model has a separate results/ folder: metrics/ for round/client CSVs and summaries, '
            'figures/ for charts, predictions/ for selected examples and evaluations/ for final evaluation exports. '
            'Save actual environment versions, initialization, manifest checksum, hardware and timing before training. '
            'Draft configs are specifications, not supported CLI arguments for the existing YOLO runner.\n')
        write_new(folder/'results/README.md',
            f'# Results: {spec["display_name"]}\n\n**Status: empty scaffold; no results yet.**\n\n'
            'metrics/ contains round/client scores and final summaries; figures/ contains charts; '
            'predictions/ contains annotated examples; evaluations/ contains evaluator exports. '
            'These outputs belong only to this model/run. Checkpoints and progress logs are kept in the parent run folder.\n')
    write_new(ROOT/'configs/comparison/README.md',
        '# Comparison configurations\n\n**Status: draft specifications, not runnable training configurations.**\n\n'
        'protocol.json defines the intended common dataset, clients, budget and evaluation. '
        'The four model JSON files specify distinct one-stage detectors and candidate resolutions. '
        'The separate SSD–MobileNetV3 configuration still needs implementation and initialization checks. '
        'Finalize settings after pilots and record the resolved configuration in each run. '
        'The existing model.py/colab_train.py do not read these files.\n')
    write_new(ROOT/'manifests/non_chili_v1/README.md',
        '# non_chili_v1 dataset manifests\n\n**Status: reserved; dataset preparation has not run.**\n\n'
        'The four crops will be banana, cauliflower, groundnut and radish, with 24 foreground classes. '
        'Original class IDs 13–18 (chili) will be excluded. IDs 0–12 stay unchanged; IDs 19–29 become 13–23.\n\n'
        'Before training, populate this folder with audited image lists for train/validation/test and clients, '
        'old-to-new class mapping, source-group/duplicate decisions, exclusion reasons, counts and checksums. '
        'Preserve the original 30-class source and existing partitions. Do not treat this README as a frozen manifest.\n')
    write_new(ROOT/'experiments/README.md',
        '# New one-stage detector comparison\n\n**All four folders are scaffolds; no comparison run has started.**\n\n'
        '- yolo_v8n_non_chili_seed42_run01/\n'
        '- efficientdet_d0_non_chili_seed42_run01/\n'
        '- ssdlite_non_chili_seed42_run01/\n'
        '- ssd_mobilenetv3_non_chili_seed42_run01/\n\n'
        'Each experiment has its own README, checkpoints/, logs/ and results/ directories. Use a new run ID for a materially different '
        'configuration; preserve completed results. Match these IDs in Colab/Google Drive. '
        'Track compact reviewed metrics and configuration in Git, with large weights/images/logs ignored.\n')
    print('Created four planned run folders, five draft JSON configs and the dataset manifest placeholder.')

if __name__ == '__main__':
    main()
