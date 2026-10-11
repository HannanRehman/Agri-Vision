# SSDLite and SSD-MobileNetV3 on 40gmail

Account: hannanrehman40@gmail.com. Open [the uploaded notebook](https://drive.google.com/file/d/1M9r_n6eLmLNW4hP6TXS1LAa33pn8ug_L/view) in Colab using this account, select T4 GPU and Run all. Authorize Drive using 40gmail. Keep the shared Agri Vision folder in My Drive under that exact name.

Only SSDLite and SSD-MobileNetV3 run. Both sources are saved at global round 10, so round 11 is next. SSDLite runs first toward round 30; SSD-MobileNetV3 then resumes toward round 30. They run sequentially on one GPU, with separate `_to30` result folders. Original source folders remain untouched. An interruption repeats an incomplete global round; Run all resumes from each latest completed checkpoint.

The folder contains the notebook, existing verified code ZIP, fourteen dataset parts plus manifest, and both original model source folders under comparison_results. Each source includes unchanged settings/latest, initial and round-10 checkpoint files, and all ten round JSONs plus the global CSV. Best and latest source checkpoints are both round 10. Colab preflight checks checkpoint hashes, original implementation, dataset, model specification, package versions and GPU before training. No additional pilot is required.

Expected remaining training time from original T4 measurements: SSDLite about 2 hours 41 minutes; SSD-MobileNetV3 about 2 hours 37 minutes. Total about 5 hours 18 minutes plus setup/Drive overhead. This is an estimate, not a runtime guarantee.

Uploads/copies made through the connected 4gmail account are initially 4gmail-owned; 40gmail has Editor access. To move storage usage to 40gmail, transfer ownership of all uploaded files, including nested dataset parts and checkpoint/metric files, from 4gmail and accept on 40gmail. Folder ownership alone does not transfer contained files. Original files in 4gmail were preserved.

Readback verification: notebook matches local bytes; copied settings/latest match original content; dataset parts, weights and metrics match original filenames/counts/byte sizes. Full dataset and checkpoint hash checks are performed by Colab before training. No GPU training was started during upload.
