# SSD-MobileNetV3 grid search on 78gmail

Account: hannanrehman78@gmail.com

Open AgriVision_GridSearch_SSD_MobileNetV3_78gmail.ipynb in Colab, select a T4 GPU, and choose Run all. Authorize Google Drive as 78gmail. The supplied Agri Vision folder must be in My Drive under that exact name.

Required uploads: this notebook, agrivision_grid_search_v1_code.zip, and non_chili_v1_upload_parts/ containing parts.json plus all 14 dataset parts. The notebook verifies dataset and code SHA256 before execution.

Only SSD-MobileNetV3 runs here. Sixteen five-round screening trials are followed automatically by two ten-round confirmations and a fresh 30-round run of the selected configuration. Reopen the same notebook and Run all to resume. Keep the uploads and settings unchanged.

40gmail continues SSDLite separately. Its original SSD pair notebook also queues SSD-MobileNetV3 after SSDLite finishes all stages; stop that session before it starts the queued model to avoid a duplicate experiment. No changes were made to its active session.

This preparation does not launch GPU training. No prior SSD-MobileNetV3 grid-search checkpoints existed on 40gmail when the separate notebook was prepared. The 30-round baseline checkpoints are separate and are not search initialization.
