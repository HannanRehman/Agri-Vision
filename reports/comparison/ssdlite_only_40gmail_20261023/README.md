# 40gmail: SSDLite only

The existing AgriVision_GridSearch_SSD_Pair.ipynb retains its name/link but now selects only ssdlite and stops after its final 30-round run. SSD-MobileNetV3 remains a separate run on 78gmail. No training was started during this update.

The saved plan.json previously queued both architectures. It was backed up in the same Drive output folder, then only its top-level models list was changed from [ssdlite, ssd_mobilenetv3] to [ssdlite]. All other plan fields remain identical, including the full search grid, data hash, code hash and hardware. This preserves the original controller's strict resume checks. No checkpoints, settings, configurations, metrics or MobileNet results were removed.

Reopen the updated notebook from Drive before Run all. Completed screening and confirmation are validated/skipped, final training resumes from its last committed round, and the notebook exits normally after SSDLite round 30.
