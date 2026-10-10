# Resume EfficientDet after the Colab interruption

The last committed EfficientDet round is 2. The saved status shows interruption during round 3 pooled validation. The Colab termination reason is unconfirmed; a disconnect alone cannot distinguish a session ending from a GPU usage limit.

1. Open `AgriVision_EfficientDet_Resume.ipynb` from the Agri Vision folder in Drive. A local copy also exists at the workspace root.
2. Select a Tesla T4 runtime and reconnect. If Colab refuses GPU access, wait until access is available; no fixed reset time is assumed.
3. Run the notebook cells in order. The notebook already selects main mode, EfficientDet alone and the existing run folder. Keep 10 global rounds, one local epoch and seed 42.
4. Setup verifies the dataset, code and pilot record. Resume rejects changes to the recorded settings, GPU type or package versions. Resolve an explicit mismatch before training; do not bypass the checks.
5. The runner loads `checkpoints/global_round_002.pt`, repeats the incomplete round 3 and continues through round 10. The first three models are complete and are not selected by this notebook.

The original main notebook with an empty `RESUME_RUNS` map starts fresh experiments. Use the prepared resume notebook for this recovery.

EfficientDet's two completed rounds took approximately 18–19 minutes each. Eight remaining rounds suggest roughly 2.5 hours of training and validation, plus setup, assuming similar hardware and throughput. This is an estimate, not a runtime guarantee.

Checkpoint and storage details are in [the results review](README.md). Final held-out evaluation remains disabled until the comparison configurations and valid checkpoints are selected. YOLO's abnormal run needs investigation before drawing an architectural conclusion.

Colab's resource availability and usage limits vary; see [Google's Colab FAQ](https://research.google.com/colaboratory/faq.html).
