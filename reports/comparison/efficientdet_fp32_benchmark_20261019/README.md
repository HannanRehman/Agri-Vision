# EfficientDet full-precision engineering benchmark

Open `AgriVision_EfficientDet_FP32_Benchmark_71gmail.ipynb` as rehmanfx71@gmail.com, select T4, keep search training stopped and Run all. The existing dataset/code are reused in a separate Colab workspace. Extra benchmark code is a separate ZIP; existing search code, configurations, hashes and weights are unchanged.

The original and candidate loops use identical initial weights, fixed augmented training-only batches, optimizer settings and seeds. Both retain FP32, the loss finite check and clipping at norm 10 with nonfinite rejection. The candidate removes the separate per-parameter gradient scan (clipping still rejects nonfinite norms), leaves loss components on GPU until logging, and accumulates loss statistics on GPU. No AMP, batch change, resolution reduction, training reduction or automatic search migration is introduced.

Four batch/resolution pairs (2/4 and 512/640) receive warmup and three alternating-order paired runs of 30 timed steps. Each measured run resets weights, optimizer and random seeds after warmup. The report records latency, throughput, peak allocated GPU memory and all parameter/buffer and per-step loss comparisons. Default tolerances are absolute 1e-7 and relative 1e-6; bitwise equality is also recorded. A failed comparison stops the test. Small-sample agreement does not prove equivalence for every subsequent training step.

Cached CPU batches isolate transfer, forward/backward, optimizer and local logging overhead. Decode/augmentation loading time, validation, aggregation and Drive latency are outside the timing. Identical local logging is used for both loops: this does **not** benchmark a change to Drive status frequency. Test that separately if the first benchmark warrants further work. Model weights are never saved. Reports go only to `benchmark_results/efficientdet_fp32_v1/benchmark.json`. Every existing search file is hashed before and after to detect accidental or concurrent modification.

Review consistency and practical improvement across all four pairs, rather than one favorable run. A useful step speedup warrants a separate representative round estimate and explicit versioned continuation review. Do not edit old saved identities or claim measured full-round gains from this microbenchmark. If there is no meaningful improvement, continue the original search.

Package with `.venv/Scripts/python.exe scripts/package_effdet_benchmark.py`. GPU performance results are pending until the notebook runs on Colab; local CPU checks only establish engineering behavior.
