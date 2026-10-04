# Agri Vision: 15-Day Thesis and Final Presentation Plan

Prepared on 5 October 2026.

## Main recommendation

Deliver a complete federated system, one well-designed non-YOLO comparison, and a strong explanation of the results. Trying several new architectures in 15 days would put the application, thesis, and presentation at risk.

The supervisor's request provides a useful research question:

> Does a different detector provide enough improvement to justify its additional computational and communication cost?

The alternative does not need to win for the comparison to be valuable.

Choose **Faster R-CNN with ResNet50-FPN** as the one experimental alternative. It provides a clear architectural contrast and a documented PyTorch implementation. Keep the existing system available throughout the experiment.

## Schedule

| Days | Main work | Required outcome |
| --- | --- | --- |
| **1** | Show the supervisor the architecture document and agree on the comparison's scope. Begin the experiment and results chapters. | A clear research question and a fixed experiment plan. |
| **2–3** | Adapt Faster R-CNN to the labels, verify box overlays, test training and checkpoint loading, and profile GPU usage. | A correct pipeline and a measured runtime estimate. |
| **4–6** | Run the bounded comparison, collect metrics, and investigate representative errors. | A comparison table, learning curves, and prediction examples. |
| **7–8** | Finalize the application and inference workflow. | A stable end-to-end demonstration. |
| **9–11** | Complete the thesis and connect the experiments to the objectives. | A complete draft for review. |
| **12–13** | Prepare slides, rehearse questions, and reproduce the project setup. | A confident presentation and reproducible submission. |
| **14–15** | Fix issues, proofread, package files, and rehearse the final demo. | Submission-ready work with a recovery buffer. |

Write throughout the process. Days 9–11 should be for completing and revising the thesis, not starting it. This is a planning schedule; actual training time must be established by profiling.

## A comparison that can be defended

The existing YOLO checkpoint has accumulated substantial training. Comparing it directly with a briefly trained Faster R-CNN would tell us little about architecture quality.

For the new comparison:

- Start both models from their respective pretrained weights.
- Use the same dataset partitions and class definitions.
- Declare the training budget, such as the same number of passes through each client's training data.
- Allow architecture-appropriate batch sizes, and report the differences.
- Select checkpoints using validation data.
- Report mAP50–95, mAP50, precision, recall, per-class results, training time, inference latency, and model size.

The completed YOLO training history can remain a separate main experiment.

### Decision point: end of day 3

If Faster R-CNN cannot complete a correct local training and validation cycle within a practical runtime, reduce the scope.

A centralized comparison between **centralized YOLO and centralized Faster R-CNN** can still answer an architecture question, while the existing federated experiment answers the federated-learning question. Clearly distinguish those studies. Do not compare centralized Faster R-CNN against federated YOLO and attribute every difference to architecture.

## Five defensible thesis contributions

1. A working three-client federated crop-disease detection pipeline.
2. A documented dataset preparation and partitioning process.
3. Evidence showing how performance changes across rounds.
4. An architecture comparison with an explicit computational tradeoff, if the pilot is completed.
5. An honest analysis of weak classes, limitations, and future improvements.

The existing results support meaningful discussion: later rounds did not consistently improve validation performance, and strong overall crop averages can hide classes with very low recall. Explain these findings rather than presenting only the highest score.

Describe proposed experiments as proposed until they have actually been completed. If the alternative-model pilot is not completed, present the architecture review as a feasibility study and future-work recommendation.

## Final presentation: approximately 10–12 slides

1. The agricultural problem and project objectives.
2. Dataset and class coverage.
3. System architecture and one federated round.
4. Why the original detector was selected.
5. Why Faster R-CNN was considered.
6. Experimental setup and fair-comparison rules.
7. Main results and resource costs.
8. Successful and failed detections.
9. Working demonstration.
10. Limitations, conclusion, and future work.

Split dense topics into separate slides where useful, while keeping the story easy to follow.

## Questions to prepare for the viva

- Why federated learning?
- Why this detector?
- What does one round mean?
- How did you check data overlap?
- Why did performance decline?
- What does the architecture comparison actually establish?

Practice the demonstration with several representative images and keep a short recorded backup. Include a difficult example and explain its failure calmly.

## Priorities and stopping rule

**Correct experiments → complete thesis → reliable demo → presentation polish → extra accuracy improvements.**

Set a hard stop on model experimentation after day 6. A well-understood result and a dependable system provide more to defend than a last-minute model with a slightly higher score.

## Supporting documents

- [Architecture review — Word](reports/Agri_Vision_Non_YOLO_Architecture_Review.docx)
- [Architecture review — PDF](reports/Agri_Vision_Non_YOLO_Architecture_Review.pdf)
- [Rounds 26–35 detailed report](reports/rounds_026_035_detailed_report.md)
- [Round-35 non-chili analysis](reports/round_035_non_chili_report.md)
- [Rounds 36–40 Colab upload instructions](reports/rounds_036_040_upload_instructions.md)
