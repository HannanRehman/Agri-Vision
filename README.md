# 🌾 Agri Vision

### AI System for Crop Prediction and Disease Detection using YOLOv8 and Federated Learning

---

## Overview

Agri Vision is a Final Year Project (FYP) developed as part of the Bachelor of Software Engineering program. The project aims to leverage Artificial Intelligence to improve agricultural decision-making through intelligent crop disease detection while maintaining data privacy using Federated Learning.

Traditional machine learning approaches require all training data to be centralized, which raises concerns regarding privacy, ownership, and scalability. Agri Vision addresses these challenges by implementing a decentralized learning framework where multiple clients collaboratively train a global model without sharing their local datasets.

The project combines the accuracy of **YOLOv8** for disease detection with the **Flower Federated Learning Framework** and **FedAvg** aggregation algorithm to create a scalable and privacy-preserving agricultural AI system.

---

# Project Objectives

* Develop an intelligent crop disease detection system.
* Implement Federated Learning to preserve data privacy.
* Improve disease detection performance using YOLOv8.
* Simulate distributed learning across multiple clients.
* Evaluate model performance using standard object detection metrics.
* Demonstrate the feasibility of AI-driven smart agriculture.

---

# Key Features

* AI-based crop disease detection
* Privacy-preserving Federated Learning
* Distributed client-server architecture
* YOLOv8 object detection model
* Flower framework integration
* FedAvg global model aggregation
* Multi-client training simulation
* Performance evaluation using industry-standard metrics
* Modular project structure for future scalability

---

# Technologies Used

| Category             | Technology           |
| -------------------- | -------------------- |
| Programming Language | Python               |
| Deep Learning        | YOLOv8 (Ultralytics) |
| Federated Learning   | Flower Framework     |
| Machine Learning     | PyTorch              |
| Image Processing     | OpenCV               |
| Numerical Computing  | NumPy                |
| Data Handling        | Pillow               |
| Visualization        | Matplotlib           |

---

# System Architecture

The project follows a Federated Learning architecture consisting of a central server and multiple distributed clients.

1. Crop disease images are collected and preprocessed.
2. The dataset is partitioned among multiple clients.
3. Each client trains its own YOLOv8 model locally.
4. Local model parameters are transmitted to the Flower server.
5. The server aggregates client updates using the FedAvg algorithm.
6. The updated global model is redistributed.
7. The process repeats over multiple communication rounds.
8. The final model is evaluated using standard object detection metrics.

---

# Dataset

The dataset is organized into multiple distributed clients to simulate a real Federated Learning environment.

Each client contains:

* Training data
* Validation data
* Testing data

Every subset includes:

* Images
* YOLO annotation files

This structure enables decentralized training while preserving data locality.

---

# Performance Evaluation

The project evaluates model performance using:

* Precision
* Recall
* mAP@50
* mAP@50-95
* Training Loss
* Validation Loss

These metrics provide a comprehensive assessment of the global model's detection capability.

---

# Repository Structure

```text
Agri Vision/
│
├── dataset/
│   ├── client1/
│   ├── client2/
│   └── client3/
│
├── server/
├── clients/
├── models/
├── runs/
├── requirements.txt
├── README.md
└── ...
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/HannanRehman/HannanRehman4.git
```

Move into the project directory:

```bash
cd HannanRehman4
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

Start the Flower server.

Launch each federated client.

Run the training process.

Monitor the aggregation rounds and evaluate the resulting global model.

> The exact execution commands may vary depending on your local project configuration.

---

# Future Improvements

The project is designed to be extensible. Potential future enhancements include:

* Real-time disease detection
* Mobile application integration
* Cloud deployment
* Edge AI support
* Additional crop and disease classes
* Explainable AI (XAI)
* Automated crop prediction module
* Performance optimization for large-scale deployments

---

# Team Members

* **Hannan Rehman**
* **Hasnain Ali**
* **Fuzail Talpur**

Bachelor of Software Engineering

Final Year Project

---

# Repository Status

🚧 **This project is currently under active development.**

Additional documentation, experimental results, screenshots, architecture diagrams, and performance analysis will be added as development progresses.

---

# Acknowledgements

We would like to express our sincere gratitude to our project supervisor, faculty members, and the open-source community whose tools and research have contributed significantly to the successful development of this project.

Special thanks to the developers and contributors of:

* Ultralytics YOLOv8
* Flower Federated Learning Framework
* PyTorch
* OpenCV

for providing powerful open-source technologies that made this research possible.

---

# License

This repository has been created for academic and research purposes as part of a Bachelor of Software Engineering Final Year Project.

---

## Star the Repository

If you find this project interesting or helpful, consider giving it a ⭐ on GitHub.
