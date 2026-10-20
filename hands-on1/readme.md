M
🚀 MLOps, LLMOps, & AIOps Hands-on Journey

Welcome to my comprehensive, step-by-step implementation repository. This project tracks my hands-on journey through the entire lifecycle of traditional machine learning (MLOps), large language models (LLMOps), and artificial intelligence for IT operations (AIOps).
I am building this sequentially, starting from core ML model training and scaling up to automated production pipelines.

🗺️ Roadmap & Architecture

The repository is divided into three major phases. As I progress, each section will expand with production-grade workflows.

🏗️ Phase 1: MLOps (Current Focus)

Building the foundational automation pipeline for traditional Machine Learning.
• Model Training: Experimentation tracker, data versioning, and feature engineering.
• CI/CD Pipelines: Automated testing for data, code, and model artifacts.
• Deployment: Containerized microservices using Docker and Kubernetes.
• Monitoring: Drift detection and automated model retraining triggers.

🤖 Phase 2: LLMOps (Upcoming)

Scaling operational practices to generative AI and LLMs.
• Prompt Engineering: Versioning and tracking optimal prompt templates.
• Fine-Tuning: Efficient fine-tuning loops (LoRA/QLoRA) with resource tracking.
• Vector DBs & RAG: Implementing retrieval-augmented generation pipelines.
• Evaluation & Guardrails: Tracking hallucination metrics, latency, and token costs.

🧠 Phase 3: AIOps (Upcoming)

Applying AI to optimize system operations and infrastructure.
• Log Anomaly Detection: Real-time processing of system logs to flag failures.
• Predictive Scaling: Forecasting infrastructure load to scale resources.
• Root Cause Analysis (RCA): Automated telemetry tracking to isolate system bugs.

🛠️ Tech Stack

• Languages: Python
• Machine Learning: Scikit-Learn, PyTorch
• Pipeline & Tracking: [e.g., MLflow, DVC, Prefect — update as you choose]
• DevOps & Deployment: Docker, GitHub Actions

📈 Current Progress Tracker

• Setup repository structure and developer environment
• [/] Phase 1: Baseline Machine Learning Model Training (In Progress)
• Phase 1: Packaging model with Docker
• Phase 2: Core LLM pipeline implementation
• Phase 3: Telemetry and log collection setup

🏃 How to Run the Current Iteration


Prerequisites

Make sure you have Python 3.10+ and virtualenv installed.

Setup

1. Clone the repository:bash
git clone https://github.com/Deepak-dey/mlops-handson
Use code with caution.
2. Create and activate a virtual environment:bash
python -bin/python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Use code with caution.
3. Install dependencies:bash
pip install -r requirements.txt
Use code with caution.
4. Run the baseline training script:bash
python src/mlops/train.py
Use code with caution.
