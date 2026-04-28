# Arxiv CS Research Paper Classifier

## 📌 Overview
The **Arxiv CS Research Paper Classifier** is an end-to-end machine learning system designed to classify arXiv Computer Science research papers into relevant categories.

This project focuses on **building a production-grade ML system**, not just a model. It integrates:
- Model Serving (MLflow)
- Backend API (FastAPI)
- Frontend UI (Streamlit)
- Pipeline Management (DVC)
- Monitoring (Prometheus + Grafana)
- Containerization (Docker)
- Orchestration & Data Ingestion (Airflow)

---

## 🎯 Motivation
Modern ML systems require more than model accuracy. They need:
- Reproducibility
- Observability
- Scalability
- Clean deployment pipelines

This project demonstrates a **complete ML lifecycle system**, making it ideal for learning, showcasing, and real-world applications.

---

## 🏗️ System Architecture
User → Streamlit (Docker) → FastAPI (Docker) → MLflow (Host)

Monitoring:
- Streamlit/FastAPI expose metrics
- Prometheus scrapes metrics
- Grafana visualizes dashboards

---

## ⚙️ Requirements

- Python **3.10+**
- Docker + Docker Compose
- pip / virtual environment recommended

---

## 🐍 Python Setup

```bash
python3.10 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

---

## 📁 Project Structure

```
project/
│
├── airflow/                  # Orchestration (monthly retraining)
│   ├── dags/
│   │   └── pipeline_dag.py
│   └── tasks/
│       ├── ingest.py
│       └── clean.py
│
├── backend/                  # FastAPI inference service
│   ├── main.py
│   ├── inference.py
│   ├── schema.py
│   ├── utils.py
│   └── config.yaml
│
├── frontend/                 # Streamlit UI
│   ├── app.py
│   ├── components/
│   ├── metrics.py
│   └── config.yaml
│
├── pipelines/                # DVC pipeline stages
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── data/                     # Dataset (raw + processed)
│   ├── raw/
│   └── processed/
│
├── artifacts/                # Model artifacts (DVC tracked)
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── metrics.json
│
├── monitoring/               # Prometheus + Alertmanager setup
│   ├── prometheus.yml
│   ├── alertmanager.yml
│   └── grafana/
│
├── docker/                   # Dockerfiles
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
│
├── tests/                    # Unit & API tests
│   ├── test_api.py
│   ├── test_inference.py
│   └── test_utils.py
│
├── configs/                  # Configs & parameters
│   └── params.yaml
│
├── notebooks/                # EDA / experiments
│   └── Exploratory_Data_Analysis.ipynb
│
├── logs/                     # Runtime logs (ignored in Git)
│
├── dvc.yaml
├── dvc.lock
├── docker-compose.yml
├── run.py                    # System orchestrator
├── requirements.txt
└── README.md
```

---

## 🚀 Running the System

### Start (without monitoring)
```bash
python run.py start
```

### Start (with monitoring)
```bash
python run.py start --monitoring
```

### Stop all services
```bash
python run.py stop
```

---

## 📊 Monitoring Setup

- Streamlit exposes metrics using `start_http_server(8001)`
- FastAPI exposes `/metrics`
- Prometheus runs locally and scrapes:
  - `localhost:8001` (Streamlit)
  - `localhost:8000` (Backend)

### Grafana
- Runs inside Docker
- Access: http://localhost:3000
- Default credentials:
  - user: admin
  - password: admin

### Add Prometheus Data Source
```
http://host.docker.internal:9090
```

### Grafana Dashboard
- The dashboard can be found at monitoring/grafana/grafana.json

---

## 🧪 Testing

Run all tests:

```bash
export PYTHONPATH=.
```

```bash
pytest -v
```

You can also target specific tests:

```bash
pytest tests/test_api.py
```

---

## 📦 DVC Pipeline

This project uses DVC for reproducible pipelines:

Stages:
- Preprocessing
- Feature Engineering
- Training
- Evaluation

### Run pipeline:
```bash
dvc repro
```

---

## 🔄 Workflow Orchestration (Airflow)

To enable **automated retraining and continuous data updates**, this project integrates **Apache Airflow (Standalone Mode)**.

### 🎯 Purpose

Airflow is responsible for:
- Monthly data ingestion and updates (append-only)
- Triggering the DVC pipeline
- Retraining the model on updated data
- Maintaining model freshness over time

---

### 🧠 Design

The training workflow follows:

Monthly Data Append → DVC Pipeline → Retraining → Evaluation → Updated Model

- New data is appended to the existing dataset  
- DVC ensures reproducible pipeline execution  
- MLflow tracks training experiments  
- The latest model is automatically available for serving  

---

### ⚙️ Execution Mode

This project uses **Airflow Standalone (local mode)** — not Docker.

For setup and installation, refer to:  
https://airflow.apache.org/docs/apache-airflow/stable/start.html

Access the Airflow UI at:
http://localhost:8080

---

### 📁 DAG Location

airflow/dag/pipeline_dag.py

---

### 🔁 Scheduling

The pipeline runs **monthly on the 1st day**:

schedule_interval="0 0 1 * *"

---

### 🧪 Manual Trigger

airflow dags trigger arxiv_monthly_pipeline

---

### ⚠️ Notes

- DAG name: `arxiv_monthly_pipeline`
- Data ingestion is **append-only (no overwrites)**
- Retraining uses the full accumulated dataset
- Airflow operates independently from inference (`run.py`)
- Designed for lightweight local orchestration

---

## 🔑 Key Notes

- MLflow runs **outside Docker**
- Backend accesses MLflow via:
  ```
  host.docker.internal:5001
  ```
- Prometheus runs locally (not in Docker)
- Grafana runs inside Docker

---

## 🏁 Conclusion

This project demonstrates a **clean, modular, and production-ready ML system design** integrating:
- ML pipelines
- Model serving
- Monitoring
- Containerization

It is designed to be:
- extensible
- reproducible
- interview-ready

## 👤 Author

**Jayaharish R**
- EP21B016  
- IIT Madras(Engineering Physics + Data Science)  
- GitHub: https://github.com/jr1010