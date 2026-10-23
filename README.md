# 🏁 F1 Analytics LLM-RAG + Knowledge Graph + Multi-Model Engine

An enterprise-grade, privacy-first local AI analytics platform built specifically for Formula 1 data. The application combines **Hybrid Vector & Graph RAG**, an **Automated Task-Aware Multi-Model Machine Learning Engine** (comprising 6 parallel models), and a **local LLM runner via Ollama** (Llama 3), exposed through a FastAPI backend and a Streamlit chat dashboard.

---

## 📂 Project Directory Structure

```text
F-Race/
├── data/
│   ├── dim_constructors/
│   │   └── dim_constructors.parquet
│   ├── dim_drivers/
│   │   └── dim_drivers.parquet
│   ├── dim_races/
│   │   └── dim_races.parquet
│   ├── fact_session_results/
│   │   └── fact_session_results.parquet
│   └── ref_nationality_region/
│       └── ref_nationality_region.parquet
│
└── Graphic-LlmRag/
    ├── api/
    │   ├── __init__.py
    │   └── app.py
    ├── LLM_Stremlit/
    │   ├── __init__.py
    │   └── app.py
    ├── graph_store/
    │   ├── __init__.py
    │   └── neo4j_connector.py
    ├── ml_engine/
    │   ├── __init__.py
    │   ├── autolabel.py
    │   ├── evaluator.py
    │   ├── registry/
    │   └── trainer.py
    ├── rag_engine/
    │   ├── __init__.py
    │   ├── embedder.py
    │   ├── retriever.py
    │   └── vector_store.py
    ├── README.md
    ├── requirements.txt
    ├── scripts/
    │   └── run_pipeline.sh
    └── tree.txt
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. Python Environment
Ensure Python **3.10+** is installed on your system.

### 2. Install Dependencies
Navigate to the `Graphic-LlmRag/` directory and install the required Python packages:

```bash
cd Graphic-LlmRag
pip install -r requirements.txt
```

### 3. Local LLM Runner (Ollama)
Ensure **Ollama** is installed and running locally with the `llama3` model pulled:

```bash
# Pull the Llama 3 model
ollama pull llama3

# Start the local Ollama server
ollama serve
```

---

## 🚀 Procedure to Execute the Application

The application operates on a decoupled architecture comprising a **FastAPI backend** and a **Streamlit frontend interface**.

### Step 1: Launch the FastAPI Backend
Open a terminal tab, navigate into the `Graphic-LlmRag/` directory, and start the API server:

```bash
cd Graphic-LlmRag
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

> 🌐 Once running, access the interactive Swagger API documentation at:  
> [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Step 2: Launch the Streamlit Frontend Dashboard
Open a second terminal tab, navigate to the `Graphic-LlmRag/` directory, and start the Streamlit application:

```bash
cd Graphic-LlmRag
streamlit run LLM_Stremlit/app.py
```

> 📊 This will open the ChatGPT-style F1 analytics dashboard automatically in your default browser at:  
> [http://localhost:8501](http://localhost:8501)