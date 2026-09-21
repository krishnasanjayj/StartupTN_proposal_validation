# StartupTN AI Proposal Evaluation & Monitoring System

A domain-specific AI/ML decision support platform built for StartupTN (Government of Tamil Nadu) to evaluate startup proposals prior to approval and monitor financial/operational health post-approval.

---

## 🏛 System Architecture Overview

```
                          STARTUPTN AI SYSTEM
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
    Proposal LLM             Similarity AI             Financial AI
         |                         |                         |
   QLoRA / SFT                Embeddings                 Analytics
 (Qwen3-0.6B Base)          (pgvector DB)            Anomaly Detection
         |                         |                    Forecasting
         +-------------------------+-------------------------+
                                   |
                                   v
                              Risk Engine
                                   |
                                   v
                            Explainable AI
                                   |
                                   v
                           Official Dashboard
                                   |
                                   v
                      Human Official Approval
```

---

## 💻 Hardware & Execution Environment

- **Development Hardware**: Intel Core i5-12450H CPU, 16GB RAM, Integrated Graphics (No dedicated GPU)
- **OS**: Arch Linux
- **Python Environment**: Managed Virtualenv (`/home/sanjay/startup-ai/.venv`)
- **Base Model**: `Qwen/Qwen3-0.6B` (Empirically selected over Llama-3.2-1B, SmolLM2-360M, TinyLlama-1.1B, and Qwen2.5-0.5B; see full technical proof in [`docs/MODEL_SELECTION_AND_BENCHMARK_REPORT.md`](file:///home/sanjay/startup-ai/docs/MODEL_SELECTION_AND_BENCHMARK_REPORT.md))
- **Execution Strategy**:
  - Local CPU: Fast deterministic financial calculations, vector embeddings, risk scoring, FastAPI backend, and local Qwen3-0.6B evaluation inference (2.1s latency, 1.18 GB RAM).
  - Cloud GPU: SFT LoRA fine-tuning pipeline (`training/sft_trainer.py`).

### 🔬 Empirical Model Comparison Summary (Intel i5-12450H CPU)

| Model | Parameters | RAM Footprint | CPU Speed | Latency | JSON Validity | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3-0.6B** | **596M** | **1.18 GB** | **24.6 tok/s** | **2.14 s** | **98.4%** | **SELECTED (Winner)** |
| **Qwen2.5-0.5B** | 494M | 0.95 GB | 27.8 tok/s | 1.88 s | 89.5% | Lower schema adherence |
| **Llama-3.2-1B** | 1.23B | 2.85 GB | 9.2 tok/s | 6.18 s | 92.1% | Exceeds RAM & latency limit |
| **SmolLM2-360M** | 362M | 0.72 GB | 34.1 tok/s | 1.52 s | 56.0% | Catastrophic schema drops |
| **TinyLlama-1.1B**| 1.10B | 2.41 GB | 10.4 tok/s | 5.26 s | 61.2% | Markdown leaks & slow |

---

## 🚀 Quickstart & Verification

### 1. Verify Dataset Validation & Schema
```bash
/home/sanjay/startup-ai/.venv/bin/pytest tests/test_dataset_schema.py
```

### 2. Run LLM CPU Inference Test
```bash
/home/sanjay/startup-ai/.venv/bin/pytest tests/test_llm_inference.py
```

### 3. Dry-Run Fine-Tuning Pipeline
```bash
/home/sanjay/startup-ai/.venv/bin/python training/sft_trainer.py --dry_run
```

---

## 📂 Project Structure

```
startup-ai/
├── config/
│   └── settings.py               # Application configuration settings
├── data/
│   ├── raw/                      # Raw proposal uploads
│   └── processed/
│       ├── schemas/              # JSON schemas for evaluation output
│       └── startup_proposals_sample.jsonl # Validated instruction dataset
├── src/
│   ├── __init__.py
│   ├── llm/                      # Base Qwen loader, templates & proposal evaluator
│   ├── similarity/               # Sentence embeddings & duplicate search engine
│   ├── financial/                # Financial metric calculations & anomaly detection
│   ├── risk/                     # Multi-factor explainable risk engine
│   ├── database/                 # PostgreSQL + pgvector ORM models
│   └── backend/                  # FastAPI web services & API routers
├── training/
│   ├── dataset_schema.py         # Pydantic schema for training examples
│   ├── dataset_builder.py        # Data cleaning, anonymization & dataset splitting
│   ├── sft_trainer.py            # LoRA fine-tuning trainer (TRL, PEFT, Accelerate)
│   └── evaluation.py             # LLM output accuracy & structured evaluation
├── tests/
│   ├── test_llm_inference.py     # CPU inference sanity test
│   └── test_dataset_schema.py    # Dataset schema validation test
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```
