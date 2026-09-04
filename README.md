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
- **Base Model**: `Qwen/Qwen3-0.6B`
- **Execution Strategy**:
  - Local CPU: Fast deterministic financial calculations, vector embeddings, risk scoring, FastAPI backend, and local Qwen3-0.6B evaluation inference.
  - Cloud GPU: SFT LoRA fine-tuning pipeline (`training/sft_trainer.py`).

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
