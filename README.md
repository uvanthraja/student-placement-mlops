# Student Placement Prediction System

## Project Overview

An end-to-end Student Placement Prediction System developed with
Machine Learning and MLOps practices.

The system predicts whether a student is likely to be placed based
on relevant pre-placement student information.

## Machine Learning

The final leakage-free ML pipeline includes:

- Data preprocessing
- Missing value handling
- Numerical feature scaling
- Categorical feature encoding
- Model comparison
- Hyperparameter tuning
- Model evaluation
- Model persistence

## Target Variable

Placement Status

## Leakage Prevention

The following post-placement or identity-related columns are excluded
from model training:

- Student ID
- Name
- Placement Domain
- CTC (LPA)

## Project Structure

```text
MLops project/
│
├── data/
│   └── raw/
│
├── notebooks/
│
├── models/
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── inference/
│
├── src/
│   ├── data/
│   ├── models/
│   └── validation/
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Roadmap

- Phase 1: Machine learning pipeline (data prep, training, evaluation)
- Phase 1.5: Model inference and validation
- Phase 2: MLOps (Git/GitHub, DVC, MLflow, CI/CD, Docker, FastAPI,
  Streamlit, monitoring with Evidently)
