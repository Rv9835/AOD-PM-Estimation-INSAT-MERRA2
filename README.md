# Multi-City PM2.5 Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-Multi--Model-green.svg)](#modeling-overview)

Enterprise-style monorepo for **multi-city air pollution forecasting** across India.  
The platform combines data pipelines, model training/evaluation, API services, Streamlit analytics UI, and a React frontend workspace.

---

## Architecture

```text
.
├── backend/
│   ├── src/
│   ├── scripts/
│   ├── configs/
│   ├── tests/
│   ├── artifacts/
│   └── logs/
├── frontend/
│   ├── app_dashboard.py
│   ├── app/
│   ├── components/
│   └── ... (Next.js/shadcn files)
├── data/
│   ├── openaq_location_6978_measurments.csv
│   └── processed/
├── docs/
│   ├── DASHBOARD_README.md
│   ├── LOCALHOST.md
│   ├── QUICKSTART_DASHBOARD.md
│   ├── QUICK_START_REAL_DATA.md
│   ├── REAL_DATA_INTEGRATION_GUIDE.md
│   └── SYSTEM_OVERVIEW.md
├── run_dashboard.sh
├── run_project.sh
├── pyproject.toml
└── .gitignore
```

---

## Quick Start

### 1) Install dependencies

#### Backend (ML + API + pipelines)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Frontend (React/shadcn workspace)
```bash
cd frontend
npm install
cd ..
```

### 2) Train the multi-city models

```bash
python backend/scripts/train_multi_city_models.py --config backend/configs/base.yaml
```

### 3) Evaluate all scenarios

```bash
python backend/scripts/evaluate_all_scenarios.py --config backend/configs/base.yaml
```

### 4) Launch the Streamlit dashboard

```bash
streamlit run frontend/app_dashboard.py
```

Or use helper script:

```bash
./run_project.sh
```

---

## Modeling Overview

The backend supports cross-city robustness analysis with multiple algorithms:

- Linear Regression
- Random Forest
- XGBoost
- LightGBM
- Neural Network

Training and evaluation outputs are stored under `backend/artifacts/`.

---

## Documentation

Detailed guides are now centralized in `docs/`:

- [Dashboard Guide](docs/DASHBOARD_README.md)
- [Localhost Setup](docs/LOCALHOST.md)
- [Dashboard Quickstart](docs/QUICKSTART_DASHBOARD.md)
- [Real Data Quickstart](docs/QUICK_START_REAL_DATA.md)
- [Real Data Integration](docs/REAL_DATA_INTEGRATION_GUIDE.md)
- [System Overview](docs/SYSTEM_OVERVIEW.md)

---

## Notes

- Backend configs are under `backend/configs/`.
- Dataset files are under `data/`.
- Streamlit and backend entrypoints resolve paths dynamically using `pathlib` to reduce breakage after restructuring.
