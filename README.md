# Auto-Analyst

**Full-stack automated EDA + baseline modeling tool for CSV datasets.**  
Upload a file, explore the data, run ML models, view insights, and generate reports — all from a clean React interface.

---
## ✨ Features

- **CSV Upload**
- **Automated EDA**
  - Column type inference
  - Missingness summary
  - Numeric/categorical profiling
  - Histograms, boxplots, barplots
  - Correlation heatmap
- **Data Cleaning & Transformation**
  - Drop specific columns
  - Filter rows based on conditions
  - Impute missing values (Mean, Median, Mode, Constant, or Drop rows)
- **Machine Learning Engine**
  - Auto task detection (classification / regression)
  - Preprocessing pipeline (impute → scale → one-hot)
  - **Hyperparameter Tuning** using Optuna
  - Configurable Train/Validation/Test splits
  - Candidate models:
    - Logistic / Linear Regression
    - Random Forest
    - Gradient Boosting
  - Best model selection + detailed metrics
  - Feature importances
  - **Interactive Playground** for real-time predictions
- **Insights & Reporting**
  - AI-generated narrative overview
  - Downloadable **HTML and PDF** reports
- **System**
  - Column type overrides
  - Clear server cache (Redis)
  - Docker support

---

## 🛠 Tech Stack

**Backend:** FastAPI, pandas, numpy, scikit-learn, Optuna, WeasyPrint, Redis, matplotlib, seaborn, Jinja2  
**Frontend:** React + TypeScript, Vite, Axios, Plotly  
**Deployment:** Docker, docker-compose, Nginx

---
## 📦 Run with Docker

```bash
docker compose build
docker compose up
```

Frontend: http://localhost:3000

Backend: http://localhost:8000

### To stop:

```bash
docker compose down
```

---

## 📦 Run Locally (without Docker)

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend: http://localhost:8000

**Frontend**

```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```

Frontend: http://localhost:5173

## Project Structure
```text
.
├── backend/
│   ├── app/
│   │   ├── eda/
│   │   │   ├── insights.py       # Generates summaries of data
│   │   │   ├── profiler.py       # Infers column types and calculates statistics
│   │   │   └── visualizer.py     # Creates plots 
│   │   ├── ml/
│   │   │   ├── evaluation.py     # Calculates metrics (Accuracy, F1, RMSE, R2)
│   │   │   ├── modeling.py       # Runs training for baseline models
│   │   │   ├── preprocessing.py  # Builds pipelines
│   │   │   ├── task_detection.py # Detects regression or classification
│   │   │   └── tuning.py         # Hyperparameter optimization using Optuna
│   │   ├── reporting/
│   │   │   ├── templates/
│   │   │   │   └── report.html   # Jinja2 template for the analysis report
│   │   │   ├── builder.py        # Renders the HTML report
│   │   │   └── pdf_export.py     # Converts HTML reports to PDF
│   │   ├── utils/
│   │   │   └── storage.py        # Manages Redis caching for datasets/images
│   │   ├── config.py             # App configuration
│   │   ├── main.py               # FastAPI entry point and API routes
│   │   └── schemas.py            # Pydantic models for API validation
│   ├── Dockerfile                # Python backend image definition
│   └── requirements.txt          # Dependencies
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.ts         # Axios instance configuration
│   │   ├── components/
│   │   │   └── NumericHist.tsx   # Reusable Plotly histogram component
│   │   ├── typings/              # TypeScript type definitions
│   │   ├── App.tsx               # Main dashboard UI logic and state
│   │   └── types.ts              # Shared TypeScript interfaces
│   ├── Dockerfile                # Node/Nginx frontend image definition
│   └── package.json              # Frontend dependencies
├── docker-compose.yml            # Orchestrates Backend, Frontend, and Redis
└── README.md                     # Project documentation
```

