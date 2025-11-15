# Auto-Analyst

**Full-stack automated EDA + baseline modeling tool for CSV datasets.**  
Upload a file, explore the data, run ML models, view insights, and generate reports — all from a clean React interface.

---

## ✨ Features

- **CSV upload**
- **Automated EDA**
  - Column type inference
  - Missingness summary
  - Numeric/categorical profiling
  - Histograms, boxplots, barplots
  - Correlation heatmap
- **Baseline Modeling**
  - Auto task detection (classification / regression)
  - Preprocessing pipeline (impute → scale → one-hot)
  - Candidate models:
    - Logistic / Linear Regression
    - Random Forest
    - Gradient Boosting
  - Best model selection + metrics
  - Feature importances
- **Insights generator** (narrative overview)
- **Downloadable HTML report**
- **Column type overrides**
- **Clear server cache** button
- **Docker support**

---

## 🛠 Tech Stack

**Backend:** FastAPI, pandas, numpy, scikit-learn, matplotlib, seaborn, Jinja2  
**Frontend:** React + TypeScript, Vite, Axios  
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
backend/
  app/
    eda/
    ml/
    utils/
    reporting/
    main.py
frontend/
  src/
docker-compose.yml
```

