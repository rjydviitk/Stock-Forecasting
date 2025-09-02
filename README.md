# 📈 Stock Price Forecasting (SARIMA • Prophet • LSTM • Transformer)

An end-to-end, industry-ready demo for forecasting stock prices with multiple models and an interactive Streamlit dashboard.

## Features
- SARIMA, Prophet, LSTM, Transformer
- Comparison mode + metrics (RMSE, MAPE, R²)
- Download metrics/forecasts (CSV/Excel)
- PDF report generation
- Local CSV fallback if Yahoo Finance isn't available

## Run
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```


## 🧪 CI
GitHub Actions workflow runs linting and tests on each push/PR. See `.github/workflows/ci.yml`.

## 🐳 Docker
Build and run the dashboard in a container:
```bash
docker build -t stock-forecasting:latest .
docker run --rm -p 8501:8501 stock-forecasting:latest
```
Open http://localhost:8501

## 🛠️ Makefile
Common tasks:
```bash
make install   # install deps + dev tools
make run       # run Streamlit
make lint      # flake8
make test      # pytest
make docker-build
make docker-run
```

## 📸 Screenshots (add yours)
- Historical chart
- Forecast comparison chart
- Metrics table
- PDF report

## 🔖 Badges
Add these once your repo is on GitHub (replace `USER/REPO`):
```
![CI](https://github.com/USER/REPO/actions/workflows/ci.yml/badge.svg)
```
