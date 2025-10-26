# 🌍 Groundwater Stage Prediction Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://groundwater-dashboard-model.streamlit.app/)

A sleek, interactive **Streamlit dashboard** for analyzing and predicting groundwater extraction stages — Safe, Semi-Critical, Critical, or Over-Exploited.

---

## 🧠 Overview

This dashboard:
- Analyzes groundwater extraction patterns by **State** and **District**
- Displays category distributions with **donut and bar charts**
- Allows **What-if prediction** using trained ML models
- Provides clear **recommendations** for each risk stage

---

## 🗂️ Repository Structure

groundwater-dashboard/
├── app.py # Streamlit dashboard (main entry)
├── train.py # Model training script
├── requirements.txt # Dependencies
├── data/ # Contains dataset CSV
├── models/ # Trained ML model (.pkl)
└── docs/ # Documentation & reports


---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/<your-username>/groundwater-dashboard.git
cd groundwater-dashboard
pip install -r requirements.txt
streamlit run app.py
