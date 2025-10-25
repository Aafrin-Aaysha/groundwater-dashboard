# Groundwater Stage Prediction — Project Report

## 1️⃣ Objective
Develop a machine learning–based dashboard to:
- Classify groundwater extraction stages.
- Visualize spatial patterns by state/district.
- Support decision-making for sustainable water management.

## 2️⃣ Dataset
- Source: Central Ground Water Board (India)
- Format: District-wise data with extraction %, recharge, and use details.

## 3️⃣ Model
- Training via `train.py`
- Model: Random Forest / Logistic Regression (best chosen automatically)
- Output: `groundwater_stage_model.pkl`

## 4️⃣ Dashboard
- Built with Streamlit + Plotly
- Four Tabs:
  1. **Home** — KPIs + Donut chart  
  2. **Insights** — State/District analytics  
  3. **What-if Prediction** — Interactive prediction  
  4. **Recommendations** — Policy guidance

## 5️⃣ Deployment
- Hosted via Streamlit Cloud  
- Public demo URL (example):  
  `https://<your-username>-groundwater-dashboard.streamlit.app`

---

## 6️⃣ Future Enhancements
- Add map view (geojson overlay)
- Add multi-year temporal trends
- Integrate rainfall / recharge forecasts
- Allow CSV uploads for batch predictions
