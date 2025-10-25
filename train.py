import pandas as pd, numpy as np, matplotlib.pyplot as plt, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

base = Path(__file__).parent
data_path = base / "data" / "Dynamic_2017_2_0.csv"

df = pd.read_csv(data_path)

# ✅ Change this to YOUR exact column name
percent_col = "Stage of Ground Water Extraction (%)"

# ✅ Create category label from percentage values
def classify(x):
    if x < 70:
        return "safe"
    elif 70 <= x < 90:
        return "semi-critical"
    elif 90 <= x < 100:
        return "critical"
    else:
        return "over-exploited"

df["stage_label"] = df[percent_col].apply(classify)
label_col = "stage_label"

# Features = everything numeric except the label column
X = df.select_dtypes(include=[np.number]).drop(columns=[percent_col], errors="ignore")
y = df[label_col]

# Categorical fallback (if needed)
cat = [c for c in df.columns if df[c].dtype == 'object' and c != label_col]
num = X.columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)
], remainder="drop")

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
rfp = Pipeline([("pre", pre), ("clf", rf)])

Xtr, Xte, ytr, yte = train_test_split(df[num+cat], y, test_size=0.2, random_state=42, stratify=y)
rfp.fit(Xtr, ytr)

rpt = classification_report(yte, rfp.predict(Xte), output_dict=True)
joblib.dump(rfp, base / "groundwater_stage_model.pkl")
with open(base / "model_report.json", "w") as f: json.dump(rpt, f, indent=2)

print("✅ Model saved to groundwater_stage_model.pkl")
print("✅ Report saved to model_report.json")
