
---

### 🧩 File 3 — `docs/FEATURES.md`
```markdown
# 📘 Features Description

| Feature | Type | Description |
|----------|------|-------------|
| Name of State | Categorical | State name (used for grouping and prediction) |
| Name of District | Categorical | District name (local-level analysis) |
| Stage of Ground Water Extraction (%) | Numeric | Percentage of extraction relative to availability |
| Net Annual Ground Water Availability | Numeric | Annual available groundwater resource |
| Annual Ground Water Draft | Numeric | Annual withdrawal or use |
| Ground Water Recharge | Numeric | Natural recharge rate |
| Irrigation Use | Numeric | Portion used for agriculture |
| Industrial Use | Numeric | Portion used for industry |
| Domestic Use | Numeric | Portion used for households |
| ... | ... | Other supporting metrics depending on dataset |

---

**Derived Target Label:**
- If no `stage_label` exists, it is computed using:
  - `<70%` → Safe  
  - `70–90%` → Semi-Critical  
  - `90–100%` → Critical  
  - `>100%` → Over-Exploited
