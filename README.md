# 🌾 **AgriPredict — End-to-End Crop Yield Prediction & Optimization System**

AgriPredict is a full-stack machine learning application built with **FastAPI**, **Python (ML stack)**, and a custom **HTML/CSS/JS frontend**.
It enables farmers, agronomists, and analysts to:

✔ Upload and explore agricultural datasets
✔ Perform automated EDA with visualizations
✔ Preprocess and clean data
✔ Train a machine-learning model
✔ Predict crop yield for new inputs
✔ Optimize farming inputs (fertilizer, irrigation, pesticide)
✔ Interpret model decisions using SHAP explainability

All components work together to deliver a complete ML pipeline.

---

# 📁 **Project Structure**

```
PA_Agriculture/
│
├── app/
│   ├── main.py
│   ├── routers/
│   │     ├── upload.py
│   │     ├── eda.py
│   │     ├── preprocess.py
│   │     ├── model_train.py
│   │     ├── predict.py
│   │     ├── optimize.py
│   │     └── shap.py
│   │
│   ├── services/
│   │     ├── data_loader.py
│   │     ├── eda_service.py
│   │     ├── preprocess_service.py
│   │     ├── model_service.py
│   │     ├── prediction_service.py
│   │     ├── optimization_service.py
│   │     └── shap_service.py
│   │
│   ├── schemas/
│   │     └── prediction_schema.py
│   │
│   ├── models/
│   │     ├── model.pkl
│   │     ├── scaler.pkl
│   │     ├── encoder.pkl
│   │     └── feature_names.pkl
│
├── data/
│   ├── uploads/
│   └── processed/
│
└── frontend/
    ├── index.html
    ├── upload.html
    ├── eda.html
    ├── preprocess.html
    ├── train.html
    ├── predict.html
    ├── optimize.html
    ├── shap.html
    ├── styles.css
    └── script.js
```

---

# 🚀 **System Architecture & Flow**

AgriPredict follows a clean, modular pipeline:

### **1. Data Upload**

Users upload CSV files → stored in `data/uploads/`.

### **2. Automated EDA**

Backend generates:

* Dataset summary (shape, missing values, datatypes)
* Correlation heatmap
* Outlier analysis
* Yield distribution
* Boxplots
* Pairplots

Plots are returned as **base64 images** and displayed in the UI.

### **3. Preprocessing**

* Missing value imputation (mean, median, mode)
* Date → (year, month, day_of_year)
* One-hot encoding of categorical variables
* StandardScaler applied to numeric features
* Saves processed file in `data/processed/`

### **4. Model Training**

A RandomForestRegressor is trained with:

* One-hot encoded features
* Scaled values
* Train/test split
* Saved artifacts:

  * `model.pkl`
  * `encoder.pkl`
  * `scaler.pkl`
  * `feature_names.pkl`

### **5. Predictions**

User enters input values manually.
Backend:

* Validates using Pydantic
* Encodes categorical features
* Scales inputs
* Reorders features to match training
* Predicts yield

### **6. Optimization Engine**

Given soil + climate conditions, the system finds the **best combination** of:

* Fertilizer (kg/ha)
* Irrigation (mm)
* Pesticide (ml)

to **maximize predicted yield** while respecting:

✔ Cost ≤ ₹12,000
✔ Environmental score < 10,000

### **7. SHAP Explainability Dashboard**

Shows:

* SHAP summary plot (beeswarm)
* SHAP bar plot (global feature importance)

This explains *why* the model makes predictions.

---

# 🎯 **SHAP Explainability (Deep Explanation)**

## ⭐ What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theory based method that tells you:

> **How much each feature contributes to the model’s prediction.**

It assigns a SHAP value to every feature:

* Positive → increases predicted yield
* Negative → decreases predicted yield
* Magnitude → importance

SHAP is crucial because:

* ML models like RandomForest are **black boxes**
* Farmers need to know *why* the model recommends certain values
* Helps identify which inputs matter most

---

## ⭐ How SHAP Works in This Project

Your SHAP workflow:

### **1. Load the latest processed dataset**

Drop the target column (`yield_kg_per_ha`).

### **2. Apply the same transformations**

* One-hot encoding
* StandardScaler
* Feature reordering

This ensures the SHAP explainer sees the *exact same feature matrix* the model was trained on.

### **3. Compute SHAP values**

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_scaled)
```

### **4. Generate two plots**

#### ✔ SHAP Summary Plot (Beeswarm)

Shows:

* Feature influence
* How each feature pushes predictions up/down
* Full distribution across the dataset

#### ✔ SHAP Bar Plot

Shows:

* Ranked global feature importance

These two plots provide the most interpretable view of model behavior, so we intentionally removed the dependence plot (too fragile with encoded/scaled data).

---

# 🌱 **Optimization Engine (Detailed Explanation)**

The optimization feature answers:

> **“What is the best fertilizer–irrigation–pesticide combination to maximize yield?”**

But with constraints:

* Cost ≤ ₹12,000
* Environmental Score < 10,000

To perform this optimization, we need cost and environmental equations.

---

# 🔢 **How We Derived the Equations (THIS is crucial)**

Your dataset included these columns:

* `fertilizer_kg_per_ha`
* `irrigation_mm`
* `pesticide_ml`
* `input_cost_total`
* `environmental_score`

We analyzed the relationship between:

```
input_cost_total  vs  (fertilizer, irrigation, pesticide)
environmental_score vs  (fertilizer, irrigation, pesticide)
```

We discovered these were **linear relationships**, so we fitted **two linear regression models**:

---

## ⭐ Cost Equation

We fitted the regression:

```
input_cost_total = a1*fert + a2*irr + a3*pest + bias
```

The coefficients obtained were:

| Variable   | Coefficient |
| ---------- | ----------- |
| Fertilizer | 0.50165     |
| Irrigation | 0.20026     |
| Pesticide  | 0.14837     |
| Bias       | 20.19       |

So the final cost function becomes:

```python
def compute_cost(f, i, p):
    return 0.50165*f + 0.20026*i + 0.14837*p + 20.19
```

---

## ⭐ Environmental Score Equation

Regression:

```
environmental_score = b1*fert + b2*irr + b3*pest + bias
```

Coefficients:

| Variable   | Coefficient |
| ---------- | ----------- |
| Fertilizer | 0.60055     |
| Irrigation | 0.30009     |
| Pesticide  | 0.04946     |
| Bias       | 6.73        |

Final formula:

```python
def compute_env(f, i, p):
    return 0.60055*f + 0.30009*i + 0.04946*p + 6.73
```

These formulas are **data-driven**, not guesses.

---

# ♻️ **Optimization Process**

We search over all combinations:

### Fertilizer

`0 → 400 (step 10)`

### Irrigation

`0 → 400 (step 10)`

### Pesticide

`0 → 300 (step 10)`

For each combination:

1. Compute cost
2. Compute environment score
3. Check constraints
4. Apply encoder + scaler
5. Predict yield from ML model
6. Keep the best yield

This is a **constrained brute-force optimization**, effective because search space is small.

---

# 🎨 **Frontend**

The UI is fully custom (no frameworks):

* Clean sidebar navigation
* Modern CSS styling
* Interactive buttons
* Dynamic plot loading
* Real-time updates from the API

Pages:

* `Index` – Overview
* `Upload`
* `EDA`
* `Preprocess`
* `Train Model`
* `Predict Yield`
* `Optimize Inputs`
* `SHAP Visualization`

All front-end logic lives in `script.js`.

---

# 🧪 **Running the App**

### **Start Backend**

```bash
uvicorn app.main:app --reload
```

### **Open Frontend**

Open:

```
frontend/index.html
```

(Frontend uses fetch() to call FastAPI directly.)

---

# 🌟 **Technologies Used**

### **Backend**

* FastAPI
* Pydantic
* Scikit-learn
* Pandas
* Joblib
* Matplotlib / Seaborn
* SHAP

### **Frontend**

* HTML5
* CSS3
* Vanilla JavaScript
* Fetch API

---

