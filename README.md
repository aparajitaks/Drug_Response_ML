# Drug Response Prediction using Machine Learning

This project predicts patient drug response categories using medical review data and metadata.
It follows an **industry-style ML pipeline** with modular notebooks, proper evaluation, and explainability using SHAP.

---

## 🚀 Project Overview

* **Objective:** Predict drug response category (Positive / Neutral / Negative)
* **Domain:** Healthcare, Medical NLP + ML
* **Approach:** Classical ML with strong preprocessing and interpretability

---

## 🧠 Tech Stack

* **Language:** Python
* **ML:** Scikit-learn (Logistic Regression)
* **Explainability:** SHAP
* **EDA & Visualization:** Pandas, Matplotlib, Seaborn
* **Environment:** Google Colab
* **Model Persistence:** Joblib

---

## 📁 Project Structure

```
Drug-Response-ML/
│
├── notebooks/
│   ├── 01_Data_Loading_and_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Building.ipynb
│   ├── 04_Evaluation.ipynb
│   └── 05_Explainability_SHAP.ipynb
│
├── data/
│   ├── raw/
│   │   ├── drugsComTrain_raw.csv
│   │   └── drugsComTest_raw.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── feature_data.csv
│
├── models/
│   ├── logreg_model.pkl
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── README.md
└── report.md
```

---

## ⚙️ ML Pipeline

### 1. Data Loading & EDA

* Missing value analysis
* Drug, condition & rating distributions

### 2. Feature Engineering

* One-hot encoding of categorical features
* Numerical feature scaling
* Train-test split

### 3. Model Building

* Logistic Regression with class balancing
* Pipeline-based preprocessing
* Model persistence using Joblib

### 4. Evaluation

* Accuracy, Precision, Recall, F1-score
* Confusion Matrix visualization

### 5. Explainability (SHAP)

* Feature importance analysis
* Global and local interpretability
* RAM-safe sampling for large feature space

---

## 📊 Key Learnings

* Handling high-cardinality categorical data
* Building reproducible ML pipelines
* Model evaluation beyond accuracy
* Explainable AI for healthcare applications

---

## 📌 Future Improvements

* Try tree-based models (XGBoost, LightGBM)
* Add NLP embeddings from review text
* Deploy as a web app using FastAPI

---

## 👤 Author

**Aparajita K Singh**
BTech CSE (AI & ML)
Newton School of Technology
