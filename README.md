# 🧠 Drug Response ML Dashboard

🚀 A complete **Machine Learning + Streamlit Dashboard** that predicts **drug response categories** using a trained ML model and provides interactive analytics & downloadable prediction results.

This project allows users to upload a dataset CSV file and instantly get predictions along with confidence scores, charts, and insights.

---

## 📌 Project Overview

The **Drug Response ML Dashboard** is an end-to-end ML project built with:

* **Scikit-Learn model**
* **Streamlit UI**
* **Plotly analytics charts**
* **CSV upload + downloadable output**

The goal is to predict whether a patient is a **Responder / Non-Responder** (or response category label) based on drug review-related features.

---

## 🚀 Live Demo

🔗 **Hosted App:**
👉 [https://drugresponseml.streamlit.app](https://drugresponseml.streamlit.app)

---

## 🎯 Features

-> Upload CSV dataset from sidebar
-> Displays dataset preview
-> Automatically selects required features using schema file
-> Predicts drug response category
-> Generates confidence scores (if model supports `predict_proba`)
-> Interactive analytics visualizations (Plotly)
-> Shows most confident predictions
-> Download prediction results as CSV

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Scikit-Learn**
* **Pandas**
* **Joblib**
* **Plotly**
* **JSON schema config**

---

## 📂 Project Structure

```
Drug_Response_ML/
│
├── ml-service/
│   ├── dashboard.py                # Streamlit dashboard
│   ├── app.py                      # Backend / ML service script (if used)
│   ├── requirements.txt            # Required dependencies
│   ├── feature_schema.json         # Feature schema file
│   ├── label_mapping.json          # Mapping from numeric labels to text labels
│   ├── models/
│   │   └── drug_response_model.pkl # Trained ML model
│   └── .gitignore
│
└── README.md
```

---

## 📊 Input CSV Format

The uploaded CSV must contain the following required columns (as defined in `feature_schema.json`):

Example schema:

```json
{
  "features": ["drugName", "condition", "rating", "usefulCount"]
}
```

So your CSV should include:

| Column Name | Description           |
| ----------- | --------------------- |
| drugName    | Name of the drug      |
| condition   | Medical condition     |
| rating      | Patient rating        |
| usefulCount | Count of useful votes |

---

## 🧠 Model Output

After prediction, the dashboard generates:

* `Prediction` (numeric label)
* `Confidence` (max probability score)
* `Prediction_Label` (human readable label)

Example output:

| Prediction | Confidence | Prediction_Label |
| ---------- | ---------- | ---------------- |
| 2          | 0.91       | Responder        |

---

## 📈 Analytics Provided

The dashboard also shows:

-> Prediction Distribution Chart
-> Confidence Score Distribution Histogram
-> Top 10 Most Confident Predictions Table

---

## ⚙️ How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/aparajitaks/Drug_Response_ML.git
cd Drug_Response_ML/ml-service
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```bash
streamlit run dashboard.py
```

Now open in browser:

👉 [http://localhost:8501](http://localhost:8501)

---

## 🌐 Deployment (Streamlit Cloud)

This project is deployed using **Streamlit Cloud**.

To deploy:

1. Push code to GitHub
2. Connect repo on Streamlit Cloud
3. Select file path:

   ```
   ml-service/dashboard.py
   ```
4. Add requirements.txt automatically

---

## 📌 Note About Dataset

The dataset used for training/testing is large, so it is not uploaded directly to GitHub.

Users can upload their own CSV file in the required format.

---

## 🔥 Future Improvements

🚀 Add model retraining option
🚀 Add SHAP explainability visualizations
🚀 Improve preprocessing pipeline for text review analysis
🚀 Add feature importance chart
🚀 Add support for multiple ML models

---

## 👩‍💻 Author

**Aparajita K. Singh**
BTech CSE (AI & ML) | Newton School of Technology

🔗 GitHub: [https://github.com/aparajitaks](https://github.com/aparajitaks)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it motivates me a lot! 🚀🔥

---

