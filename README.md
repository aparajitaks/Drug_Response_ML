# Drug Response ML Dashboard

A complete Machine Learning + Streamlit Dashboard that predicts drug response categories using a trained ML model and provides interactive analytics and downloadable prediction results.

This project allows users to upload a dataset CSV file and instantly get predictions along with confidence scores, charts, and insights.

---

## Project Overview

The Drug Response ML Dashboard is an end-to-end ML project built using:

- Scikit-Learn model
- Streamlit dashboard
- Plotly charts for analytics
- CSV upload and downloadable output
- JSON-based feature schema and label mapping

The goal is to predict whether a patient is a Responder / Non-Responder (or response category label) based on dataset features.

---

## Live Demo

Hosted App Link:  
https://drugresponseml.streamlit.app

---

## Features

- Upload CSV dataset from sidebar
- Dataset preview display
- Automatic feature selection using schema file
- Drug response category prediction
- Confidence scores (if model supports predict_proba)
- Interactive analytics visualizations (Plotly)
- Top most confident predictions table
- Download prediction results as CSV
- Demo dataset support (Download Sample CSV / Use Demo Dataset)

---

## Tech Stack

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Plotly
- FastAPI (optional backend support)

---

## Project Structure

```bash
Drug_Response_ML/
│
├── ml-service/
│   ├── dashboard.py
│   ├── app.py
│   ├── requirements.txt
│   ├── feature_schema.json
│   ├── label_mapping.json
│   ├── data/
│   │   └── sample_patient_data.csv
│   ├── models/
│   │   └── drug_response_model.pkl
│   └── venv/ (local only, not pushed to GitHub)
│
└── README.md
```

How to Run Locally (Step-by-Step):
```
Step 1: Clone Repository
git clone https://github.com/aparajitaks/Drug_Response_ML.git

Step 2: Enter the Project Folder
cd Drug_Response_ML

Step 3: Go to ML Service Folder
cd ml-service

Step 4: Create Virtual Environment
python3 -m venv venv

Step 5: Activate Virtual Environment


For Mac/Linux:

source venv/bin/activate


For Windows (PowerShell):

venv\Scripts\activate 

Step 6: Install Dependencies
pip install -r requirements.txt

Step 7: Run Streamlit Dashboard
python -m streamlit run dashboard.py
```

Now open in browser:

http://localhost:8501

```Demo Dataset

A sample CSV is included inside:

ml-service/data/sample_patient_data.csv
```

You can also download and use it directly from the dashboard sidebar.

VS Code Interpreter Setup (Important)

If VS Code shows errors like:

Import "streamlit" could not be resolved

Import "pandas" could not be resolved

Import "fastapi" could not be resolved

It means VS Code is not using the correct virtual environment interpreter.

Fix Interpreter in VS Code

```Open VS Code

Press:

Mac:

Cmd + Shift + P


Windows/Linux:

Ctrl + Shift + P


Search and click:

Python: Select Interpreter


Select the interpreter:

ml-service/venv/bin/python


If it is not visible, click:

Enter interpreter path...


Then manually choose:

ml-service/venv/bin/python

Reload VS Code (Recommended)

After selecting interpreter:

Press:

Cmd + Shift + P


Search and click:

Developer: Reload Window


After this, all missing import warnings will disappear.
```
