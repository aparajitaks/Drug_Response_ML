import streamlit as st
import pandas as pd
import joblib
import json
import os
import plotly.express as px

st.set_page_config(page_title="Drug Response ML Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "drug_response_model.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "label_mapping.json")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")
DEMO_CSV_PATH = os.path.join(BASE_DIR, "data", "demo_sample.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_label_mapping():
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_schema():
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r") as f:
            return json.load(f)
    return None

st.title("Drug Response ML Dashboard")
st.write("Upload patient dataset CSV and get predictions with analytics and visualizations.")
st.divider()

st.sidebar.title("Controls")

st.sidebar.markdown("### Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Patient Data CSV", type=["csv"])

st.sidebar.markdown("---")

st.sidebar.markdown("### Demo Dataset Options")

if os.path.exists(DEMO_CSV_PATH):
    with open(DEMO_CSV_PATH, "rb") as file:
        st.sidebar.download_button(
            label="Download Sample CSV",
            data=file,
            file_name="sample_patient_data.csv",
            mime="text/csv"
        )
else:
    st.sidebar.warning("Demo CSV file not found. Add it in data/sample_patient_data.csv")

if st.sidebar.button("Use Demo Dataset"):
    if os.path.exists(DEMO_CSV_PATH):
        demo_df = pd.read_csv(DEMO_CSV_PATH)
        st.session_state["uploaded_df"] = demo_df
        st.sidebar.success("Demo dataset loaded successfully.")
    else:
        st.sidebar.error("Demo CSV file not found.")

st.sidebar.markdown("---")

st.sidebar.markdown("### Dataset Management")

if st.sidebar.button("Clear Dataset"):
    if "uploaded_df" in st.session_state:
        del st.session_state["uploaded_df"]
    st.sidebar.success("Dataset cleared successfully.")

st.sidebar.markdown("---")
st.sidebar.info("Built using Streamlit and a trained ML model.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["uploaded_df"] = df

if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
else:
    st.warning("Please upload a CSV file or select the demo dataset from the sidebar to start.")
    st.stop()

st.subheader("Dataset Preview")

col1, col2 = st.columns(2)
with col1:
    st.write(f"Total Rows: **{df.shape[0]}**")
with col2:
    st.write(f"Total Columns: **{df.shape[1]}**")

st.dataframe(df.head(10), use_container_width=True)

st.divider()

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

label_mapping = load_label_mapping()
schema = load_schema()

st.subheader("Feature Selection")

if schema and "features" in schema:
    feature_cols = schema["features"]
    st.success("Feature schema loaded successfully.")
else:
    st.warning("feature_schema.json not found or invalid. Selecting numeric columns automatically.")
    feature_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

with st.expander("View Feature Columns"):
    st.code(feature_cols)

missing_cols = [col for col in feature_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing Columns in CSV: {missing_cols}")
    st.stop()

X = df[feature_cols]

st.divider()

st.subheader("Prediction Results")

try:
    predictions = model.predict(X)
    df["Prediction"] = predictions

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        df["Confidence"] = proba.max(axis=1)

    if label_mapping:
        df["Prediction_Label"] = df["Prediction"].astype(str).map(label_mapping)

    st.success("Prediction completed successfully.")
    st.dataframe(df.head(20), use_container_width=True)

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.divider()

st.subheader("Analytics and Visualization")

colA, colB = st.columns(2)

with colA:
    st.write("Prediction Distribution")
    dist = df["Prediction"].value_counts().reset_index()
    dist.columns = ["Prediction", "Count"]

    fig1 = px.bar(dist, x="Prediction", y="Count", title="Prediction Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    if "Confidence" in df.columns:
        st.write("Confidence Distribution")
        fig2 = px.histogram(df, x="Confidence", nbins=20, title="Confidence Score Distribution")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Confidence chart not available because model does not support probability prediction.")

st.divider()

st.subheader("Top Most Confident Predictions")

if "Confidence" in df.columns:
    top_cases = df.sort_values("Confidence", ascending=False).head(10)
    st.dataframe(top_cases, use_container_width=True)
else:
    st.info("Confidence scores are not available.")

st.divider()

st.subheader("Download Prediction Output")

csv_output = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Predicted CSV",
    data=csv_output,
    file_name="drug_response_predictions.csv",
    mime="text/csv"
)
