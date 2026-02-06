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
st.write("Upload patient dataset CSV and get predictions + analytics.")

st.divider()

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Patient Data CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info("Built using Streamlit + ML Model")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.write(f"Total Rows: **{df.shape[0]}**")
    st.write(f"Total Columns: **{df.shape[1]}**")

    st.divider()

    model = load_model()
    label_mapping = load_label_mapping()
    schema = load_schema()

    st.subheader("Feature Selection")

    if schema and "features" in schema:
        feature_cols = schema["features"]
        st.success("Feature schema loaded successfully")
    else:
        st.warning("feature_schema.json not found or invalid. Selecting numeric columns automatically.")
        feature_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    st.write("Using Features:")
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

        st.success("Prediction completed successfully!")

        st.dataframe(df.head(20), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()

    st.subheader("Analytics & Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Prediction Distribution")
        dist = df["Prediction"].value_counts().reset_index()
        dist.columns = ["Prediction", "Count"]

        fig1 = px.bar(dist, x="Prediction", y="Count", title="Prediction Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        if "Confidence" in df.columns:
            st.write("Confidence Distribution")
            fig2 = px.histogram(df, x="Confidence", nbins=20, title="Confidence Score Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Confidence chart not available (model doesn't support predict_proba).")

    st.divider()

    st.subheader("Top High-Risk / Most Confident Predictions")

    if "Confidence" in df.columns:
        top_cases = df.sort_values("Confidence", ascending=False).head(10)
        st.dataframe(top_cases, use_container_width=True)
    else:
        st.info("Top cases table not available (no confidence scores).")

    st.divider()

    st.subheader("Download Prediction Output")

    csv_output = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predicted CSV",
        data=csv_output,
        file_name="drug_response_predictions.csv",
        mime="text/csv"
    )

else:
    st.warning("Please upload a CSV file from the sidebar to start predictions.")
