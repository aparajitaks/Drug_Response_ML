"""
Retrain the drug response model without data leakage.

This script intentionally drops `rating` from the feature set and trains on:
- review (TF-IDF)
- condition (encoded)
- usefulCount (numeric)

Target:
- response_category
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "drug_reviews.csv"
MODEL_PATH = BASE_DIR / "models" / "drug_response_model.pkl"
SCHEMA_PATH = BASE_DIR / "feature_schema.json"
LABEL_MAP_PATH = BASE_DIR / "label_mapping.json"

FEATURE_COLUMNS = ["review", "condition", "usefulCount"]
TARGET_COLUMN = "response_category"


def _to_dense(matrix):
    """Convert sparse matrices to dense arrays for tree-based models."""
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep only required columns and clean nulls.
    data = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    data["review"] = data["review"].fillna("").astype(str)
    data["condition"] = data["condition"].fillna("unknown").astype(str)
    data["usefulCount"] = pd.to_numeric(data["usefulCount"], errors="coerce").fillna(0)
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[TARGET_COLUMN])
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("review_tfidf", TfidfVectorizer(max_features=5000), "review"),
            (
                "condition_encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["condition"],
            ),
            ("useful_count", "passthrough", ["usefulCount"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    feature_schema = {"features": FEATURE_COLUMNS}
    label_mapping = {
        0: "Non-Responder",
        1: "Responder",
        2: "Neutral/Mixed",
    }
    SCHEMA_PATH.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    LABEL_MAP_PATH.write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Feature schema updated: {SCHEMA_PATH}")
    print(f"Label mapping updated: {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
