import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.request import PredictionRequest
from app.services.model_loader import get_model


logger = logging.getLogger(__name__)
DEFAULT_LABEL_MAPPING = {"0": "Non-Responder", "1": "Responder", "2": "Neutral/Mixed"}


def _load_label_mapping() -> dict[str, str]:
    label_path = Path(__file__).resolve().parents[3] / "ml-service" / "label_mapping.json"
    try:
        with label_path.open("r", encoding="utf-8") as file_obj:
            mapping = json.load(file_obj)
            if isinstance(mapping, dict):
                return {str(k): str(v) for k, v in mapping.items()}
    except Exception:
        logger.warning("Could not load label mapping. Using defaults.")
    return DEFAULT_LABEL_MAPPING


def _impact_level(value: float, max_abs: float) -> str:
    if max_abs <= 0:
        return "low"
    ratio = abs(value) / max_abs
    if ratio >= 0.66:
        return "high"
    if ratio >= 0.33:
        return "medium"
    return "low"


def _extract_class_shap_values(raw_shap_values: Any, prediction_class: int) -> np.ndarray:
    # Preferred indexing flow for multiclass TreeExplainer outputs.
    if isinstance(raw_shap_values, list) and len(raw_shap_values) > prediction_class:
        return np.asarray(raw_shap_values[prediction_class][0], dtype=float)

    # Compatibility fallback for alternate SHAP return formats.
    shap_array = np.asarray(raw_shap_values)
    if shap_array.ndim == 3 and shap_array.shape[2] > prediction_class:
        return shap_array[0, :, prediction_class].astype(float)
    if shap_array.ndim == 2:
        return shap_array[0].astype(float)
    return np.asarray([], dtype=float)


def _compute_shap_explanation(model: Any, input_df: pd.DataFrame, prediction_class: int) -> list[dict[str, str]]:
    try:
        import shap
    except Exception:
        logger.warning("shap is not installed. Returning empty explanation.")
        return []

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        # SHAP must run on transformed feature space, not directly on full pipeline.
        transformed = preprocessor.transform(input_df)
        if "to_dense" in model.named_steps:
            transformed = model.named_steps["to_dense"].transform(transformed)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        explainer = shap.TreeExplainer(classifier)
        raw_shap_values = explainer.shap_values(transformed, check_additivity=False)
        shap_values = _extract_class_shap_values(raw_shap_values, prediction_class)

        tfidf = preprocessor.named_transformers_["review_tfidf"]
        tfidf_features = tfidf.get_feature_names_out().tolist()
        all_features = tfidf_features + ["condition_encoded", "usefulCount"]

        if shap_values.size == 0:
            return []

        max_abs = float(np.max(np.abs(shap_values))) if shap_values.size else 0.0
        top_indices = np.argsort(np.abs(shap_values))[-3:][::-1]
        explanations: list[dict[str, str]] = []
        for idx in top_indices:
            value = float(shap_values[idx])
            explanations.append(
                {
                    "feature": all_features[idx] if idx < len(all_features) else f"feature_{idx}",
                    "direction": "positive" if value >= 0 else "negative",
                    "impact": _impact_level(value, max_abs),
                }
            )
        return explanations
    except Exception:
        logger.exception("Failed to compute SHAP explanation")
        return []


def run_prediction(payload: PredictionRequest) -> dict[str, object]:
    model = get_model()
    label_mapping = _load_label_mapping()

    input_df = pd.DataFrame(
        [
            {
                "review": payload.review,
                "condition": payload.condition,
                "usefulCount": payload.usefulCount,
            }
        ]
    )

    prediction_class = int(model.predict(input_df)[0])
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(input_df)[0]))

    prediction_label = label_mapping.get(str(prediction_class), "Unknown")
    shap_explanation = _compute_shap_explanation(model, input_df, prediction_class)

    return {
        "prediction_class": prediction_class,
        "prediction_label": prediction_label,
        "confidence": confidence,
        "shap_explanation": shap_explanation,
    }
