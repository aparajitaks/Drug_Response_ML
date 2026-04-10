import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)
TABLE_NAME = "drug_reviews"


def _normalize_reviews_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["drugName"] = normalized.get("drugName", "").fillna("").astype(str)
    normalized["condition"] = normalized.get("condition", "").fillna("").astype(str)
    normalized["review"] = normalized.get("review", "").fillna("").astype(str)
    normalized["usefulCount"] = pd.to_numeric(normalized.get("usefulCount", 0), errors="coerce").fillna(0).astype(int)
    normalized["rating"] = pd.to_numeric(normalized.get("rating"), errors="coerce")
    normalized["response_category"] = pd.to_numeric(normalized.get("response_category"), errors="coerce")
    return normalized


def _load_reviews_from_csv() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[3] / "ml-service" / "data" / "drug_reviews.csv"
    logger.info("Loading reviews from CSV fallback: %s", data_path)
    df = pd.read_csv(data_path)
    return _normalize_reviews_df(df)


def _create_supabase_client() -> Any | None:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None

    try:
        from supabase import Client, create_client

        client: Client = create_client(url, key)
        return client
    except Exception:
        logger.exception("Failed to initialize Supabase client. Falling back to CSV.")
        return None


def _load_reviews_from_supabase() -> pd.DataFrame | None:
    client = _create_supabase_client()
    if client is None:
        logger.info("Supabase config missing or unavailable. Using CSV fallback.")
        return None

    try:
        response = client.table(TABLE_NAME).select("*").limit(50000).execute()
        data = response.data or []
        if not data:
            logger.warning("Supabase table '%s' returned no rows. Using CSV fallback.", TABLE_NAME)
            return None
        df = pd.DataFrame(data)
        logger.info("Loaded %s rows from Supabase table '%s'.", len(df), TABLE_NAME)
        return _normalize_reviews_df(df)
    except Exception:
        logger.exception("Supabase query failed. Falling back to CSV.")
        return None


def load_reviews() -> pd.DataFrame:
    supabase_df = _load_reviews_from_supabase()
    if supabase_df is not None:
        return supabase_df
    return _load_reviews_from_csv()
