from pathlib import Path
import os
import pandas as pd

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

CSV_PATH = Path(__file__).resolve().parents[3] / "ml-service" / "data" / "drug_reviews.csv"


def _is_supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def get_drug_reviews(drug_name: str, condition: str) -> pd.DataFrame:
    """
    Returns filtered reviews for a drug + condition.
    Uses Supabase if configured, otherwise falls back to CSV.
    """
    if _is_supabase_configured():
        try:
            from supabase import create_client

            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            response = (
                client.table("drug_reviews")
                .select("*")
                .ilike("drugName", f"%{drug_name}%")
                .ilike("condition", f"%{condition}%")
                .execute()
            )
            return pd.DataFrame(response.data)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Supabase query failed, falling back to CSV: {e}"
            )
    # CSV fallback
    df = pd.read_csv(CSV_PATH)
    mask = (
        df["drugName"].str.contains(drug_name, case=False, na=False)
        & df["condition"].str.contains(condition, case=False, na=False)
    )
    return df[mask].copy()


def get_all_drugs() -> list[str]:
    """
    Returns list of distinct drug names.
    Uses Supabase if configured, otherwise falls back to CSV.
    """
    if _is_supabase_configured():
        try:
            from supabase import create_client

            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            response = client.table("drug_reviews").select("drugName").execute()
            names = list({row["drugName"] for row in response.data if row.get("drugName")})
            return sorted(names)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Supabase get_all_drugs failed, falling back to CSV: {e}"
            )
    df = pd.read_csv(CSV_PATH)
    return sorted(df["drugName"].dropna().unique().tolist())
