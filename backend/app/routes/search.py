import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.services.supabase_service import load_reviews


router = APIRouter()


def _load_reviews() -> pd.DataFrame:
    return load_reviews()


def _build_search_payload(df: pd.DataFrame, drug: str, condition: str) -> dict:
    drug_mask = df["drugName"].str.contains(drug, case=False, na=False)
    condition_mask = df["condition"].str.contains(condition, case=False, na=False)
    filtered = df[drug_mask & condition_mask].copy()

    total_reviews = int(len(filtered))
    if total_reviews == 0:
        raise HTTPException(status_code=404, detail="No reviews found for this drug/condition")

    positive = int((filtered["response_category"] == 1).sum())
    negative = int((filtered["response_category"] == 0).sum())
    neutral = int((filtered["response_category"] == 2).sum())

    sentiment_distribution = {
        "positive": int(round((positive / total_reviews) * 100)),
        "neutral": int(round((neutral / total_reviews) * 100)),
        "negative": int(round((negative / total_reviews) * 100)),
    }

    # Keep percentages coherent after rounding.
    drift = 100 - sum(sentiment_distribution.values())
    if drift != 0:
        sentiment_distribution["neutral"] += drift
        sentiment_distribution["neutral"] = max(0, min(100, sentiment_distribution["neutral"]))

    top = (
        filtered.sort_values("usefulCount", ascending=False)
        .head(10)[["review", "usefulCount", "rating"]]
        .to_dict(orient="records")
    )

    return {
        "drug": drug,
        "condition": condition,
        "total_reviews": total_reviews,
        "sentiment_distribution": sentiment_distribution,
        "top_reviews": top,
    }


@router.get("")
def search(
    drug: str = Query(..., min_length=1),
    condition: str = Query(..., min_length=1),
) -> dict:
    df = _load_reviews()
    return _build_search_payload(df, drug.strip(), condition.strip())
