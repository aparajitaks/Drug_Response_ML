from fastapi import APIRouter, Query

from app.routes.search import _build_search_payload, _load_reviews


router = APIRouter()


def _effectiveness_score(payload: dict) -> int:
    dist = payload.get("sentiment_distribution", {})
    return int(dist.get("positive", 0)) - int(dist.get("negative", 0))


@router.get("")
def compare(
    drug1: str = Query(..., min_length=1),
    drug2: str = Query(..., min_length=1),
    condition: str = Query(..., min_length=1),
) -> dict:
    condition_value = condition.strip()
    df = _load_reviews()
    drug1_data = _build_search_payload(df, drug1.strip(), condition_value)
    drug2_data = _build_search_payload(df, drug2.strip(), condition_value)

    score1 = _effectiveness_score(drug1_data)
    score2 = _effectiveness_score(drug2_data)

    if score1 > score2:
        simple_insight = "Drug1 has higher effectiveness score. Drug2 has fewer reported negative responses."
    elif score2 > score1:
        simple_insight = "Drug2 has higher effectiveness score. Drug1 has fewer reported negative responses."
    else:
        simple_insight = "Both drugs show similar effectiveness for the selected condition."

    return {
        "drug1_data": drug1_data,
        "drug2_data": drug2_data,
        "simple_insight": simple_insight,
    }
