from fastapi import APIRouter, Query

from app.routes.search import _build_search_payload, _load_reviews


router = APIRouter()


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

    drug1_name = drug1_data["drug"]
    drug2_name = drug2_data["drug"]
    positive_1 = int(drug1_data["sentiment_distribution"].get("positive", 0))
    positive_2 = int(drug2_data["sentiment_distribution"].get("positive", 0))
    negative_1 = int(drug1_data["sentiment_distribution"].get("negative", 0))
    negative_2 = int(drug2_data["sentiment_distribution"].get("negative", 0))

    if positive_1 > positive_2:
        lower_negative_name = drug1_name if negative_1 < negative_2 else drug2_name
        simple_insight = (
            f"{drug1_name.title()} has a higher positive response rate ({positive_1}%) "
            f"compared to {drug2_name.title()} ({positive_2}%). "
            f"{lower_negative_name.title()} shows fewer negative responses."
        )
    elif positive_2 > positive_1:
        lower_negative_name = drug2_name if negative_2 < negative_1 else drug1_name
        simple_insight = (
            f"{drug2_name.title()} has a higher positive response rate ({positive_2}%) "
            f"compared to {drug1_name.title()} ({positive_1}%). "
            f"{lower_negative_name.title()} shows fewer negative responses."
        )
    else:
        if negative_1 < negative_2:
            lower_negative_name = drug1_name
        elif negative_2 < negative_1:
            lower_negative_name = drug2_name
        else:
            lower_negative_name = ""
        simple_insight = (
            f"{drug1_name.title()} and {drug2_name.title()} have similar positive response rates ({positive_1}%). "
            + (
                f"{lower_negative_name.title()} shows fewer negative responses."
                if lower_negative_name
                else "Both show similar negative response rates."
            )
        )

    return {
        "drug1_data": drug1_data,
        "drug2_data": drug2_data,
        "simple_insight": simple_insight,
    }
