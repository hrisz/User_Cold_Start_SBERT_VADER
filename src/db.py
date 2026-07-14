import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client | None:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]

        if not url or not key:
            return None

        return create_client(url, key)

    except Exception:
        return None


def save_recommendation_to_supabase(
    user_name: str,
    keyword: str,
    category: str,
    top_n: int,
    model_config: dict,
    recommendations,
):
    supabase = get_supabase_client()

    if supabase is None:
        return None, "Supabase belum dikonfigurasi. Data tidak disimpan."

    try:
        session_payload = {
            "user_name": user_name,
            "keyword": keyword,
            "category": category,
            "top_n": top_n,
            "model_name": model_config.get("model_name", "SBERT_VADER_User_Cold_Start_Recommender"),
            "alpha_sbert": model_config.get("alpha_sbert"),
            "beta_vader": model_config.get("beta_vader"),
            "gamma_rating": model_config.get("gamma_rating"),
        }

        session_response = (
            supabase
            .table("recommendation_sessions")
            .insert(session_payload)
            .execute()
        )

        if not session_response.data:
            return None, "Session berhasil dikirim, tetapi ID tidak dikembalikan."

        session_id = session_response.data[0]["id"]

        item_payloads = []

        for _, row in recommendations.iterrows():
            item_payloads.append({
                "session_id": session_id,
                "rank": int(row.get("rank", 0)),
                "product_id": str(row.get("product_id", "")),
                "product_title": str(row.get("product_title", "")),
                "product_category": str(row.get("product_category", "")),
                "avg_rating": _safe_float(row.get("avg_rating")),
                "review_count": _safe_int(row.get("review_count")),
                "sbert_similarity": _safe_float(row.get("sbert_similarity")),
                "final_score": _safe_float(row.get("final_score")),
            })

        if item_payloads:
            (
                supabase
                .table("recommendation_items")
                .insert(item_payloads)
                .execute()
            )

        return session_id, None

    except Exception as e:
        return None, str(e)


def get_recent_sessions(limit: int = 10):
    supabase = get_supabase_client()

    if supabase is None:
        return []

    try:
        response = (
            supabase
            .table("recommendation_sessions")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data or []

    except Exception:
        return []


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None