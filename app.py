import streamlit as st
import pandas as pd

from src.recommender import SBERTVADERRecommender
from src.db import save_recommendation_to_supabase, get_recent_sessions


st.set_page_config(
    page_title="User Cold-Start Product Recommendation",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource(show_spinner="Memuat model SBERT-VADER...")
def load_recommender():
    return SBERTVADERRecommender(model_dir="model")


def render_recommendation_cards(recommendations: pd.DataFrame):
    for _, row in recommendations.iterrows():
        with st.container(border=True):
            st.markdown(f"### #{int(row['rank'])} — {row.get('product_title', 'Produk')}")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Final Score", f"{row.get('final_score', 0):.4f}")

            with col2:
                st.metric("SBERT Similarity", f"{row.get('sbert_similarity', 0):.4f}")

            with col3:
                avg_rating = row.get("avg_rating", None)
                if pd.notna(avg_rating):
                    st.metric("Avg Rating", f"{avg_rating:.2f}")
                else:
                    st.metric("Avg Rating", "-")

            with col4:
                review_count = row.get("review_count", None)
                if pd.notna(review_count):
                    st.metric("Review Count", int(review_count))
                else:
                    st.metric("Review Count", "-")

            st.caption(f"Product ID: {row.get('product_id', '-')}")
            st.caption(f"Category: {row.get('product_category', '-')}")


def main():
    st.title("🛒 User Cold-Start Product Recommendation")
    st.write(
        "Aplikasi ini mensimulasikan pengguna baru yang belum memiliki histori. "
        "User cukup mengisi preferensi awal berupa keyword, lalu sistem menampilkan rekomendasi produk "
        "berdasarkan SBERT Similarity, VADER Sentiment, dan Rating."
    )

    try:
        recommender = load_recommender()
    except Exception as e:
        st.error("Model gagal dimuat.")
        st.exception(e)
        return

    with st.sidebar:
        st.header("⚙️ Konfigurasi Model")
        st.write("Model:", recommender.config.get("model_name", "SBERT-VADER"))
        st.write("SBERT:", recommender.config.get("sbert_model_name", "-"))
        st.write("Alpha SBERT:", recommender.alpha)
        st.write("Beta VADER:", recommender.beta)
        st.write("Gamma Rating:", recommender.gamma)

        st.divider()

        st.header("🕘 Riwayat Terbaru")
        sessions = get_recent_sessions(limit=5)

        if sessions:
            for item in sessions:
                st.caption(
                    f"{item.get('user_name') or 'User'} — "
                    f"{item.get('keyword')} — "
                    f"{item.get('created_at')}"
                )
        else:
            st.caption("Belum ada riwayat atau Supabase belum dikonfigurasi.")

    st.subheader("👤 Simulasi Preferensi User Baru")

    with st.form("recommendation_form"):
        col1, col2 = st.columns([1, 1])

        with col1:
            user_name = st.text_input(
                "Nama user / skenario user",
                value="User Baru 1",
                placeholder="Contoh: User Baru 1",
            )

        with col2:
            category = st.selectbox(
                "Kategori",
                options=["All", "Sports"],
                index=1,
            )

        keyword = st.text_area(
            "Keyword preferensi user",
            placeholder="Contoh: running shoes lightweight comfortable, football jersey, basketball outdoor durable",
            height=110,
        )

        col3, col4 = st.columns([1, 1])

        with col3:
            top_n = st.slider("Jumlah rekomendasi", min_value=5, max_value=20, value=10, step=1)

        with col4:
            candidate_size = st.slider("Jumlah kandidat awal SBERT", min_value=100, max_value=1000, value=500, step=100)

        submitted = st.form_submit_button("Cari Rekomendasi Produk")

    if submitted:
        if not keyword.strip():
            st.warning("Masukkan keyword preferensi terlebih dahulu.")
            return

        with st.spinner("Menghitung rekomendasi produk..."):
            recommendations = recommender.recommend(
                preference_keywords=keyword,
                category=category,
                top_n=top_n,
                candidate_size=candidate_size,
            )

        if recommendations.empty:
            st.warning("Tidak ada rekomendasi yang ditemukan.")
            return

        session_id, error = save_recommendation_to_supabase(
            user_name=user_name,
            keyword=keyword,
            category=category,
            top_n=top_n,
            model_config=recommender.config,
            recommendations=recommendations,
        )

        if error:
            st.warning(f"Rekomendasi berhasil dibuat, tetapi gagal disimpan ke Supabase: {error}")
        else:
            st.success(f"Rekomendasi berhasil dibuat dan disimpan ke Supabase. Session ID: {session_id}")

        st.subheader("📌 Hasil Rekomendasi")

        tab1, tab2 = st.tabs(["Card View", "Table View"])

        with tab1:
            render_recommendation_cards(recommendations)

        with tab2:
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
            )

        csv = recommendations.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download hasil rekomendasi CSV",
            data=csv,
            file_name="recommendation_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()