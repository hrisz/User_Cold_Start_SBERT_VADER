import os
import json
import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from sentence_transformers import SentenceTransformer


class SBERTVADERRecommender:
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir

        config_path = os.path.join(model_dir, "model_config.json")
        catalog_path = os.path.join(model_dir, "product_catalog.csv")
        embedding_path = os.path.join(model_dir, "product_embeddings.npy")
        local_sbert_path = os.path.join(model_dir, "sbert_model")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File config tidak ditemukan: {config_path}")

        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"File product catalog tidak ditemukan: {catalog_path}")

        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"File embedding tidak ditemukan: {embedding_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.catalog = pd.read_csv(catalog_path)
        self.product_embeddings = np.load(embedding_path, mmap_mode="r")

        if len(self.catalog) != self.product_embeddings.shape[0]:
            raise ValueError(
                f"Jumlah catalog dan embedding tidak sama. "
                f"Catalog: {len(self.catalog)}, Embedding: {self.product_embeddings.shape[0]}"
            )

        if os.path.exists(local_sbert_path):
            self.sbert_model = SentenceTransformer(local_sbert_path)
        else:
            self.sbert_model = SentenceTransformer(self.config["sbert_model_name"])

        self.alpha = float(self.config.get("alpha_sbert", 0.95))
        self.beta = float(self.config.get("beta_vader", 0.03))
        self.gamma = float(self.config.get("gamma_rating", 0.02))
        self.default_candidate_size = int(self.config.get("top_candidate_size", 500))

    def recommend(
        self,
        preference_keywords: str,
        category: str | None = None,
        top_n: int = 10,
        candidate_size: int | None = None,
    ) -> pd.DataFrame:
        if not preference_keywords or preference_keywords.strip() == "":
            return pd.DataFrame()

        if candidate_size is None:
            candidate_size = self.default_candidate_size

        query_text = preference_keywords.strip()

        if category and category.strip() and category.lower() != "all":
            query_text = f"{category.strip()} {query_text}"

        query_embedding = self.sbert_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]

        similarity_scores = np.asarray(self.product_embeddings @ query_embedding).reshape(-1)

        candidate_size = min(candidate_size, len(self.catalog))
        candidate_idx = np.argsort(-similarity_scores)[:candidate_size]

        candidate_data = self.catalog.iloc[candidate_idx].copy()
        candidate_data["sbert_similarity"] = similarity_scores[candidate_idx]

        sentiment_col = self._first_existing_column(
            candidate_data,
            ["weighted_sentiment_norm", "avg_sentiment_norm", "sentiment_norm"],
        )

        rating_col = self._first_existing_column(
            candidate_data,
            ["weighted_rating_norm", "rating_norm", "avg_rating_norm"],
        )

        if sentiment_col is None:
            candidate_data["sentiment_score_used"] = 0.5
        else:
            candidate_data["sentiment_score_used"] = (
                pd.to_numeric(candidate_data[sentiment_col], errors="coerce")
                .fillna(0.5)
                .clip(0, 1)
            )

        if rating_col is None:
            if "avg_rating" in candidate_data.columns:
                candidate_data["rating_score_used"] = (
                    (pd.to_numeric(candidate_data["avg_rating"], errors="coerce").fillna(3) - 1) / 4
                ).clip(0, 1)
            else:
                candidate_data["rating_score_used"] = 0.5
        else:
            candidate_data["rating_score_used"] = (
                pd.to_numeric(candidate_data[rating_col], errors="coerce")
                .fillna(0.5)
                .clip(0, 1)
            )

        candidate_data["final_score"] = (
            self.alpha * candidate_data["sbert_similarity"]
            + self.beta * candidate_data["sentiment_score_used"]
            + self.gamma * candidate_data["rating_score_used"]
        )

        if category and category.strip() and category.lower() != "all":
            category_lower = category.lower().strip()

            if "product_category" in candidate_data.columns:
                filtered = candidate_data[
                    candidate_data["product_category"]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(category_lower, regex=False)
                ]

                if len(filtered) > 0:
                    candidate_data = filtered

        result = (
            candidate_data
            .sort_values("final_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        result.insert(0, "rank", range(1, len(result) + 1))

        display_cols = [
            "rank",
            "product_id",
            "product_title",
            "product_category",
            "avg_rating",
            "review_count",
            "sbert_similarity",
            "sentiment_score_used",
            "rating_score_used",
            "final_score",
        ]

        display_cols = [col for col in display_cols if col in result.columns]

        return result[display_cols]

    @staticmethod
    def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None