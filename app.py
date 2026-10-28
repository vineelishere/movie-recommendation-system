"""
Streamlit app for a content-based movie recommender.

Expected project layout:
.
├── app.py
└── model/
    ├── movie_list.pkl
    ├── vectorizer.pkl       # optional for this app
    └── tag_vectors.pkl      # optional for this app
    └── similarity.pkl       # legacy option, supported if present

Set TMDB_API_KEY as an environment variable before running the app.
"""

import os
import pickle
from pathlib import Path
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
MOVIE_LIST_PATH = MODEL_DIR / "movie_list.pkl"
SIMILARITY_PATH = MODEL_DIR / "similarity.pkl"
TAG_VECTORS_PATH = MODEL_DIR / "tag_vectors.pkl"
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
REQUEST_TIMEOUT = 8


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading recommendation model...")
def load_artifacts() -> tuple[Any, Any, str]:
    """Load the movie table and either a similarity matrix or tag vectors."""
    if not MOVIE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"Movie data not found at {MOVIE_LIST_PATH}. "
            "Run the notebook first and save movie_list.pkl into the model/ folder."
        )

    with MOVIE_LIST_PATH.open("rb") as file:
        movie_data = pickle.load(file)

    if not hasattr(movie_data, "columns") or "title" not in movie_data.columns:
        raise ValueError("movie_list.pkl must contain a DataFrame with a 'title' column.")

    if "movie_id" not in movie_data.columns:
        raise ValueError("movie_list.pkl must contain a 'movie_id' column for poster lookup.")

    if SIMILARITY_PATH.exists():
        with SIMILARITY_PATH.open("rb") as file:
            similarity_data = pickle.load(file)
        return movie_data.reset_index(drop=True), similarity_data, "similarity"

    if TAG_VECTORS_PATH.exists():
        with TAG_VECTORS_PATH.open("rb") as file:
            tag_vectors_data = pickle.load(file)
        return movie_data.reset_index(drop=True), tag_vectors_data, "vectors"

    raise FileNotFoundError(
        "No recommendation features found. Add model/similarity.pkl (legacy model) "
        "or model/tag_vectors.pkl (sparse feature matrix)."
    )


def get_recommendations(
    movie_title: str,
    movie_data,
    features,
    feature_type: str,
    number: int = 5,
):
    """Return recommended rows with similarity scores, excluding the selected movie."""
    matches = movie_data.index[
        movie_data["title"].astype(str).str.casefold() == movie_title.casefold()
    ].tolist()

    if not matches:
        raise ValueError(f"Movie title not found: {movie_title}")

    selected_index = matches[0]

    if feature_type == "similarity":
        if len(features) != len(movie_data):
            raise ValueError(
                "The similarity matrix size does not match movie_list.pkl. "
                "Regenerate both artifacts from the same dataset."
            )
        scores = features[selected_index]
        # Supports list, NumPy array, and other indexable row representations.
        ranked_indices = sorted(
            range(len(movie_data)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )
    else:
        if features.shape[0] != len(movie_data):
            raise ValueError(
                "The feature matrix row count does not match movie_list.pkl. "
                "Regenerate both artifacts from the same dataset."
            )
        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity(features[selected_index], features).ravel()
        ranked_indices = scores.argsort()[::-1].tolist()

    selected_id = selected_index
    results = []
    for idx in ranked_indices:
        if idx == selected_id:
            continue
        row = movie_data.iloc[idx]
        results.append(
            {
                "movie_id": row["movie_id"],
                "title": row["title"],
                "similarity_score": float(scores[idx]),
            }
        )
        if len(results) >= number:
            break

    import pandas as pd

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# TMDB poster lookup
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_movie_details(movie_id: int | str) -> dict:
    """Fetch movie details from TMDB; return an empty dictionary on API errors."""
    if not TMDB_API_KEY:
        return {}

    try:
        response = requests.get(
            f"{TMDB_API_BASE}/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "overview": data.get("overview") or "",
            "poster_url": (
                f"{TMDB_IMAGE_BASE}{data['poster_path']}"
                if data.get("poster_path")
                else None
            ),
            "release_date": data.get("release_date") or "",
            "vote_average": data.get("vote_average"),
            "runtime": data.get("runtime"),
        }
    except (requests.RequestException, ValueError, TypeError):
        return {}


def poster_url_from_details(details: dict) -> str | None:
    return details.get("poster_url")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.title("🎬 Movie Recommender")
    st.caption("Discover films with similar genres, themes, cast, and story elements.")

    try:
        movies, features, feature_type = load_artifacts()
    except (FileNotFoundError, ValueError, pickle.UnpicklingError) as exc:
        st.error(str(exc))
        st.info(
            "Expected files: model/movie_list.pkl and either "
            "model/similarity.pkl or model/tag_vectors.pkl."
        )
        st.stop()

    # Preserve row order exactly: feature rows must stay aligned with movie rows.
    if movies["title"].isna().any():
        st.warning(
            "Some titles are missing in the saved movie table. "
            "Please clean the source notebook and regenerate the model artifacts."
        )
        st.stop()
    titles = movies["title"].astype(str).tolist()

    if not titles:
        st.error("The movie dataset contains no usable titles.")
        st.stop()

    with st.sidebar:
        st.header("Find a movie")
        selected_movie = st.selectbox(
            "Choose a movie",
            options=titles,
            index=0,
            help="Start typing to search the movie list.",
        )
        recommendation_count = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
        )
        show_scores = st.toggle("Show similarity scores", value=False)
        st.divider()
        st.caption("Recommendations are based on metadata similarity, not user ratings.")

    if st.button("✨ Recommend movies", type="primary", use_container_width=True):
        try:
            recommendations = get_recommendations(
                selected_movie,
                movies,
                features,
                feature_type,
                number=recommendation_count,
            )
        except (ValueError, IndexError, TypeError) as exc:
            st.error(f"Could not generate recommendations: {exc}")
            st.stop()

        if recommendations.empty:
            st.warning("No recommendations were found for this movie.")
            st.stop()

        st.subheader(f"Because you selected **{selected_movie}**")
        columns = st.columns(min(5, len(recommendations)))

        for position, (_, recommendation) in enumerate(recommendations.iterrows()):
            movie_id = recommendation["movie_id"]
            details = fetch_movie_details(movie_id)
            with columns[position % len(columns)]:
                poster = poster_url_from_details(details)
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='height:270px;display:flex;align-items:center;"
                        "justify-content:center;border:1px solid #444;border-radius:8px;'>"
                        "🎞️<br>Poster unavailable</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**{recommendation['title']}**")
                if details.get("release_date"):
                    st.caption(details["release_date"][:4])
                if details.get("vote_average") is not None:
                    st.caption(f"TMDB rating: {details['vote_average']:.1f}/10")
                if show_scores:
                    st.caption(
                        f"Similarity: {recommendation['similarity_score']:.3f}"
                    )
                if details.get("overview"):
                    with st.expander("Overview"):
                        st.write(details["overview"])

    else:
        st.info("Choose a movie in the sidebar, then select **Recommend movies**.")


if __name__ == "__main__":
    main()
