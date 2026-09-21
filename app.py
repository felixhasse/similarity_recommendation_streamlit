"""Deployment-only Streamlit UI with selectable fashion embedding models."""

from __future__ import annotations

import secrets
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from index_store import (
    MODEL_SPECS,
    DeploymentDataError,
    load_embedding_index,
    resolve_data_path,
)
from recommender import (
    EmbeddingIndex,
    RecommendationError,
    aggregate_preference,
    canonicalize_clothing_type,
    choose_outfit_indices,
    rank_candidates,
    rank_candidates_by_type,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTFIT_COUNT = 15
MIN_OUTFIT_COUNT = 5
MAX_OUTFIT_COUNT = 30
RECOMMENDATION_COUNT = 10
TYPE_RECOMMENDATION_COUNT = 5
GENDER_OPTIONS = {
    "Masculine": "Men",
    "Feminine": "Women",
    "Both": "Both",
}
MODEL_OPTIONS = {spec["label"]: key for key, spec in MODEL_SPECS.items()}


st.set_page_config(
    page_title="Style Compass",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem;}
      h1 {letter-spacing: -0.045em;}
      div[data-testid="stImage"] img {
        aspect-ratio: 3 / 4;
        object-fit: contain;
        background: #f3f1ed;
        border-radius: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _load_indexes(
    app_dir: str, model_key: str
) -> tuple[EmbeddingIndex, EmbeddingIndex]:
    root = Path(app_dir)
    return (
        load_embedding_index("clothing", root, model_key),
        load_embedding_index("outfits", root, model_key),
    )


def _start_rating_session(
    outfits: EmbeddingIndex, gender: str, outfit_count: int
) -> None:
    selected = choose_outfit_indices(outfits.manifest, gender, outfit_count)
    st.session_state.rating_gender = gender
    st.session_state.rating_count = outfit_count
    st.session_state.outfit_indices = selected.tolist()
    st.session_state.rating_nonce = secrets.token_hex(6)
    st.session_state.pop("recommendations", None)
    st.session_state.pop("type_recommendations", None)
    st.session_state.pop("recommendation_signature", None)


def _product_text(row: pd.Series) -> str:
    value = row.get("productDisplayName", "")
    if pd.isna(value) or not str(value).strip():
        return str(row.get("articleType", "Clothing item"))
    return str(value)


def _product_details(row: pd.Series) -> str:
    item_type = canonicalize_clothing_type(
        row.get("type", row.get("articleType", "Clothing item"))
    )
    colors = row.get("colors", row.get("baseColour", "Unspecified"))
    return f"{item_type} · {colors}"


st.title("Find your style direction")
st.write(
    "Choose how many outfits to rate, then receive recommendations shaped by "
    "what you like—and what you do not."
)

model_label = st.selectbox(
    "Embedding model",
    list(MODEL_OPTIONS),
    help=(
        "The same ratings are projected through the selected model's precomputed "
        "outfit and clothing embeddings."
    ),
)
model_key = MODEL_OPTIONS[model_label]
try:
    clothing_index, outfit_index = _load_indexes(str(APP_DIR), model_key)
except DeploymentDataError as error:
    st.error(f"Deployment data is incomplete: {error}")
    st.stop()

controls = st.columns([1.5, 1.5, 1, 2.2], vertical_alignment="bottom")
with controls[0]:
    gender_label = st.radio(
        "Show outfits for",
        list(GENDER_OPTIONS),
        horizontal=True,
    )
gender = GENDER_OPTIONS[gender_label]

with controls[1]:
    outfit_count = st.slider(
        "Outfits to rate",
        min_value=MIN_OUTFIT_COUNT,
        max_value=MAX_OUTFIT_COUNT,
        value=DEFAULT_OUTFIT_COUNT,
        step=1,
    )

if (
    st.session_state.get("rating_gender") != gender
    or st.session_state.get("rating_count") != outfit_count
    or "outfit_indices" not in st.session_state
):
    _start_rating_session(outfit_index, gender, outfit_count)

with controls[2]:
    if st.button("↻ New set", width="stretch"):
        _start_rating_session(outfit_index, gender, outfit_count)
        st.rerun()

with controls[3]:
    lambda_negative = st.slider(
        "Dislike weight (lambda)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="1.0 gives the average disliked embedding its full negative weight.",
    )

selected_indices = np.asarray(st.session_state.outfit_indices, dtype=int)
selected_outfits = outfit_index.manifest.iloc[selected_indices]
st.subheader(f"Your {outfit_count} outfits")
st.caption("Every outfit needs one rating before recommendations can be generated.")

ratings: list[str | None] = []
card_columns = st.columns(3)
for card_number, (position, row) in enumerate(selected_outfits.iterrows(), start=1):
    with card_columns[(card_number - 1) % 3]:
        st.image(
            str(resolve_data_path(row["image_path"], APP_DIR)),
            width="stretch",
        )
        rating = st.radio(
            f"Rate outfit {card_number}",
            options=("Like", "Don't like"),
            index=None,
            horizontal=True,
            key=f"rating_{st.session_state.rating_nonce}_{position}",
            label_visibility="collapsed",
        )
        ratings.append(rating)

rated_count = sum(rating is not None for rating in ratings)
st.progress(
    rated_count / outfit_count,
    text=f"{rated_count} of {outfit_count} rated",
)

current_signature = (tuple(ratings), float(lambda_negative), gender, model_key)
if st.session_state.get("recommendation_signature") != current_signature:
    st.session_state.pop("recommendations", None)
    st.session_state.pop("type_recommendations", None)

generate = st.button(
    "Show my recommendations",
    type="primary",
    width="stretch",
    disabled=rated_count != outfit_count,
)

if generate:
    selected_embeddings = np.asarray(outfit_index.embeddings[selected_indices])
    liked_mask = np.asarray([rating == "Like" for rating in ratings])
    disliked_mask = ~liked_mask
    try:
        preference = aggregate_preference(
            selected_embeddings[liked_mask],
            selected_embeddings[disliked_mask],
            lambda_negative=lambda_negative,
        )
        recommendations = rank_candidates(
            preference,
            clothing_index,
            gender,
            top_k=RECOMMENDATION_COUNT,
        )
        type_recommendations = rank_candidates_by_type(
            preference,
            clothing_index,
            gender,
            top_k=TYPE_RECOMMENDATION_COUNT,
        )
    except RecommendationError as error:
        st.warning(str(error))
    else:
        st.session_state.recommendations = recommendations
        st.session_state.type_recommendations = type_recommendations
        st.session_state.recommendation_signature = current_signature

if "recommendations" in st.session_state:
    recommendations = st.session_state.recommendations
    st.divider()
    st.subheader("Your closest matches")
    st.caption(
        f"Ranked by cosine similarity to your normalized {model_label} preference vector."
    )
    result_columns = st.columns(5)
    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
        with result_columns[(rank - 1) % 5]:
            st.image(
                str(resolve_data_path(row["image_path"], APP_DIR)),
                width="stretch",
            )
            st.markdown(f"**{rank}. {_product_text(row)}**")
            st.caption(
                f"{_product_details(row)}  \n"
                f"Similarity: {row['similarity']:.3f}"
            )

    st.divider()
    st.subheader("Closest matches by clothing type")
    st.caption(
        f"Up to {TYPE_RECOMMENDATION_COUNT} matches for every available type in "
        "the selected catalog."
    )
    for item_type, type_results in st.session_state.type_recommendations.items():
        st.markdown(f"#### {item_type}")
        type_columns = st.columns(5)
        for rank, (_, row) in enumerate(type_results.iterrows(), start=1):
            with type_columns[(rank - 1) % 5]:
                st.image(
                    str(resolve_data_path(row["image_path"], APP_DIR)),
                    width="stretch",
                )
                st.markdown(f"**{rank}. {_product_text(row)}**")
                st.caption(
                    f"{_product_details(row)}  \n"
                    f"Similarity: {row['similarity']:.3f}"
                )

st.divider()
st.caption(
    "Precomputed embeddings: [patrickjohncyh/fashion-clip]"
    "(https://huggingface.co/patrickjohncyh/fashion-clip) and "
    "[Marqo/marqo-fashionSigLIP]"
    "(https://huggingface.co/Marqo/marqo-fashionSigLIP) · Recommendations: "
    "adult and Unisex front views from the [Second-Hand Fashion Dataset v3]"
    "(https://huggingface.co/datasets/chibifire/zenodo-second-hand-fashion-v3) "
    "including Unisex items when Both is selected "
    "([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))"
)
