"""Deployment-only Streamlit UI for FashionCLIP recommendations."""

from __future__ import annotations

import secrets
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from index_store import DeploymentDataError, load_embedding_index, resolve_data_path
from recommender import (
    EmbeddingIndex,
    RecommendationError,
    aggregate_preference,
    choose_outfit_indices,
    rank_candidates,
)


APP_DIR = Path(__file__).resolve().parent
OUTFIT_COUNT = 15
RECOMMENDATION_COUNT = 10
GENDER_OPTIONS = {"Man": "Men", "Woman": "Women"}


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
def _load_indexes(app_dir: str) -> tuple[EmbeddingIndex, EmbeddingIndex]:
    root = Path(app_dir)
    return load_embedding_index("clothing", root), load_embedding_index("outfits", root)


def _start_rating_session(outfits: EmbeddingIndex, gender: str) -> None:
    selected = choose_outfit_indices(outfits.manifest, gender, OUTFIT_COUNT)
    st.session_state.rating_gender = gender
    st.session_state.outfit_indices = selected.tolist()
    st.session_state.rating_nonce = secrets.token_hex(6)
    st.session_state.pop("recommendations", None)
    st.session_state.pop("recommendation_signature", None)


def _product_text(row: pd.Series) -> str:
    value = row.get("productDisplayName", "")
    if pd.isna(value) or not str(value).strip():
        return str(row.get("articleType", "Clothing item"))
    return str(value)


try:
    clothing_index, outfit_index = _load_indexes(str(APP_DIR))
except DeploymentDataError as error:
    st.error(f"Deployment data is incomplete: {error}")
    st.stop()

st.title("Find your style direction")
st.write(
    "Rate 15 outfits and receive ten apparel recommendations shaped by what you "
    "like—and what you do not."
)

controls = st.columns([1.4, 1, 2.6], vertical_alignment="bottom")
with controls[0]:
    gender_label = st.radio(
        "Show outfits for",
        list(GENDER_OPTIONS),
        horizontal=True,
    )
gender = GENDER_OPTIONS[gender_label]

if (
    st.session_state.get("rating_gender") != gender
    or "outfit_indices" not in st.session_state
):
    _start_rating_session(outfit_index, gender)

with controls[1]:
    if st.button("↻ New set", width="stretch"):
        _start_rating_session(outfit_index, gender)
        st.rerun()

with controls[2]:
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
st.subheader("Your 15 outfits")
st.caption("Every outfit needs one rating before recommendations can be generated.")

ratings: list[str | None] = []
card_columns = st.columns(3)
for card_number, (position, row) in enumerate(selected_outfits.iterrows(), start=1):
    with card_columns[(card_number - 1) % 3]:
        st.image(
            str(resolve_data_path(row["image_path"], APP_DIR)),
            caption=f"{card_number}. {row['category']}",
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
st.progress(rated_count / OUTFIT_COUNT, text=f"{rated_count} of {OUTFIT_COUNT} rated")

current_signature = (tuple(ratings), float(lambda_negative), gender)
if st.session_state.get("recommendation_signature") != current_signature:
    st.session_state.pop("recommendations", None)

generate = st.button(
    "Show my recommendations",
    type="primary",
    width="stretch",
    disabled=rated_count != OUTFIT_COUNT,
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
    except RecommendationError as error:
        st.warning(str(error))
    else:
        st.session_state.recommendations = recommendations
        st.session_state.recommendation_signature = current_signature

if "recommendations" in st.session_state:
    recommendations = st.session_state.recommendations
    st.divider()
    st.subheader("Your closest matches")
    st.caption(
        "Ranked by cosine similarity to your normalized FashionCLIP preference vector."
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
                f"{row['articleType']} · {row['baseColour']}  \n"
                f"Similarity: {row['similarity']:.3f}"
            )

st.divider()
st.caption(
    "Precomputed embeddings: base patrickjohncyh/fashion-clip model · "
    "Recommendations: adult apparel matching the selected gender"
)

