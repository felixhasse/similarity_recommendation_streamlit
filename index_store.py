"""Load and validate the precomputed deployment indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommender import EmbeddingIndex


MODEL_ID = "patrickjohncyh/fashion-clip"
FORMAT_VERSION = 1
INDEX_NAMES = ("clothing", "outfits")


class DeploymentDataError(RuntimeError):
    """Raised when packaged deployment data is missing or incompatible."""


def resolve_data_path(path: str | Path, app_dir: Path) -> Path:
    candidate = (app_dir / Path(path)).resolve()
    try:
        candidate.relative_to(app_dir.resolve())
    except ValueError as error:
        raise DeploymentDataError(f"Image path escapes the app directory: {path}") from error
    if not candidate.is_file():
        raise DeploymentDataError(f"Image is missing: {path}")
    return candidate


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DeploymentDataError(f"Metadata is missing: {path.name}") from error
    except json.JSONDecodeError as error:
        raise DeploymentDataError(f"Metadata is invalid: {path.name}") from error


def load_embedding_index(name: str, app_dir: Path) -> EmbeddingIndex:
    if name not in INDEX_NAMES:
        raise DeploymentDataError(f"Unknown index: {name}")
    artifact_dir = app_dir / "artifacts"
    embeddings_path = artifact_dir / f"{name}_embeddings.npy"
    manifest_path = artifact_dir / f"{name}_manifest.csv"
    metadata_path = artifact_dir / f"{name}_metadata.json"

    metadata = _read_metadata(metadata_path)
    if (
        metadata.get("format_version") != FORMAT_VERSION
        or metadata.get("model_id") != MODEL_ID
        or metadata.get("name") != name
        or metadata.get("normalized") is not True
    ):
        raise DeploymentDataError(f"{name} metadata is incompatible.")

    try:
        embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise DeploymentDataError(f"Could not load {embeddings_path.name}.") from error
    try:
        manifest = pd.read_csv(manifest_path, dtype={"id": str, "variant": str})
    except (FileNotFoundError, OSError, ValueError) as error:
        raise DeploymentDataError(f"Could not load {manifest_path.name}.") from error

    expected_shape = (
        metadata.get("row_count"),
        metadata.get("embedding_dimension"),
    )
    if embeddings.ndim != 2 or embeddings.shape != expected_shape:
        raise DeploymentDataError(
            f"{name} embedding shape {embeddings.shape} does not match {expected_shape}."
        )
    if len(manifest) != embeddings.shape[0]:
        raise DeploymentDataError(f"{name} manifest and embeddings are not aligned.")
    if "image_path" not in manifest or "gender" not in manifest:
        raise DeploymentDataError(f"{name} manifest is missing required columns.")
    return EmbeddingIndex(manifest=manifest, embeddings=embeddings)

