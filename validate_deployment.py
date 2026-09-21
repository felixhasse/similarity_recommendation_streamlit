"""Validate all packaged assets before pushing the folder to GitHub."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from index_store import MODEL_SPECS, load_embedding_index, resolve_data_path
from recommender import (
    aggregate_preference,
    canonicalize_clothing_type,
    rank_candidates,
    rank_candidates_by_type,
)


APP_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_checksums(info: dict[str, object]) -> None:
    for filename, expected in info["artifact_sha256"].items():
        path = APP_DIR / "artifacts" / filename
        if sha256(path) != expected:
            raise RuntimeError(f"Artifact checksum mismatch: {filename}")


def validate_index(
    name: str,
    max_dimension: int,
    model_key: str,
    validate_images: bool,
) -> None:
    index = load_embedding_index(name, APP_DIR, model_key)
    norms = np.linalg.norm(index.embeddings, axis=1)
    if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=2e-5):
        raise RuntimeError(f"{model_key} {name} embeddings are not unit-normalized.")
    if not index.manifest["image_path"].is_unique:
        raise RuntimeError(f"{name} manifest contains duplicate image paths.")

    if not validate_images:
        return
    for number, image_path in enumerate(index.manifest["image_path"], start=1):
        path = resolve_data_path(image_path, APP_DIR)
        if path.suffix.lower() != ".webp":
            raise RuntimeError(f"Uncompressed image found: {image_path}")
        with Image.open(path) as image:
            if max(image.size) > max_dimension:
                raise RuntimeError(f"Oversized deployment image: {image_path}")
            image.verify()
        if number % 1000 == 0 or number == len(index.manifest):
            print(
                f"\r{name}: verified {number:,}/{len(index.manifest):,} images",
                end="",
            )
    print()


def main() -> None:
    info = json.loads((APP_DIR / "DEPLOYMENT_INFO.json").read_text(encoding="utf-8"))
    max_dimension = int(info["image_max_dimension"])
    validate_artifact_checksums(info)
    for model_number, model_key in enumerate(MODEL_SPECS):
        validate_index(
            "clothing",
            max_dimension,
            model_key,
            validate_images=model_number == 0,
        )
        validate_index(
            "outfits",
            max_dimension,
            model_key,
            validate_images=model_number == 0,
        )

        clothing = load_embedding_index("clothing", APP_DIR, model_key)
        outfits = load_embedding_index("outfits", APP_DIR, model_key)
        if "Unisex" not in set(clothing.manifest["gender"]):
            raise RuntimeError(f"{model_key} clothing index contains no Unisex items.")
        raw_types = clothing.manifest["type"].fillna("").astype(str)
        noncanonical = sorted(
            {
                item_type
                for item_type in raw_types
                if canonicalize_clothing_type(item_type) != item_type
            }
        )
        if noncanonical:
            raise RuntimeError(
                f"{model_key} manifest contains noncanonical types: {noncanonical}"
            )
        preference = aggregate_preference(
            np.asarray(outfits.embeddings[[0]]),
            np.asarray(outfits.embeddings[[1]]),
        )
        results = rank_candidates(
            preference,
            clothing,
            gender=str(outfits.manifest.iloc[0]["gender"]),
            top_k=10,
        )
        if len(results) != 10:
            raise RuntimeError(
                f"{model_key} recommendation smoke test did not return ten results."
            )
        combined_results = rank_candidates(
            preference,
            clothing,
            gender="Both",
            top_k=10,
        )
        if len(combined_results) != 10:
            raise RuntimeError(
                f"{model_key} combined recommendation smoke test failed."
            )
        typed_results = rank_candidates_by_type(
            preference,
            clothing,
            gender="Both",
            top_k=5,
        )
        if not typed_results or any(len(group) > 5 for group in typed_results.values()):
            raise RuntimeError(f"{model_key} per-type recommendation smoke test failed.")
    print("Deployment validation passed.")


if __name__ == "__main__":
    main()
