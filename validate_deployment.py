"""Validate all packaged assets before pushing the folder to GitHub."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from index_store import load_embedding_index, resolve_data_path
from recommender import aggregate_preference, rank_candidates


APP_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_checksums() -> None:
    info = json.loads((APP_DIR / "DEPLOYMENT_INFO.json").read_text(encoding="utf-8"))
    for filename, expected in info["artifact_sha256"].items():
        path = APP_DIR / "artifacts" / filename
        if sha256(path) != expected:
            raise RuntimeError(f"Artifact checksum mismatch: {filename}")


def validate_index(name: str) -> None:
    index = load_embedding_index(name, APP_DIR)
    norms = np.linalg.norm(index.embeddings, axis=1)
    if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=2e-5):
        raise RuntimeError(f"{name} embeddings are not unit-normalized.")
    if not index.manifest["image_path"].is_unique:
        raise RuntimeError(f"{name} manifest contains duplicate image paths.")

    for number, image_path in enumerate(index.manifest["image_path"], start=1):
        path = resolve_data_path(image_path, APP_DIR)
        if path.suffix.lower() != ".webp":
            raise RuntimeError(f"Uncompressed image found: {image_path}")
        with Image.open(path) as image:
            if max(image.size) > 640:
                raise RuntimeError(f"Oversized deployment image: {image_path}")
            image.verify()
        if number % 1000 == 0 or number == len(index.manifest):
            print(f"\r{name}: verified {number:,}/{len(index.manifest):,} images", end="")
    print()


def main() -> None:
    validate_artifact_checksums()
    validate_index("clothing")
    validate_index("outfits")

    clothing = load_embedding_index("clothing", APP_DIR)
    outfits = load_embedding_index("outfits", APP_DIR)
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
        raise RuntimeError("Recommendation smoke test did not return ten results.")
    print("Deployment validation passed.")


if __name__ == "__main__":
    main()
