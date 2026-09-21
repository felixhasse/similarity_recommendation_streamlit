# Fashion embedding Style Compass — Streamlit deployment

This folder is a self-contained deployment repository. It includes:

- Precomputed, L2-normalized FashionCLIP and Marqo FashionSigLIP embeddings.
- `Ladies`, `Men`, and `Unisex` front-view recommendation candidates from the training split of
  [`chibifire/zenodo-second-hand-fashion-v3`](https://huggingface.co/datasets/chibifire/zenodo-second-hand-fashion-v3).
- Only the clothing and representative outfit images referenced by the manifests.
- WebP display images with a maximum dimension of 512 pixels.
- A model-free Streamlit runtime: PyTorch and Transformers are not installed or
  loaded on the server.

The user can switch between the base `patrickjohncyh/fashion-clip` checkpoint
and `Marqo/marqo-fashionSigLIP`, pinned to revision
`c56244cc94f92419e8369fa71efdaf403b124ce8`. No fine-tuned FashionCLIP
classifier weights are used.

The second-hand clothing snapshot is pinned to commit
`d32b983103be67af13365dbfcc9db41faa9aadab`. `Ladies` items map to Feminine,
`Men` items map to Masculine, and the Both option searches those categories plus
`Unisex`; children and the publisher's blinded test split are excluded. Users
rate between 5 and 30 outfits, then receive ten overall matches and up to five
matches for every available clothing type. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and the
CC BY 4.0 terms.

Clothing-type grouping is insensitive to capitalization, whitespace, and
punctuation. Known source spelling variants are mapped to stable labels; for
example, `Night gown` becomes `Nightgown` and `Jacker` becomes `Jacket`.

## Validate locally

Use Python 3.11. From this folder:

```bash
python -m pip install -r requirements.txt
python validate_deployment.py
streamlit run app.py
```

`validate_deployment.py` verifies every compressed image, all four embedding
matrices, manifest/index alignment, embedding norms, and a ten-result
recommendation smoke test for each model.

## Push to GitHub

This package is intentionally below GitHub's 100 MiB per-file limit and does
not require Git LFS. The folder is already initialized as a Git repository on
the `main` branch with the deployment contents committed. Its `origin` points
to `git@github.com:felixhasse/similarity_recommendation_streamlit.git`.

Create that empty GitHub repository without adding a README or license if it
does not exist yet, verify the configured destination, and push:

```bash
git remote -v
git push -u origin main
```

To use another repository instead, first run:

```bash
git remote set-url origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
```

Avoid repeatedly replacing the binary assets in later commits because Git
retains previous versions in repository history.

## Deploy on Streamlit Community Cloud

1. Open <https://share.streamlit.io> and connect the GitHub account that owns
   the repository.
2. Select **Create app** and choose the repository and `main` branch.
3. Set the entrypoint to `app.py`.
4. In **Advanced settings**, select Python **3.11**. No secrets are required.
5. Select **Deploy**.

The app memory-maps the selected model's two embedding matrices and serves only
the images shown for the active session. It does not download either model or
calculate embeddings in the cloud.

## Data attribution

The app footer links to the source dataset and its CC BY 4.0 license. Keep the
footer and `THIRD_PARTY_NOTICES.md` with redistributed or deployed copies.
