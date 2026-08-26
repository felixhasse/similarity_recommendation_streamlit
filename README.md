# FashionCLIP Style Compass — Streamlit deployment

This folder is a self-contained deployment repository. It includes:

- Precomputed, L2-normalized base FashionCLIP embeddings.
- Only the clothing and representative outfit images referenced by the manifests.
- WebP display images with a maximum dimension of 640 pixels.
- A model-free Streamlit runtime: PyTorch and Transformers are not installed or
  loaded on the server.

The embeddings were generated with the base
`patrickjohncyh/fashion-clip` checkpoint. No fine-tuned classifier weights are
used.

## Validate locally

Use Python 3.11. From this folder:

```bash
python -m pip install -r requirements.txt
python validate_deployment.py
streamlit run app.py
```

`validate_deployment.py` verifies every compressed image, the manifest/index
alignment, embedding norms, and a ten-result recommendation smoke test.

## Push to GitHub

This package is intentionally below GitHub's 100 MiB per-file limit and does
not require Git LFS. Create an empty GitHub repository, then run from this
folder:

```bash
git init
git add .
git commit -m "Deploy FashionCLIP recommendation app"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
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

The app memory-maps the two embedding matrices and serves only the images shown
for the active session. It does not download the model or calculate embeddings
in the cloud.

## Important

Before publishing the repository or app, confirm that the source image dataset
license permits the intended public redistribution and use.

