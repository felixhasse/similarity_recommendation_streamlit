# Third-party data notice

## Second-Hand Fashion Dataset (v3)

The clothing recommendation candidates are derived from
[`chibifire/zenodo-second-hand-fashion-v3`](https://huggingface.co/datasets/chibifire/zenodo-second-hand-fashion-v3),
a repack by K. S. Ernest Lee (iFire) of the Second-Hand Fashion Dataset by
Farrukh Nauman, RISE Research Institutes of Sweden AB, Wargön Innovation AB,
and Myrorna AB.

- Upstream record: <https://doi.org/10.5281/zenodo.13788681>
- Pinned Hugging Face revision: `d32b983103be67af13365dbfcc9db41faa9aadab`
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

Changes made for this app: only the training split is used; rows are limited to
the adult `Ladies` and `Men` categories; only each garment's front view is
retained; the images are resized and re-encoded as WebP; and normalized image
embeddings are derived with the base FashionCLIP model.
