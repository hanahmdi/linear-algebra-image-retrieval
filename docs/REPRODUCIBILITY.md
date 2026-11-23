# Reproducibility guide

## Environment

Use an isolated Python environment, then install `requirements.txt`. The package list is a compatibility range, not a frozen lockfile. Record `python --version` and package versions when reproducing results.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Supplied-image experiment

```bash
python src/image_retrieval.py match --scene data/sample_scene.png --template data/sample_template.png --method correlation --output outputs/match.png
python src/image_retrieval.py compress --image data/sample_scene.png --rank 15 --output outputs/svd_rank15.png
```

## PCA evaluation

```bash
python src/image_retrieval.py pca --components 50 --seed 42
```

Olivetti faces are fetched on first run; internet/cache access may be required. The current implementation uses a stratified image-disjoint split with the same identities represented in training and test sets. Report the seed, train/test counts, PCA components and measured accuracy for every run. Do not interpret this as unseen-identity recognition.

## Offline figures and tables

The rank-sweep and noise measurements in `docs/EXPERIMENT_RESULTS.md` were generated with a fixed random seed on the user-supplied image pair. SVD plots use an image resized to a maximum dimension of 300 pixels to make comparison economical. These are additional analyses, separately labeled from the historical classroom trials.

## Data rights

Before publishing a public repository, confirm permission to redistribute the sample imagery. Otherwise remove `data/sample_scene.png`, `data/sample_template.png`, and derived previews, and substitute self-produced or openly licensed samples.
