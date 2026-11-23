# Recorded supplementary analyses

The CSV tables contain the numerical values reported in `docs/ACADEMIC_REPORT.md` and `docs/EXPERIMENT_RESULTS.md`. Results are associated with one supplied scene/template pair and a resized SVD sample image, not a benchmark dataset.

- `template_noise_results.csv`: Gaussian template noise experiment, seed 42; SSD and normalized correlation use different numerical scales.
- `svd_rank_results.csv`: relative Frobenius reconstruction error for four retained ranks.

No numerical PCA dataset accuracy is reported because the dataset-backed code was not verified end to end.
