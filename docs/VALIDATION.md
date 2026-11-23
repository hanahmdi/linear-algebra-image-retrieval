# Validation and reproducibility

## Verification levels

- **Original evidence:** the supplied practical report describes template search, SVD reconstruction, and PCA/Olivetti experiments. The original scripts are preserved in `archive/`.
- **Automated checks:** `tests/test_math.py` verifies exact synthetic patch recovery, normalized correlation, and monotonic SVD reconstruction error. These are unit tests, not real-image accuracy benchmarks.
- **New numerical PCA evaluation:** `src/image_retrieval.py pca` uses an image-disjoint, stratified train/test split, training-only PCA fitting, and nearest-neighbor matching. Each identity is represented in both subsets; this measures retrieval among known people, not open-set recognition.
- **Unverified:** environment-specific dataset download, performance on new photographs, rotation/scale robustness, out-of-distribution generalization, and biometric safety/accuracy.

## Limitations in the original code

1. Original face-crowd SSD search uses a hand-written sliding window and no seeded RNG; runs are not deterministic.
2. Its clean patch begins at arbitrary pixel coordinates inside a collage, which can cross face-image tile boundaries. The selected target is therefore a patch, not necessarily one complete face.
3. Original PCA was fitted on all available images, including the query, then nearest neighbors were computed in that same set. This does not establish independent test-set performance.
4. The label `Identified as ID` in the original output denotes a retrieved image index, not an authenticated person identity.
5. The original cat/image experiment uses correlation score × 100 as a displayed percentage; this is not calibrated detection probability.

## Suggested next improvements

- Produce quantitative tables for multiple noise levels and different query transformations.
- Add ground-truth bounding boxes for multi-image matching benchmarks.
- Compare PCA dimensionalities on held-out data and report accuracy with the evaluation protocol.
- Check dataset licenses and citation requirements prior to public release.
