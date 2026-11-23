# Research overview — image retrieval through linear algebra

## Research question

How can matrix representations of images support pattern search and compact image representations, and how do classical techniques compare conceptually with modern feature-learning methods?

## Original research scope

The provided research proposal and final report examine image representation, filtering, edge detection, local-feature descriptors (SIFT and SURF), template/region matching, PCA and SVD, common similarity measures, and modern approaches such as CNNs, Siamese networks, and transfer learning.

These topics are part of a **literature-oriented review**; they are not all implemented in this repository.

## Implemented experiments

1. **SSD template search:** select a patch from a synthetic 8-by-8 tiled face scene and search for the closest patch after adding Gaussian noise to the template. The original experiment is illustrative; the result can vary because its random seed was not fixed.
2. **SVD reconstruction:** truncate a single grayscale face image to ranks 5 and 15 and visualize the loss of image detail. This demonstrates low-rank approximation, not a compression file format.
3. **PCA face-image retrieval:** fit PCA to the Olivetti face dataset and compare a selected face's projected vector to other image vectors. The original code excludes the query image from candidates, but does not hold out the query before fitting PCA. The new `src/` extension addresses this particular evaluation leakage.
4. **Custom template search:** use OpenCV normalized cross-correlation on a supplied scene and template image. The result is one best-scoring image location, not an object detection model trained on classes.

## Core equations

### Template matching (sum of squared differences)

```math
\operatorname{SSD}(x,y) = \sum_{i,j}(I_{x,y}(i,j)-T(i,j))^2
```

### SVD and best rank-k approximation

```math
A=U\Sigma V^T,\qquad A_k = \sum_{i=1}^{k}\sigma_i u_i v_i^T
```

### PCA projection

```math
z=W_k^T(x-\mu)
```

## Further evaluation

Useful next steps include repeated train/test splits, top-k retrieval, runtime profiling, noise-level sweeps, and comparisons against the same held-out queries. Image similarity alone should not be interpreted as biometric identity verification.

## Source attribution

This overview was newly written from the original student research proposal, research manuscript, practical report, and source code supplied with the project. Original literature citations should be reviewed and transcribed from the source reports before claiming a verified bibliography in English.
