# Research and methodology | Linear algebra for image retrieval

## Research motivation

The original student research report addresses the problem of finding a particular image or visual pattern among a collection of images. It surveys classical image representation, preprocessing, local-feature detectors, region matching, linear-algebra methods, and modern image-retrieval approaches. The **research survey** and the **implemented experiments** are deliberately distinguished in this repository.

## 1. Problem formulation

Given a query image or patch `Q` and a set of candidate images `{I_1, ..., I_N}`, a retrieval system scores candidate similarity to `Q` and returns one or more high-ranked matches. Image location search is a related but different problem: sliding a smaller query template across one larger image to identify the best aligned patch.

## 2. Mathematical methods directly demonstrated

### Template matching

The sum of squared differences (SSD) compares intensity values at corresponding pixels:

```math
\mathrm{SSD}(x,y)=\sum_{i=1}^{h}\sum_{j=1}^{w}\left[I(x+i,y+j)-T(i,j)\right]^2
```

The minimum SSD indicates the closest match under this specific pixel metric. Normalized cross-correlation provides a second, intensity-normalized measure. Neither method is inherently robust to large geometric transformations.

### Singular value decomposition

For a grayscale image matrix `A`:

```math
A=U\Sigma V^T, \qquad A_k=\sum_{j=1}^{k}\sigma_j u_j v_j^T
```

A truncated rank-`k` matrix is the best rank-`k` approximation in Frobenius norm (Eckart–Young–Mirsky theorem). The experiment measures reconstruction error, not actual file-size savings from a codec.

### Principal component analysis

For training images vectorized as rows, PCA estimates the training-set mean and high-variance directions. A mean-centered query is projected into the learned feature basis:

```math
z=W_k^T(x-\mu)
```

Nearest-neighbor retrieval in this space is a simple illustration of *known-identity* face image search. Training/test separation is needed to prevent query-data leakage. Retrieval among known identities is not verification of new or unknown people's identities.

## 3. Literature-review coverage (not implemented here)

The original survey additionally discusses SIFT and SURF descriptors, edge detection and filtering, image similarity criteria, convolutional neural networks, transfer learning, Siamese architectures and other modern retrieval techniques. These subjects are documented as background context only, **not** features offered by this codebase.

## 4. Experimental design and interpretation

The cleaned code includes SSD/correlation patch search, rank-limited SVD reconstruction and PCA nearest-neighbor face-image retrieval on Olivetti. The new offline appendix varies input-template noise and rank `k` for the provided sample images. A robust comparative study would additionally use labeled ground-truth matches across independent images, repeated seeds, runtime profiling and dataset-level retrieval metrics.

## 5. Research limitations

The student research is a course literature survey rather than a peer-reviewed publication. Example screenshots and new illustrative experiments do not show a new state-of-the-art method. No external claim of improvement over literature baselines is made without comparative evidence.

## 6. Research integrity

The supplied original documents are in Persian. This English synthesis is an independent portfolio presentation of their research structure and methods, not a line-by-line English translation. Bibliographic references from the original report should be carefully checked before reproducing them in a formally referenced manuscript.
