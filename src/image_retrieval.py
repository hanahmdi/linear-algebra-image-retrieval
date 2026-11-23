"""Linear-algebra image retrieval experiments: SSD, SVD and PCA."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def find_template(scene: np.ndarray, template: np.ndarray, method: str = 'ssd'):
    """Return upper-left pixel location, score, and the matching score map.

    ssd: smaller is better, normalized correlation: larger is better.
    """
    if scene.ndim != 2 or template.ndim != 2:
        raise ValueError('Both inputs must be 2-D grayscale images')
    h, w = template.shape
    if h > scene.shape[0] or w > scene.shape[1]:
        raise ValueError('Template must fit within scene')
    if method == 'ssd':
        op = cv2.TM_SQDIFF
    elif method == 'correlation':
        if np.std(template) < 1e-6:
            raise ValueError('Normalized correlation requires a non-constant template')
        op = cv2.TM_CCOEFF_NORMED
    else:
        raise ValueError('Method must be ssd or correlation')
    scores = cv2.matchTemplate(scene.astype('float32'), template.astype('float32'), op)
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(scores)
    loc, score = (minloc, minval) if method == 'ssd' else (maxloc, maxval)
    return loc, float(score), scores


def compress_svd(image: np.ndarray, rank: int):
    """Compute rank-k reconstruction and relative Frobenius error."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError('Image must be grayscale')
    if not 1 <= rank <= min(image.shape):
        raise ValueError('Rank out of range')
    u, s, vt = np.linalg.svd(image, full_matrices=False)
    reconstruction = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    denom = np.linalg.norm(image)
    err = np.linalg.norm(image-reconstruction) / denom if denom else 0.0
    return reconstruction, float(err)


def pca_face_retrieval(test_size=0.25, components=50, seed=42):
    """Nearest-neighbor recognition on a reproducible image-disjoint split.

    Olivetti dataset may require Internet access on the first invocation.
    This is image-disjoint but identity-overlapping, not unseen-person testing.
    """
    ds = fetch_olivetti_faces(shuffle=False)
    x, y = ds.data, ds.target
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=test_size,
                                          stratify=y, random_state=seed)
    if components >= len(train_idx):
        raise ValueError('PCA components must be fewer than training images')
    pca = PCA(n_components=components, svd_solver='randomized', random_state=seed)
    train_z = pca.fit_transform(x[train_idx])
    test_z = pca.transform(x[test_idx])
    # Euclidean nearest neighbor in PCA coordinates; no query leakage.
    d2 = ((test_z[:, None, :] - train_z[None, :, :])**2).sum(axis=2)
    pred = y[train_idx][d2.argmin(axis=1)]
    return {'accuracy': float(accuracy_score(y[test_idx], pred)),
            'n_train': len(train_idx), 'n_test': len(test_idx),
            'n_components': components, 'seed': seed}


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    m=sub.add_parser('match');m.add_argument('--scene',required=True);m.add_argument('--template',required=True)
    m.add_argument('--method',choices=['ssd','correlation'],default='ssd');m.add_argument('--output',default='outputs/template_result.png')
    c=sub.add_parser('compress');c.add_argument('--image',required=True);c.add_argument('--rank',type=int,default=15)
    c.add_argument('--output',default='outputs/svd_reconstruction.png')
    p=sub.add_parser('pca');p.add_argument('--components',type=int,default=50);p.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.command == 'pca':
        print(pca_face_retrieval(components=args.components,seed=args.seed));return
    image=cv2.imread(args.scene if args.command=='match' else args.image,cv2.IMREAD_GRAYSCALE)
    if image is None: parser.error('Input image could not be read')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='compress':
        arr,err=compress_svd(image,args.rank)
        cv2.imwrite(str(output),np.clip(arr,0,255).astype('uint8'))
        print(f'Relative Frobenius error: {err:.6f}')
    else:
        template=cv2.imread(args.template,cv2.IMREAD_GRAYSCALE)
        if template is None: parser.error('Template image could not be read')
        (x,y),score,_=find_template(image,template,args.method)
        h,w=template.shape
        result=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imwrite(str(output),result)
        print(f'Match location: ({x}, {y}); {args.method} score: {score:.6f}')
    print(f'Saved: {output}')

if __name__=='__main__':cli()
