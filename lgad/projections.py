import torch
from torch import Tensor
from torch.nn import functional as F


__all__ = [
    "inner_projection",
    "orthogonal_projection",
]


def inner_projection(
    features: Tensor,  # [batch_size, feature_size]
    vectors: Tensor,   # [feature_size] or [num_vectors, feature_size]
    *,
    basis: bool = False,
) -> Tensor:           # [batch_size, feature_size]
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(dim=0)
        scales = torch.inner(features, vectors) / vectors.square().sum(dim=1)
        return scales @ vectors
    else:
        vector_basis = vectors if basis else torch.linalg.svd(vectors, full_matrices=False)[2]
        return (features @ vector_basis.T) @ vector_basis


def orthogonal_projection(
    features: Tensor,  # [batch_size, feature_size]
    vectors: Tensor,   # [feature_size] or [num_vectors, feature_size]
    *,
    normalize: bool = False,
    basis: bool = False,
) -> Tensor:           # [batch_size, feature_size]
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(dim=0)

    vector_basis = vectors if basis else torch.linalg.svd(vectors, full_matrices=False)[2]
    proj = features - (features @ vector_basis.T) @ vector_basis

    if normalize:
        proj = F.normalize(proj, dim=-1)

    return proj
