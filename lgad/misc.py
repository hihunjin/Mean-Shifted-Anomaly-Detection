from typing import Callable, Union, Optional, List, Literal, Iterable, Tuple, overload, Sequence

import torch
from torch import Tensor, IntTensor
from torch.nn import functional as F

from sklearn.cluster import KMeans

from .models.base import VisionLanguageModel


__all__ = [
    "cosine_similarity_matrix",
    "cosine_distance_matrix",
    "cosine_similarity",
    "cosine_distance",
    "knn",
    "kmeans",
    "kmeans_tune",
    "align_vectors",
    "zeroshot_ad",
    "prompt_pair",
    "pca",
]


def cosine_similarity_matrix(
    x1: Tensor,
    x2: Optional[Tensor] = None,
    eps: float = 1e-8,
):
    x2 = x1 if x2 is None else x2
    x1_norm = x1 / torch.max(x1.norm(dim=-1, keepdim=True), eps * torch.ones_like(x1))
    x2_norm = x2 / torch.max(x2.norm(dim=-1, keepdim=True), eps * torch.ones_like(x2))
    sim_matrix = torch.mm(x1_norm, x2_norm.transpose(-1, -2))
    return sim_matrix


def cosine_distance_matrix(
    x1: Tensor,
    x2: Optional[Tensor] = None,
    eps: float = 1e-8,
):
    return 1 - cosine_similarity_matrix(x1, x2, eps=eps)


# Alias of cosine_similarity_matrix
def cosine_similarity(
    x1: Tensor,
    x2: Optional[Tensor] = None,
    eps: float = 1e-8,
):
    return cosine_similarity_matrix(x1, x2, eps=eps)


# Alias of cosine_distance_matrix
def cosine_distance(
    x1: Tensor,
    x2: Optional[Tensor] = None,
    eps: float = 1e-8,
):
    return 1 - cosine_similarity_matrix(x1, x2, eps=eps)


def knn(
    train_features: Tensor,
    test_features: Tensor,
    *,
    n_neighbors: int = 50,
    metric: Union[str, Callable] = "cosine",
    normalize: bool = False,
):
    if isinstance(metric, str):
        if metric == "cosine":
            metric_fn = cosine_distance_matrix
        elif metric == "euclidean":
            metric_fn = torch.cdist
        else:
            raise NotImplementedError(f"Unknown metric: {metric}")
    else:
        metric_fn = metric

    distance_matrix = metric_fn(test_features, train_features)
    distances, _ = distance_matrix.topk(n_neighbors, largest=False)
    scores = distances.mean(dim=1)

    if normalize:
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    return scores


def _sklearn_kmeans(
    vectors: Tensor,
    n_clusters: int,
    seed: int = 0,
    **kwargs,
) -> Tensor:

    vectors = F.normalize(vectors, dim=1)

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed, **kwargs)
    kmeans.fit(vectors.cpu())

    centers = torch.from_numpy(kmeans.cluster_centers_)
    centers = centers.to(device=vectors.device, dtype=vectors.dtype)
    return centers


def _torch_kmeans(
    vectors: Tensor,
    n_clusters: int,
    n_init: Union[int, Literal["auto"]] = "auto",
    iter_limit: int = 0,
    seed: int = 0,
    tol: float = 1e-4,
    **kwargs,
) -> Tensor:
    # https://github.com/subhadarship/kmeans_pytorch

    if n_init != "auto":
        raise NotImplementedError("n_init != 'auto' is not supported")

    dtype = vectors.dtype
    device = vectors.device
    vectors = vectors.float()
    vectors = F.normalize(vectors, dim=1)

    num_samples = vectors.size(0)
    rng = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randperm(num_samples, generator=rng, device=device)[:n_clusters]
    centers = torch.index_select(vectors, 0, indices)

    iteration = 0

    while True:
        dis = cosine_distance(vectors, centers)
        cluster_ids = torch.argmin(dis, dim=1)
        centers_pre = centers.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(cluster_ids == index).squeeze().to(device)
            selected = torch.index_select(vectors, 0, selected)

            if selected.size(0) == 0:
                selected = vectors[torch.randint(vectors.size(0), (1,))]

            centers[index] = selected.mean(dim=0)

        center_shift = (centers - centers_pre).square().sum(dim=1).sqrt().sum()

        iteration = iteration + 1

        if center_shift.square() < tol:
            break

        if iter_limit != 0 and iteration >= iter_limit:
            break

    return centers.type(dtype)


def kmeans(
    vectors: Tensor,
    n_clusters: int,
    seed: int = 0,
    impl: Literal["sklearn", "torch"] = "torch",
    **kwargs,
) -> Tensor:
    device = vectors.device

    if impl == "sklearn":
        centers = _sklearn_kmeans(vectors, n_clusters, seed=seed, **kwargs)
    elif impl == "torch":
        centers = _torch_kmeans(vectors, n_clusters, seed=seed, **kwargs)
    else:
        raise NotImplementedError(f"Unknown impl: {impl}")

    centers = centers.to(device=device)
    return centers


# https://github.com/KevinMusgrave/pytorch-adapt/blob/main/src/pytorch_adapt/layers/silhouette_score.py
def _silhouette_score(feats: Tensor, labels: Tensor) -> float:
    unique_labels = torch.unique(labels)
    num_samples = feats.size(0)

    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")

    scores = []

    for L in unique_labels:
        curr_cluster = feats[labels == L]
        num_elements = len(curr_cluster)

        if num_elements > 1:
            intra_cluster_dists = torch.cdist(curr_cluster, curr_cluster)
            mean_intra_dists = torch.sum(intra_cluster_dists, dim=1) / (num_elements - 1)  # minus 1 to exclude self distance
            dists_to_other_clusters = []

            for otherL in unique_labels:
                if otherL != L:
                    other_cluster = feats[labels == otherL]
                    inter_cluster_dists = torch.cdist(curr_cluster, other_cluster)
                    mean_inter_dists = torch.sum(inter_cluster_dists, dim=1) / len(other_cluster)
                    dists_to_other_clusters.append(mean_inter_dists)

            dists_to_other_clusters = torch.stack(dists_to_other_clusters, dim=1)
            min_dists, _ = torch.min(dists_to_other_clusters, dim=1)
            curr_scores = (min_dists - mean_intra_dists) / torch.maximum(min_dists, mean_intra_dists)

        else:
            curr_scores = torch.tensor([0], device=feats.device, dtype=feats.dtype)

        scores.append(curr_scores)

    scores = torch.cat(scores, dim=0)

    if len(scores) != num_samples:
        raise ValueError(f"scores (shape {scores.shape}) should have same length as feats (shape {feats.shape})")

    return torch.mean(scores).item()


def kmeans_tune(
    vectors: Tensor,
    n_clusters_list: Iterable[int],
    seed: int = 0,
    progress: bool = False,
    **kwargs,
) -> Tuple[Tensor, int]:
    norm_vectors = F.normalize(vectors, dim=1)
    norm_vectors_cpu = norm_vectors.cpu()

    n_clusters_list = [n_clusters for n_clusters in n_clusters_list if 2 <= n_clusters <= len(vectors)]

    best_score = -1.
    best_centers = None
    best_n_clusters = -1

    if progress:
        from tqdm.auto import tqdm
        n_clusters_list = tqdm(n_clusters_list)

    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed, **kwargs)
        labels = kmeans.fit_predict(norm_vectors_cpu)
        score = _silhouette_score(norm_vectors, torch.from_numpy(labels))

        if score > best_score:
            best_score = score
            best_centers = kmeans.cluster_centers_.copy()
            best_n_clusters = n_clusters

            if progress:
                print(f"best_n_clusters: {best_n_clusters}, best_score: {best_score:.4f}")

    centers = torch.from_numpy(best_centers)
    centers = centers.to(device=vectors.device, dtype=vectors.dtype)
    return centers, best_n_clusters


def align_vectors(vectors: torch.Tensor, reference_idx: int = 0):
    reference = vectors[reference_idx]
    sim = F.cosine_similarity(vectors, reference)
    aligned = torch.where((sim < 0).unsqueeze(dim=1), -vectors, vectors)
    return aligned


def zeroshot_ad(
    model: VisionLanguageModel,
    images: Tensor,  # images or image_features
    ind_texts: Union[IntTensor, List[IntTensor], List[str], List[List[str]]],                   # texts or text_features (only IND prompts)
    ood_texts: Optional[Union[IntTensor, List[IntTensor], List[str], List[List[str]]]] = None,  # texts or text_features (only OOD prompts)
    *,
    scoring: Literal["energy", "entropy", "var", "mcm", "sum"] = "mcm",
    temperature: Optional[float] = None,
):
    image_features = model.encode_image(images) if images.dim() == 4 else images
    ind_text_features = model.encode_text(ind_texts) if isinstance(ind_texts, list) else ind_texts

    if scoring not in {"energy", "entropy", "var", "mcm", "sum"}:
        raise ValueError(f"Unknown scoring: {scoring}")

    if scoring == "sum" and ood_texts is None:
        raise ValueError("scoring == 'sum' requires ood_texts")

    if ood_texts is None:
        ind_probs = model.similarity(image_features, ind_text_features, temperature=temperature)

        # https://github.com/deeplearning-wisc/MCM/blob/main/utils/detection_util.py
        if scoring == "energy":
            raise NotImplementedError("scoring == 'energy' is not implemented yet")
            # Energy = - T * logsumexp(logit_k / T), by default T = 1  (https://arxiv.org/abs/2010.03759)
            # scores = -(temperature * torch.logsumexp(logits / temperature, dim=1))
        elif scoring == "entropy":
            from scipy.stats import entropy
            scores = entropy(ind_probs.cpu().detach(), axis=1)
            scores = torch.from_numpy(scores).to(device=ind_probs.device)
        elif scoring == "var":
            scores = -torch.var(ind_probs, dim=1)
        elif scoring == "mcm":
            scores = -torch.amax(ind_probs, dim=1)
    else:
        ood_text_features = model.encode_text(ood_texts) if isinstance(ood_texts, list) else ood_texts
        text_features = torch.cat((ind_text_features, ood_text_features), dim=0)
        ind_ood_probs = model.similarity(image_features, text_features, temperature=temperature)
        ind_probs = ind_ood_probs[:, :ind_text_features.size(0)]
        ood_probs = ind_ood_probs[:, ind_text_features.size(0):]

        # https://github.com/deeplearning-wisc/MCM/blob/main/utils/detection_util.py
        if scoring == "energy":
            raise NotImplementedError("scoring == 'energy' is not implemented")
        elif scoring == "entropy":
            raise NotImplementedError("scoring == 'entropy' is not implemented")
        elif scoring == "var":
            raise NotImplementedError("scoring == 'var' is not implemented")
        if scoring == "mcm":
            scores = torch.amax(ood_probs, dim=1)
        elif scoring == "sum":
            scores = torch.sum(ood_probs, dim=1)

    return scores


@overload
def prompt_pair(prompts: Tensor, *, groups: Optional[Sequence[int]] = None):
    ...


@overload
def prompt_pair(*prompts_list: Tensor):
    ...


def prompt_pair(prompts: Tensor, *prompts_list: Tensor, groups: Optional[Sequence[int]] = None):
    if len(prompts_list) == 0:
        if groups is None:
            length = prompts.size(0)
            idxs = torch.tensor([i * length + j for i in range(length) for j in range(i + 1, length)]).to(prompts.device)
            pairwise_diff = (prompts.unsqueeze(1) - prompts.unsqueeze(0)).flatten(0, 1).index_select(0, idxs)
            pairwise_diff = align_vectors(pairwise_diff)
            return pairwise_diff
        else:
            _groups = torch.tensor(groups).to(prompts.device)
            pairwise_diff = prompt_pair(*[
                prompts.index_select(0, torch.nonzero(_groups == group).squeeze())
                for group in torch.unique(_groups)
            ])
            return pairwise_diff
    else:
        if groups is not None:
            raise ValueError("groups must be None if prompts_list is not empty")

        prompts_list = (prompts,) + prompts_list
        pairwise_diff = torch.cat([
            (prompts_list[i].unsqueeze(1) - prompts_list[j].unsqueeze(0)).flatten(0, 1)
            for i in range(len(prompts_list))
            for j in range(i + 1, len(prompts_list))
        ])
        pairwise_diff = align_vectors(pairwise_diff)
        return pairwise_diff


# https://github.com/gngdb/pytorch-pca

# def _svd_flip(u, v):
#     # columns of u, rows of v
#     max_abs_cols = torch.argmax(torch.abs(u), 0)
#     i = torch.arange(u.shape[1]).to(u.device)
#     signs = torch.sign(u[max_abs_cols, i])
#     u *= signs
#     v *= signs.view(-1, 1)
#     return u, v


# def pca(
#     vectors: Tensor,
#     n_components: Optional[int] = None,
#     skip_align: bool = False,
# ) -> Tensor:
#     d = None if n_components is None else min(n_components, vectors.size(1))
#     Z = vectors - vectors.mean(0, keepdim=True)  # center
#     U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
#     if not skip_align:
#         U, Vh = _svd_flip(U, Vh)
#     components = Vh if d is None else Vh[:d]
#     return components


def pca(
    vectors: Tensor,
    n_components: Optional[int] = None,
    center: bool = False,
    niter: int = 2,
) -> Tensor:
    min_d = min(vectors.size(0), vectors.size(1))
    d = min_d if n_components is None else min(n_components, min_d)
    components = torch.pca_lowrank(vectors, q=d, center=center, niter=niter)[2].T
    return components
