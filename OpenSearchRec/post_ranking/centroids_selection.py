import numpy as np
from typing import Optional


def select_items_greedily_and_with_similarity_suppression(
        items_weights: np.ndarray,
        items_similarity_matrix: np.ndarray,
        suppress_supression=False,
        run_input_validation=True):

    N = items_weights.shape[0]

    if run_input_validation:
        assert type(items_weights) == np.ndarray
        assert type(items_similarity_matrix) == np.ndarray
        assert items_similarity_matrix.shape == (N, N)
        assert items_weights.min() >= 0
        assert items_similarity_matrix.min() >= 0
        assert items_similarity_matrix.max() <= 1
        assert (items_similarity_matrix == items_similarity_matrix.T).all()

    items_weights = items_weights.copy()
    if suppress_supression:
        items_weight_suppression_multiplier = np.ones(N)

    idx_list = []
    for _ in range(N):
        top_idx = np.argmax(items_weights)
        idx_list.append(top_idx)

        if suppress_supression:
            items_weights[top_idx] = 0
            items_weights = \
                items_weights * (-items_similarity_matrix[top_idx] * items_weight_suppression_multiplier[top_idx] + 1)

            items_weight_suppression_multiplier = \
                items_weight_suppression_multiplier * (- items_similarity_matrix[top_idx] + 1)

        else:
            items_weights = items_weights * (- items_similarity_matrix[top_idx] + 1)

    return idx_list


def centroids_score(centroid_idx_list, items_weights, items_similarity_matrix):
    return np.dot(
        np.max(items_similarity_matrix[:, centroid_idx_list], axis=1),
        items_weights
    )


def select_top_k_centroids(
        num_centroids,
        items_similarity_matrix: np.ndarray,
        items_weights: Optional[np.ndarray] = None,
        run_input_validation=True,
        max_iteration_number=10):

    N = items_similarity_matrix.shape[0]
    num_centroids = min(num_centroids, N)
    if items_weights is None:
        items_weights = np.ones(N)

    if run_input_validation:
        assert type(items_weights) == np.ndarray
        assert type(items_similarity_matrix) == np.ndarray
        assert items_similarity_matrix.shape == (N, N)
        assert items_weights.min() >= 0
        assert items_similarity_matrix.min() >= 0
        assert items_similarity_matrix.max() <= 1
        assert (items_similarity_matrix == items_similarity_matrix.T).all()

    centroid_idx_list = [i for i in range(num_centroids)]

    # Geedily select centroids
    for i in range(num_centroids):
        if len(centroid_idx_list) == 0:
            best_score = 0
        else:
            best_score = centroids_score(centroid_idx_list, items_weights, items_similarity_matrix)
        new_centroid_idx = None

        for cidx in range(N):
            if cidx not in centroid_idx_list:
                proposed_centroid_idx_list = centroid_idx_list + [cidx]
                score = centroids_score(proposed_centroid_idx_list, items_weights, items_similarity_matrix)
                if score > best_score:
                    best_score = score
                    new_centroid_idx = cidx

        if new_centroid_idx is not None:
            centroid_idx_list[i] = new_centroid_idx

    # Try to swap every centroid to the remaining items and keep swap if score improves
    score_improved = True
    iteration_number = 0
    while score_improved and iteration_number < max_iteration_number:
        score_improved = False
        best_score = centroids_score(centroid_idx_list, items_weights, items_similarity_matrix)
        for i in range(num_centroids):
            for cidx in range(N):
                proposed_centroid_idx_list = centroid_idx_list[:]
                if cidx not in centroid_idx_list:
                    proposed_centroid_idx_list[i] = cidx
                    score = centroids_score(proposed_centroid_idx_list, items_weights, items_similarity_matrix)
                    if score > best_score:
                        best_score = score
                        centroid_idx_list[i] = cidx
                        score_improved = True

        iteration_number += 1

    centroid_scores_list = []
    for i, c_idx in enumerate(centroid_idx_list):
        centroid_idx_list_wo_cidx = [c for c in centroid_idx_list if c != c_idx]
        delta_score = \
            centroids_score(centroid_idx_list, items_weights, items_similarity_matrix) - centroids_score(centroid_idx_list_wo_cidx, items_weights, items_similarity_matrix)
        centroid_scores_list.append(delta_score)

    scores_idx = list(zip(centroid_scores_list, centroid_idx_list))
    scores_idx = sorted(scores_idx, reverse=True)

    sorted_centroid_scores_list = [e[0] for e in scores_idx]
    sorted_centroid_idx_list = [e[1] for e in scores_idx]

    return {
        "centroid_idx_list": centroid_idx_list,
        "centroid_scores_list": centroid_scores_list,
        "sorted_centroid_idx_list": sorted_centroid_idx_list,
        "sorted_centroid_scores_list": sorted_centroid_scores_list,
        "total_score": centroids_score(centroid_idx_list, items_weights, items_similarity_matrix)
    }
