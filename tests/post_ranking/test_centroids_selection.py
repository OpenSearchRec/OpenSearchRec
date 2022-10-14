import numpy as np

from OpenSearchRec.post_ranking import (
    select_items_greedily_and_with_similarity_suppression,
    select_top_k_centroids
)


def test_select_items_greedily_and_with_similarity_suppression():
    items_weights = np.array([3, 2, 1])
    items_similarity_matrix = \
        np.array([
            [1,   0.1, 0.1],
            [0.1, 1,   0.1],
            [0.1, 0.1, 1  ]
        ])

    idx_list = select_items_greedily_and_with_similarity_suppression(items_weights, items_similarity_matrix)

    assert idx_list == [0, 1, 2]

    items_weights = np.array([3, 2, 1])
    items_similarity_matrix = \
        np.array([
            [1,   0.9, 0.1],
            [0.9, 1,   0.1],
            [0.1, 0.1, 1  ]
        ])

    idx_list = select_items_greedily_and_with_similarity_suppression(items_weights, items_similarity_matrix)

    assert idx_list == [0, 2, 1]

    items_weights = np.array([10, 10, 10, 2])
    items_similarity_matrix = \
        np.array([
            [1,   0.9, 0.9, 0.1],
            [0.9, 1,   0.9, 0.1],
            [0.9, 0.9, 1,   0.1],
            [0.1, 0.1, 0.1, 1  ]
        ])

    idx_list = \
        select_items_greedily_and_with_similarity_suppression(
            items_weights, items_similarity_matrix, suppress_supression=True)

    assert idx_list == [0, 3, 1, 2]

    print("\n\n-------------\n\n")
    items_weights = np.array([10, 10, 10, 1.6])
    items_similarity_matrix = \
        np.array([
            [1,    0.8, 0.8, 0],
            [0.8,  1,   0.8, 0],
            [0.8,  0.8, 1,   0],
            [0,    0,   0,   1]
        ])

    idx_list = \
        select_items_greedily_and_with_similarity_suppression(
            items_weights, items_similarity_matrix, suppress_supression=False)

    assert idx_list == [0, 1, 3, 2]

    print("\n\n-------------\n\n")
    items_weights = np.array([10, 10, 10, 1.6])
    items_similarity_matrix = \
        np.array([
            [1,    0.8, 0.8, 0],
            [0.8,  1,   0.8, 0],
            [0.8,  0.8, 1,   0],
            [0,    0,   0,   1]
        ])

    idx_list = \
        select_items_greedily_and_with_similarity_suppression(
            items_weights, items_similarity_matrix, suppress_supression=True)

    assert idx_list == [0, 1, 2, 3]


def test_select_top_k_centroids():
    num_centroids = 2
    items_similarity_matrix = \
        np.array([
            [1,   0.1, 0.1],
            [0.1, 1,   0.1],
            [0.1, 0.1, 1  ]
        ])
    items_weights = np.array([3, 2, 1])

    idx_list = \
        select_top_k_centroids(num_centroids, items_similarity_matrix, items_weights)

    assert set(idx_list["centroid_idx_list"]) == set([0, 1])

    num_centroids = 2
    items_similarity_matrix = \
        np.array([
            [1,   0.1, 0.1],
            [0.1, 1,   0.1],
            [0.1, 0.1, 1  ]
        ])
    items_weights = np.array([1, 2, 3])

    idx_list = \
        select_top_k_centroids(num_centroids, items_similarity_matrix, items_weights)

    assert idx_list["centroid_idx_list"] == [2, 1]

    num_centroids = 2
    items_weights = np.array([1, 1, 1, 1])
    items_similarity_matrix = \
        np.array([
            [1,    0.9, 0.8,   0],
            [0.9,  1,   0.8,   0],
            [0.8,  0.8, 1,     0.2],
            [0,    0,   0.2,   1]
        ])

    r1 = \
        select_top_k_centroids(num_centroids, items_similarity_matrix, items_weights, max_iteration_number=0)

    r2 = \
        select_top_k_centroids(num_centroids, items_similarity_matrix, items_weights, max_iteration_number=10)

    print("r1", r1)
    print("r2", r2)

    assert r1["total_score"] < r2["total_score"]
