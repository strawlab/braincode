#!/usr/bin/env python
from __future__ import print_function

# Modified from https://raw.githubusercontent.com/salspaugh/machine_learning/
# 1e6235350f61ebdaab6e56268e34d43f079e68ee/clustering/kmedoids.py
from itertools import combinations

import numpy as np


def do_checks(D, slow=False):
    # make sure distance matrix is reasonable
    n = D.shape[0]
    assert D.shape == (n, n)
    assert np.allclose(np.diag(D), np.zeros((n,)))
    if slow:
        for i, j in combinations(range(n), 2):
            assert D[i, j] == D[j, i]
    else:
        for i in range(0, n, n // 5):
            for j in range(0, n, n // 5):
                if j > i:
                    continue
            assert D[i, j] == D[j, i]


def cluster(distances, k=3, seed=0, centroids=None):
    m = distances.shape[0]  # number of points
    do_checks(distances)

    assert m > k

    # Pick k random medoids
    if centroids is not None:
        print('Using precomputed medoids')
        curr_medoids = centroids
    else:
        print('Using random medoids')
        rng = np.random.RandomState(seed)
        curr_medoids = rng.choice(m, size=k, replace=False)
    print('Initial medoids: ', curr_medoids)

    old_medoids = np.array([-1] * k)  # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1] * k)

    clusters = None

    # Until the medoids stop updating, do the following:
    niters = 0
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        niters += 1
    print('k-medoids converged after %d iterations' % niters)
    return clusters, curr_medoids


def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:, medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids  # prevent medoid from being lost
    return clusters


def compute_new_medoid(cluster, distances):
    cluster_distances = distances[np.ix_(cluster, cluster)]
    costs = cluster_distances.sum(axis=1)
    lowidx = np.argmin(costs)
    return cluster[lowidx]
