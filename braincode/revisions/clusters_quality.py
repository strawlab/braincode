# coding=utf-8
"""Explores some measures of cluster quality for clusterings and individual cluster members."""
import pandas as pd
from sklearn.metrics import silhouette_samples

from braincode.dice import dicedist_metric
from braincode.util import ExpressionDataset, get_original_clustering


def silhouette_original_clusterings(dataset='CB1', neuropil='Antennal_lobe', clusterer_or_k=60):
    """Returns a pandas dataframe with the silhouette index of each cluster member.
    The dataframe have columns (cluster_id, member_id, silhouette).
    """

    # Read the expression matrix
    print('Reading expression matrix')
    Xdf = ExpressionDataset.dataset(dset=dataset, neuropil=neuropil).Xdf(index_type='string')

    # Generate a flat map cluster_id -> members
    print('Finding cluster assignments')
    clusters_df, _ = get_original_clustering(dataset=dataset, neuropil=neuropil,
                                             clusterer_or_k=clusterer_or_k)
    dfs = []
    for cluster_id, members in zip(clusters_df.cluster_id,
                                   clusters_df.original_voxels_in_cluster):
        dfs.append(pd.DataFrame({'cluster_id': cluster_id, 'member_id': members}))
    members_df = pd.concat(dfs).set_index('member_id').loc[Xdf.index]

    # Compute the distance matrix - this must be parameterised
    print('Computing distance')
    import mkl
    mkl.set_num_threads(6)
    D = dicedist_metric(Xdf)

    # Compute silhouette
    # Here we could go for the faster implementation in third_party, if needed
    print('Computing silhouette index')
    members_df['silhouette'] = silhouette_samples(D.values,
                                                  members_df.cluster_id.values,
                                                  metric='precomputed')
    return (members_df.
            reset_index().
            rename(columns=lambda col: {'index': 'member_id'}.get(col, col))
            [['cluster_id', 'member_id', 'silhouette']])


if __name__ == '__main__':
    silhouette_df = silhouette_original_clusterings()
    print(silhouette_df.groupby('cluster_id').silhouette.mean())


#
# Bonus: annotate each cluster member with its silhouette index
#
# silhouettes = silhouette_samples(X, labels, metric='dice')
# or
# silhouettes = silhouette_samples(D, labels, metric='precomputed')
#
# A better answer to reviewers wanting to bound k:
#   better look at measures of individual cluster quality
#
# This function should just take Xdf, clusters_df or similar...
#

# Simpler: (normalised) distance to medoid.
# It will be independent from other clusters if normalisation does not take into account them.
