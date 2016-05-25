#!/usr/bin/env python
from __future__ import print_function

import os
import os.path as op
from itertools import product

import pandas as pd

from braincode.util import (get_filenames, get_all_datasets, get_all_neuropils,
                            get_finished_cluster_types, get_original_clustering,
                            ExpressionDataset, ensure_dir)
from braincode.revisions.config import BRAINCODE_DATA_CACHE_DIR

FORCE_REWRITE = bool(int(os.environ.get('FORCE_REWRITE', '0')))


def cluster_count(cluster_id, cluster_members_df):
    """
    Computes the expression levels (counts) for a given cluster id and its associated cluster members.

    Parameters
    ----------
    cluster_id : object
      The identifier for the cluster

    cluster_members_df : pandas DataFrame
      The voxels in rows, drivers in columns, expression value dataframe for the voxels in the cluster.

    Returns
    -------
    A pandas dataframe with a driver per row and columns:
      - driver_id: the database id of the driver line
      - cluster_id: the cluster identifier
      - num_voxels_this_cluster: the total number of voxels in the cluster
      - num_expressing_voxels_this_cluster: the number of voxels of the cluster expressed in the line
    """
    column_order = [
        'cluster_id',
        'driver_id',
        'num_expressing_voxels_this_cluster',
        'num_voxels_this_cluster'
    ]
    return (cluster_members_df.
            sum(axis=0).
            to_frame(name='num_expressing_voxels_this_cluster').
            reset_index().
            rename(columns={'index': 'driver_id'}).
            assign(cluster_id=cluster_id, num_voxels_this_cluster=len(cluster_members_df))
            [column_order].
            astype(int))


def get_fragment_cache_fname(dataset,region,cluster_type):
    fragment_cache_dir = ensure_dir(BRAINCODE_DATA_CACHE_DIR)
    fn = 'fragment_statistics__{dataset}__{region}__{cluster_type}.h5'.format(
            dataset=dataset,
            region=region,
            cluster_type=cluster_type,
        )
    return op.join(fragment_cache_dir,fn)


def calculate_statistics(neuropil, dataset):
    # Load the expression matrix
    if neuropil != 'entire':
        edf = ExpressionDataset.dataset(dset=dataset, neuropil=neuropil).Xdf(index_type='string').astype(int)
    else:
        # this is hacked together to be like the above
        from braincode.revisions.hub import Hub
        from braincode.revisions.pipelineit import Xcsr, DOWNSAMPLER_CB1, DOWNSAMPLER_T1
        downsampler = DOWNSAMPLER_CB1 if dataset == 'CB1' else DOWNSAMPLER_T1

        hub = Hub.hub(dataset)
        X, img_shape = Xcsr(pipeline=hub.original_pipeline())
        raise NotImplementedError()
        # voxels = X
        # self._voxels = X['voxels'][:]
        # Xarray = X.toarray()
        # index = pd.MultiIndex.from_arrays(voxels.T, names=['x', 'y', 'z'])
        # edf = pd.DataFrame(Xarray,
        #                 index=index,
        #                 columns=self.lines(),
        #                 copy=False)


    # Counts for "everywhere" region
    no_cluster_counts = cluster_count(cluster_id=-1, cluster_members_df=edf)

    for cluster_type in get_finished_cluster_types(dataset, neuropil):
        print(dataset, neuropil, cluster_type)

        filenames = get_filenames(dataset, neuropil, cluster_type)
        fragment_cache_fname = get_fragment_cache_fname(dataset=dataset,
            region=neuropil, cluster_type=cluster_type)
        print('\tfragment_cache_fname', fragment_cache_fname)
        if not FORCE_REWRITE and os.path.isfile(fragment_cache_fname):
            print('\t\tcache exists, will not recompute %r' % fragment_cache_fname)
            continue

        # Load the clustering results
        print('\tReading clusters...')
        clusters_df, _ = get_original_clustering(dataset=dataset,
                                                 neuropil=neuropil,
                                                 clusterer_or_k=cluster_type)

        # Cross product clusters and lines / drivers
        print('\tCounting each driver expression level...')
        this_clustering_counts = [
            cluster_count(cluster_id=cluster_id, cluster_members_df=edf.loc[cluster_voxels])
            for cluster_id, cluster_voxels in zip(clusters_df.cluster_id,
                                                  clusters_df.original_voxels_in_cluster)]
        # Merge
        this_clustering_counts = pd.concat([no_cluster_counts] + this_clustering_counts)

        # Save to disk
        print('\tSaving %r' % fragment_cache_fname)
        this_clustering_counts.to_hdf(fragment_cache_fname, 'fragment_stats')

if __name__=='__main__':
    for neuropil, dataset in product(get_all_neuropils(), get_all_datasets()):
        calculate_statistics(neuropil, dataset)
