#!/usr/bin/env python
from __future__ import print_function

import sys

import pandas as pd

if 1:
    fname = sys.argv[1]
    orig_store = pd.HDFStore(fname,mode='r')
    stats_df = orig_store['stats']
    for dataset,dataset_df in stats_df.groupby('dataset'):
        for neuropil,neuropil_df in dataset_df.groupby('neuropil'):
            for K,clustering_result_df in neuropil_df.groupby('K'):
                n_voxels = clustering_result_df['N_voxels_in_cluster'].values
                print('%s %s %s' % (dataset, neuropil, K ))

                # print cluster id by number of voxels in the cluster
                num2id = {}
                for cluster_id,cluster_df in clustering_result_df.groupby('cluster_id'):
                    num = cluster_df['N_voxels_in_cluster'].values[0]
                    num2id[ num ] = cluster_id
                nums = num2id.keys()
                nums.sort()
                mystr = ', '.join(['%d: %d' % ( num2id[num], num ) for num in nums ])
                print( '  {' + mystr + '}' )
