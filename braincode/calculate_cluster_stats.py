#!/usr/bin/env python
from __future__ import print_function

import os
import sys

import h5py
import numpy as np
import pandas as pd
import nrrd  # pip install pynrrd

mydir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,mydir)
from braincode.util import get_filenames, get_all_neuropils, get_all_datasets, \
    get_finished_cluster_types, calculate_nrrd_to_h5_LUTs, \
    get_position_keys_from_cluster_id
from braincode import util
from braincode.thirdparty import kmedoids_salspaugh as kmedoids

for neuropil in get_all_neuropils():
    for dataset in get_all_datasets():
        filenames = get_filenames( dataset, neuropil )
        print(filenames['h5'])
        assert os.path.exists( filenames['h5'] )

for neuropil in get_all_neuropils():
    for dataset in get_all_datasets():
        filenames = get_filenames( dataset, neuropil )
        data = h5py.File( filenames['h5'], mode='r')

        voxel_names = data['positionKeys'][()].strip(',').split(',')

        print('%r: %d positions' % (filenames['h5'], len(voxel_names)))
        ids = data['ids']
        N = len(voxel_names)

        # read sparse data
        set_cache = util.generate_set_cache( voxel_names, ids )
        # make a list of all driver ids
        all_drivers = set()
        for sc in set_cache:
            all_drivers.update( sc )

        all_drivers = list(all_drivers)
        all_drivers.sort()
        M = len(all_drivers)
        print('%d drivers'%M)

        print('calculating dense matrix of size %d x %d'%(N,M))
        dense_matrix = np.zeros( (N,M), dtype=np.uint8 )
        for i in range(N):
            sc = set_cache[i]
            for driver in sc:
                j = all_drivers.index(driver)
                dense_matrix[i,j] = 1

        # load the distance matrix - you need to have run first "calculate_distance.py"
        print('loading',filenames['distance_npy'])
        distance_matrix = np.load(filenames['distance_npy'])
        print('loaded')

        luts = calculate_nrrd_to_h5_LUTs( data )
        cluster_types = get_finished_cluster_types( dataset, neuropil )
        for cluster_type in cluster_types:
            clustered_filenames = get_filenames( dataset, neuropil, cluster_type )
            nrrd_data, _ = nrrd.read(clustered_filenames['clustering_result_nrrd'])
            cluster_ids = np.unique(nrrd_data.ravel())

            clusters_by_id = {}
            for cluster_id in cluster_ids:
                if cluster_id==0:
                    continue
                position_keys = get_position_keys_from_cluster_id(
                    cluster_id, nrrd_data, luts )
                cluster = np.array([voxel_names.index(pk) for pk in position_keys])
                assert -1 not in cluster
                clusters_by_id[cluster_id] = cluster

            # compute medoids -------------------------------------------
            if 1:
                medoids = []
                for cluster_id in clusters_by_id:
                    cluster = clusters_by_id[cluster_id]
                    medoids.append( kmedoids.compute_new_medoid(cluster, distance_matrix) )
                medoids = np.array(medoids)

            # calculate average intra-cluster distance -------------------------------------
            if 1:
                intra_data = {'cluster_id':[],
                              'mean_dist':[],
                              'max_dist':[],
                              'num_elements':[],
                }
                inter_data = {'cluster_id_i':[],
                              'cluster_id_j':[],
                              'min_dist':[],
                              'medoids_dist':[],
                          }
                for midx, cluster_id in enumerate(clusters_by_id):
                    cluster = clusters_by_id[cluster_id]
                    medoid = medoids[midx]
                    cluster_distances = distance_matrix[np.ix_(cluster,[medoid])] # get distances to medoid from each cluster voxel

                    intra_data['cluster_id'].append(cluster_id)
                    intra_data['mean_dist'].append( np.mean( cluster_distances ) )
                    intra_data['max_dist'].append( np.max( cluster_distances ) )
                    intra_data['num_elements'].append( len(cluster) )

                    for midxj, cluster_id_j in enumerate(clusters_by_id):
                        if cluster_id_j >= cluster_id:
                            continue
                        medoid_j = medoids[midxj]
                        cluster_j = clusters_by_id[cluster_id_j]

                        intra_cluster_distances = distance_matrix[np.ix_(cluster,cluster_j)] # get distance between all elements
                        min_dist = np.min( np.min( intra_cluster_distances) )

                        inter_data['cluster_id_i'].append( cluster_id )
                        inter_data['cluster_id_j'].append( cluster_id_j )
                        inter_data['min_dist'].append( min_dist )
                        inter_data['medoids_dist'].append( distance_matrix[medoid,medoid_j] ) # get distance between medoids


                intra_df = pd.DataFrame(intra_data)
                intra_df.to_csv( clustered_filenames['cluster_stats_intra_csv'] )
                inter_df = pd.DataFrame(inter_data)
                inter_df.to_csv( clustered_filenames['cluster_stats_inter_csv'] )
