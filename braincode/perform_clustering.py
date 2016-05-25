#!/usr/bin/env python
import os
import errno

import h5py
import numpy as np
import nrrd # called pynrrd on PyPI

import calculate_distance
import util
from braincode.thirdparty import kmedoids_salspaugh as kMedoids


def save_to_nrrd_xyz(input_h5_fname, clusters, xyz, nrrd_fname):
    cluster_medoids = list(np.unique(clusters))
    h5_data = h5py.File(input_h5_fname, mode='r')

    stepXY = h5_data['stepXY'][()][0]
    stepZ = h5_data['stepZ'][()][0]

    nrrd_locs = []
    nrrd_values = []
    for medoid, (x, y, z) in zip(clusters, xyz):
        r = util.h5_coord_to_nrrd(x, stepXY)
        s = util.h5_coord_to_nrrd(y, stepXY)
        t = util.h5_coord_to_nrrd(z, stepZ)
        nrrd_locs.append( (r,s,t) )
        cluster_id = cluster_medoids.index(medoid) + 1
        assert cluster_id != 0 # must be found
        nrrd_values.append( cluster_id )
    nrrd_locs = np.array(nrrd_locs)
    nrrd_shape = np.max( nrrd_locs, axis=0) + 1
    nrrd_data = np.zeros( nrrd_shape )
    for loc,val in zip(nrrd_locs, nrrd_values):
        r,s,t=loc
        nrrd_data[r,s,t] = val
    assert np.sum(np.isnan(nrrd_data)) == 0

    assert np.max(nrrd_data) <= 255
    nrrd_data = nrrd_data.astype(np.uint8)


    outdir = os.path.dirname( nrrd_fname )
    try:
        os.makedirs(outdir)
    except OSError as err:
        if err.errno!=errno.EEXIST:
            raise

    nrrd.write(nrrd_fname, nrrd_data)


def save_to_nrrd( input_h5_fname, clusters, nrrd_fname ):
    cluster_medoids = list(np.unique(clusters))
    n_clusters = len(cluster_medoids)
    h5_data = h5py.File( input_h5_fname, mode='r')

    stepXY = h5_data['stepXY'][()][0]
    stepZ = h5_data['stepZ'][()][0]

    voxel_names = h5_data['positionKeys'][()].strip(',').split(',')
    nrrd_locs = []
    nrrd_values = []
    for i,vn in enumerate(voxel_names):
        x,y,z = util.voxel_name_to_idx(vn)
        r = util.h5_coord_to_nrrd(x, stepXY)
        s = util.h5_coord_to_nrrd(y, stepXY)
        t = util.h5_coord_to_nrrd(z, stepZ)
        nrrd_locs.append( (r,s,t) )
        medoid = clusters[i]
        cluster_id = cluster_medoids.index(medoid) + 1
        assert cluster_id != 0 # must be found
        nrrd_values.append( cluster_id )
    nrrd_locs = np.array(nrrd_locs)
    nrrd_shape = np.max( nrrd_locs, axis=0) + 1
    nrrd_data = np.zeros( nrrd_shape )
    for loc,val in zip(nrrd_locs, nrrd_values):
        r,s,t=loc
        nrrd_data[r,s,t] = val
    assert np.sum(np.isnan(nrrd_data)) == 0
    
    assert np.max(nrrd_data) <= 255
    nrrd_data = nrrd_data.astype(np.uint8)


    outdir = os.path.dirname( nrrd_fname )
    try:
        os.makedirs(outdir)
    except OSError as err:
        if err.errno!=errno.EEXIST:
            raise

    nrrd.write(nrrd_fname, nrrd_data)

def main():
    dataset = 'CB1'
    neuropil = 'fake'
    filenames = util.get_filenames(dataset, neuropil)
    calculate_distance.save_distances( filenames['h5'], filenames['distance_npy'])
    D = np.load(filenames['distance_npy'])
    k=3
    clusters, curr_medoids = kMedoids.cluster(D,k=k)
    cluster_type = 'K%02d_dicedist'%(k,)
    filenames = util.get_filenames(dataset, neuropil, cluster_type=cluster_type)
    nrrd_fname = filenames['clustering_result_nrrd']
    save_to_nrrd( filenames['h5'], clusters, nrrd_fname )

if __name__=='__main__':
    main()
