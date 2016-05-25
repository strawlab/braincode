#!/usr/bin/env python
from __future__ import print_function

import os, errno, sys
import h5py
import numpy as np
import progressbar

from braincode.dice import dice_coefficient
from braincode.util import get_filenames, get_all_datasets, get_all_neuropils
from braincode import util

def save_distances_wrapper( dataset, neuropil ):
    filenames = get_filenames(dataset, neuropil)
    if not os.path.exists( filenames['h5'] ):
        # skip because data does not exist
        print('warning: in save_distances_wrapper(): file %r does not exist. skipping.'%(filenames['h5'],),file=sys.stderr)
        return

    outfname = filenames['distance_npy']
    print('output filename: %r' % outfname )
    if os.path.exists( outfname ):
        print('will not overwrite %r'%outfname, file=sys.stderr)
        return

    save_distances(filenames['h5'], outfname )

def save_distances(input_h5_fname, output_npy_fname ):
    data = h5py.File( input_h5_fname, mode='r')

    outdir = os.path.dirname( output_npy_fname )
    try:
        os.makedirs(outdir)
    except OSError as err:
        if err.errno!=errno.EEXIST:
            raise

    voxel_names = data['positionKeys'][()].strip(',').split(',')

    print('%r: %d positions' % (input_h5_fname, len(voxel_names)))
    ids = data['ids']
    N = len(voxel_names)

    if 1:
        test_result = np.zeros( (N,N), dtype=np.float32 )
        n_bytes=N*N*4
        n_GB=n_bytes/1024./1024./1024.
        print('writing %.1f GB test file %s'%(n_GB,output_npy_fname))
        np.save( output_npy_fname, test_result ) # npy file
        del test_result
        print('test file saved OK, will perform computation now')
        os.unlink( output_npy_fname )

    total_computations = N**2//2

    widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    pbar=progressbar.ProgressBar(widgets=widgets,maxval=total_computations).start()

    # generate place to store result
    result = np.zeros( (N,N), dtype=np.float32 )

    # generate cache
    set_cache = util.generate_set_cache( voxel_names, ids )
    assert len(set_cache)==N

    count = 0
    for i in range(N):
        row = []
        A = set_cache[i]
        for j in range(N):
            if j > i:
                # symmetric matrix, no need to compute
                break

            count += 1
            if count <= total_computations:
                pbar.update(count)

            B = set_cache[j]
            dist = 1.0 - dice_coefficient( A, B )
            result[i,j] = dist
            result[j,i] = dist

    pbar.finish()
    print('writing',output_npy_fname)
    np.save( output_npy_fname, result ) # npy file

def main():
    for neuropil in get_all_neuropils():
        for dataset in get_all_datasets():
            print('%s %s ----------------' % (dataset, neuropil) )
            save_distances_wrapper( dataset, neuropil )

if __name__=='__main__':
    main()
