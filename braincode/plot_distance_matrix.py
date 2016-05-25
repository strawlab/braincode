#!/usr/bin/env python
from __future__ import print_function

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from braincode.thirdparty import colormaps
import os
import h5py
import numpy as np

import nrrd # pip install pynrrd

from braincode.util import get_filenames, calculate_nrrd_to_h5_LUTs, \
    get_position_keys_from_cluster_id

def get_index_for_position_key( position_key, all_voxel_names ):
    '''given a position key, find the index into the distance matrix'''
    cond = all_voxel_names == position_key
    idx = np.nonzero(cond)
    assert len(idx)==1
    idx = idx[0]
    assert len(idx)==1
    return idx[0]

def sort_columns(arr):
    assert arr.ndim==2

    remaining_cols = {}
    for j in range(arr.shape[1]):
        this_col = arr[:,j]
        remaining_cols[j] = this_col

    col_order = []
    for i in range(len(arr)):
        bestj = None
        highest = -np.inf
        for j in remaining_cols:
            this_col = remaining_cols[j]
            test_val = this_col[i]
            if test_val > highest:
                highest = test_val
                bestj = j
        if bestj is not None:
            del remaining_cols[bestj]
            col_order.append( bestj )
    col_order.extend( remaining_cols.keys() )
    cols = [ arr[:,j] for j in col_order ]
    result = np.array(cols).T
    assert result.shape == arr.shape
    return result, col_order

def main():

    if 1:
        dataset = 'CB1'
        neuropil = 'Optic_Glomeruli'
        cluster_type = 'K60_dicedist_orig'
    elif 0:
        dataset = 'T1'
        neuropil = 'Antennal_lobe'
        cluster_type = 'K60_dicedist'

    filenames = get_filenames( dataset, neuropil, cluster_type )
    nrrd_data, _ = nrrd.read(filenames['clustering_result_nrrd'])

    data = h5py.File( filenames['h5'], mode='r')

    all_voxel_names = np.array(data['positionKeys'][()].strip(',').split(','))
    local_filenames = get_filenames(dataset, neuropil)
    print('loading',local_filenames['distance_npy'])
    distance_matrix = np.load(local_filenames['distance_npy'])
    print('done loading')
    assert distance_matrix.ndim==2
    assert len(all_voxel_names)==distance_matrix.shape[0]
    assert distance_matrix.shape[0]==distance_matrix.shape[1]

    # ----------------------------

    # Get the indices in the distance matrix of the voxels in each cluster -----
    idxs_by_cluster_id = {}
    luts = calculate_nrrd_to_h5_LUTs( data )
    cluster_ids = np.unique(nrrd_data.ravel())
    for cluster_id in cluster_ids:
        if cluster_id==0:
            continue
        pks = get_position_keys_from_cluster_id( cluster_id, nrrd_data, luts )
        idxs_by_cluster_id[int(cluster_id)] = [ get_index_for_position_key( pk, all_voxel_names ) for pk in pks ]

    # ----------------------------

    # Now plot a subset of the distance matrix for a few clusters ----

    clusters = [8, 25, 24, 11, 43]
    idxs = []
    for cluster_id in clusters:
        idxs.extend( idxs_by_cluster_id[cluster_id] )
    N = len(idxs)

    shuffled=False

    # shuffled mode is to show the pre-sorting distance matrix.  We
    # shuffle it so that the inherent structure from the voxel
    # rasterization ordering is not visible and thus not confusing as
    # it already looks very structured if not shuffled.

    if shuffled:
        import random
        # overwrite idxs list with a new one
        idxs = range(distance_matrix.shape[0])
        random.shuffle(idxs)
        idxs = idxs[:N]

    row_idx = np.array(idxs)[:,None]
    col_idx = np.array(idxs)
    subsampled = 1.0 - distance_matrix[row_idx, col_idx] # convert from distance (1-dice) back to dice
    del distance_matrix
    print(subsampled.shape)

    for i in range(N):
        subsampled[i,i] = 1 # In older versions, we didn't compute this, so this lets us plot older cached files.

    assert N == subsampled.shape[0]

    cluster_ticks = not shuffled

    nan_cluster = True
    if cluster_ticks:
        ticks = [0]

        cum = 0
        for enum,cluster_id in enumerate(clusters):
            cum += len( idxs_by_cluster_id[cluster_id] )
            ticks.append( cum )
            if enum==2:
                nan_start = ticks[-2]
                nan_stop = ticks[-1]
        if nan_cluster:
            subsampled[ nan_start:nan_stop, : ] = np.nan
            subsampled[ :, nan_start:nan_stop ] = np.nan
    else:
        ticks = None

    out_figname = os.path.splitext(local_filenames['distance_npy'])[0] + '-partial'
    if shuffled:
        out_figname += '-shuffled'
    plot_distance_matrix(subsampled,
                         cbar_label='coexpression (Dice coefficient)',
                         fname_base=out_figname,
                         xlabel='voxel',
                         ylabel='voxel',
                         xticks=ticks,
                         yticks=ticks)


def setup_plot_defaults():
    from strawlab_mpl.defaults import setup_defaults
    setup_defaults()

    import matplotlib
    rcParams = matplotlib.rcParams
    rcParams['font.size'] = 5.0

def plot_distance_matrix(subsampled,cbar_label,ax=None,fname_base=None,xlabel=None,ylabel=None,
                         xticks=None, xticklabels=None, yticks=None, yticklabels=None, vmin=None, vmax=None):
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # ensure we don't accidentally clip data
    if vmin is not None:
        datamin = np.min(subsampled)
        print('datamin:',datamin)
        assert vmin <= datamin

    if vmax is not None:
        datamax = np.max(subsampled)
        print('datamax:',datamax)
        assert vmax >= datamax

    cmap = ax.imshow( subsampled, origin='upper', interpolation='nearest', cmap=colormaps.viridis, vmin=vmin, vmax=vmax )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks( xticks )

    if yticks is not None:
        ax.set_yticks( yticks )

    if xticklabels is not None:
        ax.set_xticklabels( xticklabels )
    if yticklabels is not None:
        ax.set_yticklabels( yticklabels )

    if 1:
        ax.set_frame_on(False)
        for axis in [ax.xaxis, ax.yaxis]:
            for tic in axis.get_major_ticks():
                tic.tick1On = tic.tick2On = False

    if 1:
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=90)

    if fig is not None:
        if fname_base is None:
            raise ValueError('if no ax given, must set fname_base')

        cbar = plt.colorbar(cmap)
        cbar.set_label(cbar_label)

        #for ext in ['.png','.svg','.pdf']:
        for ext in ['.png','.svg']:
            out_figname = fname_base+ext
            fig.savefig(out_figname,dpi=200)
            print('saved %s' % out_figname)
    return cmap

if __name__=='__main__':
    main()
