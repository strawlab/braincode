# coding=utf-8
"""Adding hierarchies to the analysis."""
from __future__ import division, print_function
from future.utils import string_types

# Avoid matplotlib crashes when sshing or in the cluster
import os

from scipy.cluster.hierarchy import ClusterNode

if not os.environ.get('DISPLAY'):
    print('DISPLAY NOT SET: USING AGG BACKEND')
    import matplotlib
    matplotlib.use('agg')

import json
import os.path as op
from functools import partial
from itertools import product

from braincode.thirdparty import colormaps

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from fastcluster import linkage
from scipy.misc import imsave
from scipy.spatial.distance import squareform
from sklearn.utils.extmath import cartesian

from braincode.dice import dicedist_metric
from braincode.revisions.config import BRAINCODE_SUP_DATA_DIR, BRAINCODE_DATA_CACHE_DIR
from braincode.revisions.hub import Hub
from braincode.revisions.images import masked2value, choose_min_int_dtype, VoxelLabels
from braincode.revisions.filenames import (get_hierarchy_temp_dir,
    get_hierarchy_dir, get_hierarchy_file_prefix)
from braincode.util import get_original_clustering, get_all_neuropils, get_all_datasets, \
    get_finished_cluster_types, ensure_dir, voxelid2coords
import braincode.util as util
from braincode.fragments_per_cluster_step1_compute import get_fragment_cache_fname

def colormap_gamma(input_colors, A=1.0, gamma=1.9):
    viridis_data = np.array(input_colors)
    assert viridis_data.ndim == 2
    n_rows = viridis_data.shape[0]
    orig_x = np.arange(viridis_data.shape[0], dtype=np.float) / n_rows
    A = 1.0
    gamma = 1.9
    new_x = A*orig_x**gamma

    expanded_data = np.nan*np.ones_like(viridis_data)
    for axis in [0,1,2]:
        expanded_data[:,axis] = np.interp( orig_x, new_x, viridis_data[:,axis] )

    expanded = ListedColormap(expanded_data)
    return expanded

def hierarchy2dictionary(Z, dendrogram=False, base=0):
    """
    Copies a linkage clustering matrix Z into a recursive dictionary structure.
    It is possible to add associated grapical dendrogram coordinates to the dictionary.

    Parameters
    ----------
    Z : scipy linkage matrix
      See http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    dendrogram : boolean or scipy dendrogam R matrix, default False
      If an scipy dendrogram, data to reproduce the graphical dendrogram is added to each node in the dictionary.
      If True, a dendrogram with default parameters is generated.
      If False or None, no dendrogram information is added to the dictionary.
      See http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

    base : int, default 0
      A number to add to the cluster id in Z, which originally is 0-based.
      Set to 1 to reserve 0 to use 1-based clustering (e.g. to reserve 0 for "no cluster").

    Returns
    -------
    The hierachical tree as a dictionary, including the leaves.
    Each node has "id", "left", "right", "distance", and "count", with the same meaning as in ClusterNode.
      See http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterNode.html
    Leaf nodes have left and right as None, distance as 0 and count as 1.
    If a dendrogram is provided, each node has in addition information to reproduce the dendrogram:
      leaf (aka singleton) nodes:
        - order of the node in the tree (usually left-to-right, but it depends on the dendrogram)
        - label of the node, typically its id
      cluster nodes:
        - icoord: a 4-tuple with segments x coordinates
        - dcoord: a 4-tuple with segments y (subtree depth relative to merged clusters distance) coordinates
        - color: a matplotlib color string; these are used to signify very different clusters,
                 for some meaning of "very"

    Examples
    --------
    Here is a 9-point dataset with 4 obvious clusters, which in turn merge into 2 obvious clusters
    >>> blob1 = np.array([[-9, -9], [-8.5, -8.5], [-8, -8]])
    >>> blob2 = np.array([[-5, -5], [-6, -6]])
    >>> blob3 = -blob2
    >>> blob4 = -blob1[:2]
    >>> X = np.vstack((blob1, blob2, blob3, blob4))

    Average linkage clustering
    >>> Z = hierarchy.average(X)

    We can generate a dendrogram; for better understanding the example, do plot the dendrogram
    >>> dendrogram = hierarchy.dendrogram(Z, no_plot=False)

    Let's now get these clustering and visualisation info into a dict
    >>> tree = hierarchy2dictionary(Z, dendrogram=dendrogram)

    Here we have information about the "left-most" leaf
    >>> left_most = tree['left']['left']['left']
    >>> left_most == {'id': 2,
    ...               'count': 1, 'distance': 0,
    ...               'left': None, 'right': None,
    ...               'order': 0, 'label': '2'}
    True

    What about the "right-most" leaf?
    >>> right_most = tree['right']['right']['right']
    >>> right_most == {'id': 6,
    ...                'count': 1, 'distance': 0,
    ...                'left': None, 'right': None,
    ...                'order': 8, 'label': '6'}
    True

    These were singleton clusters; what about non singletons? For example, the tree root
    >>> sorted(tree.keys())
    ['color', 'count', 'dcoord', 'distance', 'icoord', 'id', 'left', 'right']
    >>> tree['id'], tree['count']
    (16, 9)
    >>> tree['left']['id'], tree['left']['count']
    (14, 5)
    >>> tree['right']['id'], tree['right']['count']
    (15, 4)
    """

    # possibly compute and extract info from the dendrogram
    if dendrogram:
        if dendrogram is True:
            dendrogram = hierarchy.dendrogram(Z, no_plot=False)
        leaves, leaf_labels, icoords, dcoords, colors = (
            dendrogram['leaves'],      # leaf nodes order
            dendrogram['ivl'],         # leaf nodes labels
            dendrogram['icoord'],      # segments x coordinates
            dendrogram['dcoord'],      # segments y coordinates (graphical tree depth)
            dendrogram['color_list'],  # link colors
        )

    # this function will map a node_id to the relevant dendrogram info
    def id2dendroinfo(node_id):
        if not dendrogram:
            return {}
        elif node_id <= len(Z):
            # Leaf node
            return {'order': leaves.index(node_id),
                    'label': leaf_labels[leaves.index(node_id)]}
        else:
            # Cluster
            node_id = len(Z) + 1 - node_id
            return {
                'icoord': icoords[node_id],
                'dcoord': dcoords[node_id],
                'color': colors[node_id],
            }

    # generate a python dict with the clusters dict, possibly adding dendrogram information
    def h2dict(tree):

        if tree is None:
            return None

        tree = {'id': tree.id + base,
                'left': h2dict(tree.get_left()),
                'right': h2dict(tree.get_right()),
                'distance': tree.dist,
                'count': tree.count}
        tree.update(id2dendroinfo(tree['id']))

        return tree

    return h2dict(hierarchy.to_tree(Z))


def dictionary2hierarchy(hdict):
    """
    Returns the ClusterNode root of the tree for the scipy hierarchical clustering represented by dictionary hdict.
    """
    # TODO: check roundtripping with hierarchy2dictionary
    if hdict is None:
        return None
    return ClusterNode(
        id=hdict['id'],
        left=dictionary2hierarchy(hdict['left']),
        right=dictionary2hierarchy(hdict['right']),
        dist=hdict['distance'],
        count=hdict['count'],
    )


def hclusters2singletons(Z, id_base=0):
    """Given a hierarchical clustering, returns a dictionary {cluster_id: [singletons]}.
    The dictionary maps non-singleton clusters to a sorted list of singleton cluster ids it contains.
    Z can be either a dictionary, a ClusterNode or a scipy linkage matrix.

    Examples
    --------
    Here is a 9-point dataset with 4 obvious clusters, which in turn merge into 2 obvious clusters
    >>> blob1 = np.array([[-9, -9], [-8.5, -8.5], [-8, -8]])
    >>> blob2 = np.array([[-5, -5], [-6, -6]])
    >>> blob3 = -blob2
    >>> blob4 = -blob1[:2]
    >>> X = np.vstack((blob1, blob2, blob3, blob4))

    Average linkage clustering.
    >>> Z = hierarchy.average(X)

    Singleton clusters (at the leaves) are numbered 0 to 8.
    Here we get 8 non-singleton clusters, numbered 9 to 16.
    Note how cluster 9 is not "the best" cluster possible (i.e. [-9, -9] should be linked first to [-8.5, -8.5]).
    >>> expected = {9: [0, 1], 10: [7, 8],
    ...             11: [0, 1, 2], 12: [3, 4], 13: [5, 6],
    ...             14: [0, 1, 2, 3, 4], 15: [5, 6, 7, 8],
    ...             16: [0, 1, 2, 3, 4, 5, 6, 7, 8]}
    >>> hclusters2singletons(Z) == expected
    True

    We can rebase numbering, for example, to reserve 0 for "no cluster"
    >>> expected = {10: [1, 2], 11: [8, 9],
    ...             12: [1, 2, 3], 13: [4, 5], 14: [6, 7],
    ...             15: [1, 2, 3, 4, 5], 16: [6, 7, 8, 9],
    ...             17: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    >>> hclusters2singletons(Z, id_base=1) == expected
    True

    We can also use other representations for the hierarchical tree.
    For example, we can get a dictionary
    >>> expected = {10: [1, 2], 11: [8, 9],
    ...             12: [1, 2, 3], 13: [4, 5], 14: [6, 7],
    ...             15: [1, 2, 3, 4, 5], 16: [6, 7, 8, 9],
    ...             17: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    >>> hclusters2singletons(hierarchy.to_tree(Z), id_base=1) == expected
    True

    We can also pass the tree in a plain python dictionary
    >>> expected = {10: [1, 2], 11: [8, 9],
    ...             12: [1, 2, 3], 13: [4, 5], 14: [6, 7],
    ...             15: [1, 2, 3, 4, 5], 16: [6, 7, 8, 9],
    ...             17: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    >>> hdict = hierarchy2dictionary(Z, dendrogram=False, base=0)
    >>> hclusters2singletons(hdict, id_base=1) == expected
    True
    """

    # Accumulator for non-singleton nodes mapping to its singleton components...
    hc2s = {}

    # ...tree transversal with intermediate result caching
    def singletons(tree):
        if tree is None:
            return []
        cluster_id = tree.id + id_base
        if tree.is_leaf():
            return [cluster_id]
        singletons_in_tree = sorted(singletons(tree.left) + singletons(tree.right))
        hc2s[cluster_id] = singletons_in_tree
        return singletons_in_tree

    if isinstance(Z, ClusterNode):
        singletons(Z)
    elif isinstance(Z, dict):
        singletons(dictionary2hierarchy(Z))
    else:
        singletons(hierarchy.to_tree(Z))

    return hc2s


def singletons_from_json(json_path):
    """Given a hierarchical clustering in a json file, returns a dictionary {cluster_id: [singletons]}."""
    with open(json_path, 'r') as reader:
        return hclusters2singletons(json.load(reader)['tree'])


class AgglomeratedClusteringResult:
    def __init__(self,df,agglomeration_json_path):
        self.df = df
        self.singletons = singletons_from_json(agglomeration_json_path)

    def iter_clusters(self):

        results = []

        # get all singleton data (and cache underlying dataframes)
        singleton_frames = {}
        for singleton_cluster_id, cluster_df in self.df.groupby(['cluster_id']):
            singleton_frames[singleton_cluster_id] = cluster_df
            results.append( (singleton_cluster_id, cluster_df) )

        # get all agglomerated data
        for agglomerated_cluster_id in self.singletons:
            singleton_ids = self.singletons[agglomerated_cluster_id]
            my_frames = [ singleton_frames[q] for q in singleton_ids ]
            agglomerated_df = pd.concat( my_frames )
            results.append( (agglomerated_cluster_id, agglomerated_df) )

        return results


def voxel2voxels_in_volume(x, y, z, stepX, stepY, stepZ):
    """
    Returns a numpy array with all the voxels in the volume corresponding to representative (x, y, z).

    Here we assume that the representative is the upper, left, front pixel of a (stepX, stepY, stepZ) sized volume.
    """
    # This is what Andrew originally used. Probably not fully correct, but practical.
    # We could also just return slices and let numpy do its tiling magic...
    # This should be hidden in an up/down sampler object
    return cartesian((np.arange(x, x + stepX),
                      np.arange(y, y + stepY),
                      np.arange(z, z + stepZ)))


def voxels2voxels_in_volume(original_voxels_in_cluster, stepX, stepY, stepZ):
    cluster_representative_voxels = [voxelid2coords(voxel_id) for voxel_id in original_voxels_in_cluster]
    # Expand each cluster representative to all the voxels in the volume
    cluster_members = np.vstack([voxel2voxels_in_volume(x, y, z, stepX, stepY, stepZ)
                                 for x, y, z in cluster_representative_voxels])
    return cluster_members


def projection2png(cluster_id,
                   image,
                   dest_file=None,
                   aggregator=np.max,
                   axis='z'):
    """Projects a 3d image mask over an axis ('x', 'y', 'z'), possibly saving the result to a file."""
    if isinstance(axis, string_types):
        axis = [axis]
    projecteds = {}
    for axis in sorted(set(axis)):
        projected = aggregator(image, axis={'x': 0, 'y': 1, 'z': 2}[axis])
        if axis != 'x':
            projected = projected.T
        if dest_file is not None:
            dest_png = '{dest_file}_K{cluster_id:03d}_{axis}.png'.format(
                dest_file=dest_file,
                axis=axis,
                cluster_id=cluster_id)
            imsave(dest_png.format(axis=axis), projected)
        projecteds[axis] = projected
    return projecteds


def generate_clustering_image(template,
                              clusters_df,
                              fail_if_overlap=False):
    # --- Generate an image with cluster label assignments
    voxel2clusterid = np.zeros_like(template,
                                    dtype=choose_min_int_dtype(len(clusters_df)))

    for _, cluster in clusters_df.iterrows():
        assert cluster.cluster_id > 0
        # This is very arbitrary at the moment:
        #   - cluster_df contract should be to include a "members" columns
        #   - map to volume should be a movable part
        cluster_members = voxels2voxels_in_volume(cluster.original_voxels_in_cluster,
                                                  cluster.stepX, cluster.stepY, cluster.stepZ)
        cluster_members = tuple(cluster_members.T)
        if fail_if_overlap:
            if np.any(voxel2clusterid[cluster_members] > 0):
                raise Exception('Clusters overlap in the original space.\n'
                                'There might be problems on assigning volumes to cluster representatives.')
        voxel2clusterid[cluster_members] = cluster.cluster_id

    # Generate a table of singleton clusters, to feed VoxelLabels
    # Actually we could just use a trimmed down version of clusters_df (TODO)
    cluster_ids = list(clusters_df.cluster_id)
    labels_df = pd.DataFrame({
        'name': cluster_ids,
        'id': cluster_ids
    })

    return VoxelLabels(labels_df=labels_df, voxel2labelid=voxel2clusterid)


def process_cluster_images(dataset,               # this should be hub
                           clusters_df,           # the singleton clusters
                           Z,                     # scipy linkage matrix (make optional)
                           func=lambda *_: None,  # function cluster_id, cluster_image (visitor)
                           verbose=True):         # speak loud about all the nasty things we find?

    # Note that this does not average from the original intensity images,
    # but we just still just consider on-off voxels.
    # The former could be interesting too.

    template, bb, bb_info = Hub.hub(dataset).template()

    # Generate an image with cluster label assignments
    vl = generate_clustering_image(template, clusters_df, fail_if_overlap=verbose)

    # Generate the image for each singleton
    for cluster_id in clusters_df.cluster_id:
        func(cluster_id=cluster_id,
             image=masked2value(~vl.mask(cluster_id), masked_value=255))

    # Generate the image for each nin singleton
    if Z is not None:
        # Generate the map non-singleton-cluster -> [singleton1, singleton2...]
        vl.add_label_groups(hclusters2singletons(Z, id_base=1))
        for cluster_id in vl.label_group_names():
            func(cluster_id=cluster_id,
                 image=masked2value(~vl.mask(cluster_id), masked_value=255))


def cophenetic_best(condensedD, methods=('single', 'complete', 'average', 'weighted')):
    # What hierarchical clustering method is the best, according to the cophenetic correlation?
    # 'centroid', 'median' and 'ward' do not make sense with dice, since the dm needs to be Euclidean
    # In fact, they require the original matrix and not the distance matrix
    # (so change the API if ever considereing them).
    results = {}
    for method in methods:
        Z = linkage(condensedD, method=method)
        cophenetic_correlation, _ = hierarchy.cophenet(Z, condensedD)
        results[method] = cophenetic_correlation
    results = pd.Series(results)
    return results.sort_values(ascending=False), results.idxmax()


def get_agglomerated_clustering_result(dataset=None, region=None, cluster_type=None):
    package_data_dir = get_hierarchy_dir(dataset=dataset,
        region=region, cluster_type=cluster_type)
    fn = get_hierarchy_file_prefix(dataset=dataset,
        region=region, cluster_type=cluster_type)
    agglomeration_json_path = op.join(package_data_dir, fn + '.json')

    fragment_cache_fname = get_fragment_cache_fname(dataset=dataset,
        region=region, cluster_type=cluster_type)
    print('reading %r'%fragment_cache_fname)
    store = pd.HDFStore( fragment_cache_fname, mode='r' )
    df = store['fragment_stats']
    store.close()
    print('done reading')

    agglom = AgglomeratedClusteringResult(df, agglomeration_json_path)
    return agglom


if __name__ == '__main__':
    import seaborn as sns # avoid side effects of import by deferring to __main__
    from braincode.plot_distance_matrix import setup_plot_defaults
    setup_plot_defaults()

    sns.set(context='talk')

    testing = False
    datasets = ['CB1'] if testing else get_all_datasets()
    neuropils = ['Antennal_lobe'] if testing else get_all_neuropils()

    for dataset, neuropil in product(datasets, neuropils):
        cluster_types = ['K60_dicedist'] if testing else get_finished_cluster_types(dataset, neuropil)
        for cluster_type in cluster_types:
            print(neuropil, dataset, cluster_type)
            package_data_dir = ensure_dir(get_hierarchy_dir(dataset=dataset,
                region=neuropil, cluster_type=cluster_type))
            intermediate_dir = ensure_dir(get_hierarchy_temp_dir(dataset=dataset,
                neuropil=neuropil, cluster_type=cluster_type))

            # Load the clusters
            print('Loading the clustering data')
            clusters_df, medoids_df = get_original_clustering(dataset=dataset,
                                                              neuropil=neuropil,
                                                              clusterer_or_k=cluster_type)

            # --- Hierarchical clustering of medoids
            #  For consistency, use the dice distance (we could also use pdist(metric='dice))
            #  We stick stick with average-linkage
            #  Maybe we should check how original medoid order affects the clustering
            print('Clustering the medoids')
            D = dicedist_metric(medoids_df)  # And this is as cool as for spitting back a pandas dataframe
            condensedD = squareform(D)

            # Clustering time is irrelevant, report the qualities all at once
            cophenetic_ranking, best_method = cophenetic_best(condensedD)
            print('Cophenetic ranking\n%s\nbest: %s' % (cophenetic_ranking, best_method))

            linkage_method = 'average'
            print('Linkage: %s' % linkage_method)

            # --- Perform the linkage calculation
            Z = linkage(condensedD, method=linkage_method)
            cophenetic_correlation, _ = hierarchy.cophenet(Z, condensedD)

            # --- Save clustering to a json file for web-ingestion

            fn = get_hierarchy_file_prefix(dataset=dataset,
                region=neuropil, cluster_type=cluster_type)

            def save_hierarchy_json():
                # To keep the json small we should probably reduce the digits we save
                #   http://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module
                # And of course, remove spaces, and maybe, use much shorter keys...
                # So make a function out of this, with parameter "small", and coordinate with the js world
                print('Saving json')
                tree = hierarchy2dictionary(Z, dendrogram=False, base=1)
                hdict = {
                    'dataset': dataset,
                    'region': neuropil,
                    'clusterer': cluster_type,
                    'linkage': linkage_method,
                    'cophenetic_correlation': cophenetic_correlation,
                    'dendrogram': 'default',
                    'tree': tree,
                }
                json_file = op.join(package_data_dir, fn + '.json')
                with open(json_file, 'w') as writer:
                    json.dump(hdict, writer, indent=2, sort_keys=True)
            save_hierarchy_json()

            # --- Save max-projections
            if 0:
                # N.B. if we ever generate also images for non-singletons, linkage_method must come here
                print('Saving singleton and agglomerated images')

                imgsaver = partial(projection2png,
                                   axis=['x', 'y', 'z'],
                                   dest_file=op.join(ensure_dir(intermediate_dir), 'cluster'))
                process_cluster_images(dataset,
                                       clusters_df,
                                       Z=Z,
                                       func=imgsaver,
                                       verbose=False)
            else:
                print('Skipping: saving singleton and agglomerated images')

            # --- Save a couple quick plots
            def save_scipy_default_dendrogram():
                print('Saving dendrogram')
                hierarchy.dendrogram(Z)
                plt.suptitle('Medoids for {dataset}, {neuropil}, {cluster_type}, {hmethod} linkage'.
                             format(dataset=dataset,
                                    neuropil=neuropil,
                                    cluster_type=cluster_type,
                                    hmethod=linkage_method))
                plt.savefig(op.join(intermediate_dir, fn + '.scipy-dendrogram.png'))
                plt.savefig(op.join(intermediate_dir, fn + '.scipy-dendrogram.svg'))
                plt.close()
            save_scipy_default_dendrogram()

            def save_clustermap():
                expanded = colormap_gamma(colormaps.viridis.colors)
                expanded_r = ListedColormap(expanded.colors[::-1])
                print('Saving clustermap')
                plt.figure()
                cluster_grid = sns.clustermap(D,
                                              row_cluster=True, col_cluster=True,
                                              row_linkage=Z, col_linkage=Z,
                                              cmap=expanded_r)
                # These are oriented wrongly...
                #   http://stackoverflow.com/questions/34572177/labels-for-clustermap-in-seaborn
                plt.setp(cluster_grid.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
                plt.setp(cluster_grid.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
                # N.B. dendograms are the same on rows and columns, just because...
                plt.suptitle('Medoids for {dataset}, {neuropil}, {cluster_type}, {hmethod} linkage'.
                             format(dataset=dataset,
                                    neuropil=neuropil,
                                    cluster_type=cluster_type,
                                    hmethod=linkage_method))
                plt.savefig(op.join(intermediate_dir, fn + '.clustermap.png'))
                plt.savefig(op.join(intermediate_dir, fn + '.clustermap.svg'))
                plt.close()
            save_clustermap()

            print('-' * 80)

    print('DONE')
