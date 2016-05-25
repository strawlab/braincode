# coding=utf-8
"""Analysis of brain and clusterings symmetry."""
from __future__ import print_function, division

import os.path as op

import numpy as np
import pandas as pd
from sklearn.utils.extmath import cartesian

from braincode.dice import dicedist_metric
from braincode.revisions.hub import Hub
from scipy.optimize import minimize
from braincode.revisions.config import mkl
from braincode.util import braincode_basedir, ensure_dir


# --- Find landmarks using left-right pairs

def mean_landmarker(coords):
    return {'coords_mean': np.mean(coords, axis=0)}


def find_landmarks(regions, landmarkers=(mean_landmarker,), verbose=False):
    """
    Find landmark voxels that should match in left and right regions.

    Parameters
    ----------
    regions : VoxelLabels instances
      It will be used to find voxels pertaining to the same neuropil in the left and right hemispheres.
      Left and right regions should be marked by suffixes '_L' and '_R', respectively.
      :type regions: braincode.revisions.hub.VoxelLabels

    landmarkers : list of functions (coordinates) -> {landmark_name: landmark_voxel}
      Functions that find landmarks on each region independently, for example the mean of the coordinates.
      These return a dictionary of named landmarks, where a landmark is a single voxel.

    verbose : bool, default False
      If True, prints logging info

    Returns
    -------
    A pandas dataframe with a row per landmark duple and columns:
      - region: the region name (e.g. 'AL')
      - landmark: the landmark name (e.g. 'coords_mean')
      - left: the voxel in the left hemisphere
      - right: the voxel in the right hemisphere
    """

    # Link left and right segmentations
    def zip_left_right(regions):
        all_labels = regions.labels()
        lefts = [label for label in all_labels if label.endswith('_L')]
        rights = [label for label in all_labels if label.endswith('_R')]
        zipped = list(zip(lefts, rights))
        for left, right in zipped:
            assert left[:-2] == right[:-2]
        return zipped

    left_rights = zip_left_right(regions)

    landmarks = []
    for left_name, right_name in left_rights:
        region_name = left_name[:-2]
        if verbose:
            print('Landmarking', region_name)
        # This should come from VoxelLabels directly
        left = np.vstack(np.where(~regions.mask(left_name))).T
        right = np.vstack(np.where(~regions.mask(right_name))).T
        for landmarker in landmarkers:
            landmarks_left = landmarker(left)
            landmarks_right = landmarker(right)
            for landmark_name in set(landmarks_left.keys() + landmarks_right.keys()):
                landmarks.append({
                    'region': region_name,
                    'landmark': landmark_name,
                    'left': landmarks_left[landmark_name],
                    'right': landmarks_right[landmark_name],
                })
    return pd.DataFrame(landmarks)


def find_symmetry_axis(landmarks_df):
    # segment midpoint
    # maybe we should allow to do it by landmark name
    return (landmarks_df.left + landmarks_df.right) / 2

# --- Find reflection planes via optimisation
#  Porting Florian's julia code.
#  http://docs.julialang.org/en/release-0.4/manual/noteworthy-differences/#noteworthy-differences-from-python


def build_reflection_matrix(plane):
    """Given a plane [a, b, c, d], returns a matrix to mirror points around it."""
    plane = np.asarray(plane)
    t = plane[:3] * plane[3]
    M1 = np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, 1],
    ])

    M2 = np.array([
        [1, 0, 0, -t[0]],
        [0, 1, 0, -t[1]],
        [0, 0, 1, -t[2]],
        [0, 0, 0, 1],
    ])

    a, b, c = plane[:3]

    # Householder reflection through zero
    H = np.array([
        [1 - 2 * a ** 2,     -2 * a * b,     -2 * a * c, 0],
        [    -2 * a * b, 1 - 2 * b ** 2,     -2 * b * c, 0],
        [    -2 * a * c,     -2 * b * c, 1 - 2 * c ** 2, 0],
        [             0,              0,              0, 1]
    ])

    return M2.dot(H).dot(M1).T


# --- Serial pure-numpy verion


def transform_coords(transformation, coords):
    dtype = coords.dtype
    # Maybe we should avoid mallocs and memcpys here by using out params...
    coords = np.hstack((coords,
                        np.ones((len(coords), 1), dtype=dtype)))
    transformed_coords = coords.dot(transformation)
    return np.round(transformed_coords)[:, 0:3].astype(dtype)


def symmetry_score(transformation, left, right, stepz=100, ignore_value=0):
    """Counts how many elements in reflected img2 are equal in img1."""
    sizex, sizey, sizez = left.shape
    score = 0
    for zstart in range(0, sizez, stepz):
        # Generate original coordinates
        coords = cartesian((np.arange(sizex),
                            np.arange(sizey),
                            np.arange(zstart, min(sizez, zstart + stepz))))
        # Reflect coordinates
        reflected_coords = transform_coords(transformation, coords)
        # Find valid transformations
        valid_coords = ((reflected_coords >= 0) &
                        (reflected_coords < (sizex, sizey, sizez))).all(axis=1)
        coords = coords[valid_coords]
        reflected_coords = reflected_coords[valid_coords]
        # print('There were %d of %d reflected points out of boundaries' %
        #       ((~valid_coords).sum(), len(valid_coords)))
        # Compute score
        equal = left[tuple(coords.T)] == right[tuple(reflected_coords.T)]
        valid = (left[tuple(coords.T)] != ignore_value) & (right[tuple(reflected_coords.T)] != ignore_value)
        score += np.sum(equal & valid)

    return score


def symmetry(plane, left, right, center_left, center_right):

    R = build_reflection_matrix(plane)
    reflected_center_left = center_left.dot(R)
    centers_distance = np.sqrt(np.sum((center_right - reflected_center_left) ** 2))
    # Here the centers distance have little to no weight in the objective function, right?
    score = symmetry_score(R, left, right) - centers_distance
    print(plane, score, centers_distance)
    return score


def find_plane(left, right):

    center_left = np.append(np.mean(np.vstack(np.where(left)), axis=1), [1])
    center_right = np.append(np.mean(np.vstack(np.where(left)), axis=1), [1])

    # function to minimise
    f = lambda plane: - symmetry(plane,
                                 left=left, right=right,
                                 center_left=center_left, center_right=center_right)
    # place the initial plane normal to the x axis and divide the volume equally
    x0 = np.array([1, 0, 0, -left.shape[0] / 2])

    # minimise; I think Florian used also Nelder-Meald
    result = minimize(f, x0, method='Nelder-Mead')

    return result.x


# REFLECTION_PLANE_T1_GLOM = [0.944373164629663, -0.01286895621761357, 0.005394607162490096, -383.6670661067513]
# REFLECTION_PLANE_CB1_GLOM = [0.9420408945534485, -0.014620835030903166, -0.01067825755774324, -516.2519460267946]
# REFLECTION_PLANE_T1_AL = [0.9126683261999301, -0.06253630085686733, -0.00016631546577492503, -383.9311307355962]
# REFLECTION_PLANE_CB1_AL = [0.9637241441943504, -0.023981711268061413, -0.022675228710894786, -511.105334530398]
#
# print('Loading data...')
# hub = Hub.hub('CB1')
# left = hub.regions().mask('Optic_Glomeruli_L')
# right = hub.regions().mask('Optic_Glomeruli_R')
# from braincode.revisions.config import mkl
# mkl.set_num_threads(6)
# print('Using at most %d threads' % mkl.get_max_threads())
# find_plane(left, right)
# exit(22)

#
# ATM, the julia version is easier in memory and probably faster,
# as does less pass through the data (although it dispatchs many more calls)
# It also avoids malloc and memcpy overhead, which we can easily do in numpy world.
# Also I need to give a thought on array memory layout and which coord is better to split on
# This seems fast enough though, although I have not really tested it yet.
#

# --- Parallel versions in Julia
#
# It seems to me they have not been used yet except for profiling in the notebook
# Ask Florian
#
# def p_symmetry_score_slap(M, I1, I2, startZ, spanZ):
#     """Counts how many "on elements" in reflected I2 are "on" too in I1, on a Z slice."""
#     pass
#
#
# def p_symmetry_score(M, I1, I2):
#     """Computes in parallel the symmetry score for transformation matrix M on Z slices."""
#     pass


def reflect_image(plane, img, upsample_ratio=(1, 1, 1)):
    """Reflects an image about a plane, possibly."""

    M = build_reflection_matrix(plane)

    if upsample_ratio is not None:

        xratio, yratio, zratio = upsample_ratio
        up = np.array([
            [xratio, 0, 0, 0],
            [0, yratio, 0, 0],
            [0, 0, zratio, 0],
            [0, 0, 0, 1],
        ])
        down = np.array([
            [1 / xratio, 0, 0, 0],
            [0, 1 / yratio, 0, 0],
            [0, 0, 1 / zratio, 0],
            [0, 0, 0, 1],
        ])

        M = down.dot(M).dot(up)
        # Isn't this an identity?

    def transform_coordinates(xyz, matrix, dtype=np.int32):
        return np.round(matrix.dot(xyz)).astype(dtype)

    # TODO: finish this


# --- Quick and dirty generation of results from Florian mirrored data

root_mirrored = op.expanduser('~/strawscience/santi/andrew/clustering-paper/forFLORIAN')


def mirror_pair(dataset='CB1', neuropil='Optic_Glomeruli', k=100):
    original = op.join(root_mirrored, 'clustering_nrrds', '{dataset}_{neuropil}_L_{k}.nrrd'.
                       format(dataset=dataset, neuropil=neuropil, k=k))
    mirrored = op.join(root_mirrored, 'clustering_nrrds_mirrored', 'mirrored_{dataset}_{neuropil}_R_{k}.nrrd'.
                       format(dataset=dataset, neuropil=neuropil, k=k))
    return original, mirrored


def measure_symmetry(original='~/CB1_Optic_Glomeruli_L_100.nrrd',
                     mirrored='~/mirrored_CB1_Optic_Glomeruli_R_100.nrrd',
                     dest_png=None):

    import nrrd

    def nrrd2labels(nrrd_file):
        img, _ = nrrd.read(op.expanduser(nrrd_file))
        return img.ravel(), img.shape

    def cluster_fingerprints(labels, prefix='left', banned_labels=(0,)):
        if banned_labels is None:
            banned_labels = ()
        cluster_fingerprints = {}
        for cluster_id in np.sort(np.unique(labels)):
            if cluster_id not in banned_labels:
                fingerprint = labels == cluster_id
                if prefix is not None:
                    cluster_id = '{prefix}{label}'.format(prefix=prefix, label=cluster_id)
                cluster_fingerprints[cluster_id] = fingerprint
        return pd.DataFrame(cluster_fingerprints).T

    print('Reading cluster fingerprints')
    original_clusters = cluster_fingerprints(nrrd2labels(original)[0],
                                             prefix='original',
                                             banned_labels=None)
    mirrored_clusters = cluster_fingerprints(nrrd2labels(mirrored)[0],
                                             prefix='mirrored',
                                             banned_labels=None)

    #
    # Options for fast computation inter-cluster computation without hair lost:
    #  cdist is probably slow (did not try)
    #  represent as sparse, sklearn computes distances there blazingly fast
    #  use our fast BLAS based distances
    #  ...

    print('Computing distances between clusters')
    X = pd.concat((original_clusters, mirrored_clusters)).astype(np.float32)
    mkl.set_num_threads(8)
    # This spits out a nicelly labelled dataframe, but is damn slow:
    #   D = dicedist_metric(X)
    # So we compute the dataframe manually
    D = dicedist_metric(X.values)
    # For the moment, let's just get rid of the diagonal
    np.fill_diagonal(D, np.inf)
    D = pd.DataFrame(data=D, index=X.index, columns=X.index, copy=False)
    # Because out of laziness I'm computing too much,
    # let's just keep original in rows and mirrored in cols
    D = D.loc[[o for o in D.index if o.startswith('original')]]
    D = D[[m for m in D.columns if m.startswith('mirrored')]]

    print('Plotting')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    cluster_grid = sns.clustermap(D, row_cluster=True, col_cluster=True)
    plt.setp(cluster_grid.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cluster_grid.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    figid = op.splitext(op.basename(dest_png))[0]
    plt.suptitle(figid)
    plt.savefig(dest_png)
    plt.close()

    # Nearest neighbors...
    # nearest_neighbors = D.idxmin(axis=0)
    # mindists = D.min(axis=0)
    # print(mindists)
    #
    # Unfortunatelly these distances are, in general, very large
    # So we need a distance that takes into account spatial location and not only cluster memberships
    # Or a better way to pair the voxels
    # Also, these clusterings are quite stochastic...
    #
    # This was a large k though...
    #

# for k in range(5, 126, 5):
#     print(k)
#     dest_png = op.join(op.expanduser('~'), 'CB1_Optic_Glomeruli_{k:03d}'.format(k=k))
#     original, mirrored = mirror_pair(k=k)
#     measure_symmetry(original, mirrored, dest_png=dest_png)
# exit(22)


# --- Some examples

if __name__ == '__main__':

    symmetry_dir = ensure_dir(op.join(braincode_basedir(), 'plots', 'symmetry'))
    landmarks_pkl = op.join(symmetry_dir, 'landmarks_df.pkl')

    if not op.isfile(landmarks_pkl):
        landmarks_df_cb1 = find_landmarks(Hub.hub('CB1').regions(), verbose=True)
        landmarks_df_t1 = find_landmarks(Hub.hub('T1').regions(), verbose=True)

        landmarks_df_cb1['symmetry_axis'] = find_symmetry_axis(landmarks_df_cb1)
        landmarks_df_cb1['dataset'] = 'CB1'
        landmarks_df_t1['symmetry_axis'] = find_symmetry_axis(landmarks_df_t1)
        landmarks_df_t1['dataset'] = 'T1'

        landmarks_df = pd.concat((landmarks_df_t1, landmarks_df_cb1))

        pd.to_pickle(landmarks_df, landmarks_pkl)

    # Show the "symmetry axis"
    landmarks_df = pd.read_pickle(landmarks_pkl)
    for dataset, ldf in landmarks_df.groupby('dataset'):
        sax = np.vstack(ldf.symmetry_axis)
        print(dataset, sax.mean(axis=0), sax.std(axis=0))
        # CB1 [ 509.68746066  199.54171642  113.66635353] [  2.65145381  79.66099853  34.78462986]
        # T1 [ 387.72334883  351.53474212   76.53648231] [   4.01389088  107.25527533   30.06527197]
        # So as a simple-minded approach, we could use x=510 (for CB1) and x=388 (for T1) as symmetry planes
        # These do barely correspond to size_x / 2

    # --- If we ever want to run something like procrustes, or other superimposition technique...
    # def procrustes_poc():
    #     from scipy.linalg import orthogonal_procrustes
    #     from scipy.spatial import procrustes
    #     for (dataset, landmark), dldf in landmarks_df.groupby(['dataset', 'landmark']):
    #         R, scale = orthogonal_procrustes(np.vstack(dldf.left),
    #                                          np.vstack(dldf.right))
    #         mtx1, mtx2, disparity = procrustes(np.vstack(dldf.left),
    #                                            np.vstack(dldf.right))


# Assume that everything is symmetric
# Maybe the assumption holds better in the Janelia dataset than in the T1 dataset
# Apparently Jefferis and co. did try to symmetrize the whole template

# Other simple landmarks:
#  - region medoid
#  - farthest from center on each direction

