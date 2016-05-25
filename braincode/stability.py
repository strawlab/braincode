from __future__ import print_function, absolute_import

import tempfile
from glob import glob
from itertools import product, combinations
import os.path as op
import os
from subprocess import check_call, check_output
from time import time

import numpy as np
import pandas as pd
import h5py
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.metrics.cluster.unsupervised import silhouette_score
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.utils.extmath import row_norms

from braincode.util import ExpressionDataset, ensure_dir, neuropil_dir, get_all_datasets, get_all_neuropils
from braincode.dice import dicedist
from braincode.thirdparty.kmedoids_salspaugh import cluster
from braincode.thirdparty.evaluate_gala import vi
from braincode.perform_clustering import save_to_nrrd_xyz
import braincode.util as util

CLUSTER_RUNS_CACHE_PATH = '/mnt/strawscience/santi/andrew/clustering-paper/clusterings-first-submission'


def viinv(x, y, normalise=True):
    # I found this looking for a python implementation of "variation of information"
    # It looks really neat
    # https://github.com/janelia-flyem/gala
    if normalise:
        return -vi(x, y, ignore_x=[], ignore_y=[]) / np.log2(len(x))
    return -vi(x, y, ignore_x=[], ignore_y=[])


# Some clusterers with clumsy common API
def mbk(X, D, k, seed):
    clusterer = MiniBatchKMeans(n_clusters=k,
                                init='k-means++',
                                n_init=3,
                                max_iter=1000,
                                batch_size=1000,
                                random_state=seed)
    clusterer.fit(X)
    return clusterer.labels_, clusterer.cluster_centers_


def kmeans(X, D, k, seed):
    clusterer = KMeans(n_clusters=k,
                       init='k-means++',
                       max_iter=1000,
                       n_init=10,  # repeat 10 times, use best clustering according to inertia
                       n_jobs=4,
                       random_state=seed,
                       precompute_distances=True)
    clusterer.fit(X)
    return clusterer.labels_, clusterer.cluster_centers_


def kmedoids(X, D, k, seed):
    # k-medoids from salspaugh (it converges very fast for these data)
    return cluster(D, k=k, seed=seed)


def kmeanspp(X, k, seed):
    # That we need to do this is a bug in _init_centroids
    x_squared_norms = row_norms(X, squared=True)
    # Use k-means++ to initialise the centroids
    centroids = _init_centroids(X, k, 'k-means++', random_state=seed, x_squared_norms=x_squared_norms)
    # OK, we should just short-circuit and get these from k-means++...
    # quick and dirty solution
    nns = NearestNeighbors()
    nns.fit(X)
    centroid_candidatess = nns.radius_neighbors(X=centroids, radius=0, return_distance=False)
    # Account for "degenerated" solutions: serveral voxels at distance 0, each becoming a centroid
    centroids = set()
    for centroid_candidates in centroid_candidatess:
        centroid_candidates = set(centroid_candidates) - centroids
        if len(set(centroid_candidates) - centroids) == 0:
            raise Exception('Cannot get an unambiguous set of centers;'
                            'theoretically this cannot happen, so check for bugs')
        centroids.add(centroid_candidates.pop())
    return np.array(sorted(centroids))


def kmedoids_kpp(X, D, k, seed):
    # k-medoids from salspaugh (it converges very fast for these data)
    return cluster(D, k=k, seed=seed, centroids=kmeanspp(X=X, k=k, seed=seed))


def kmedoids_julia(X, D, k, seed, tmpdir=None, julia_exe=None):
    """Run kmedoids using Julia's Clustering.jl.

    Preparation steps
      - install julia (pacman -S / apt-get install julia)
      - Install Clustering: julia -e 'Pkg.add("Clustering")'
      - Install HDF5: julia -e 'Pkg.add("HDF5")'
      - use this function

    https://github.com/JuliaStats/Clustering.jl
    https://github.com/JuliaLang/HDF5.jl

    Implementation notes
    --------------------
    pyjulia does not work at the moment with recent Julia versions (and development seem slow)
      - fails with arch's Julia (newer Julia and one needs to change some deprecated stuff in julia/core.py)
      - if using conda julia, the julia installation is kinda broken and cannot install packages from source
    In any case looking at the code, it does not seem pyjulia to be very clever on what it does
    (for example, it could use some child-processes-share-memory-in-linux or mmaps to pipe info without copying)

    So for the time being, we can just use files to talk to Julia:
    (note we are doubling the amount of memory needed and using disk for communication,
     but that should be fine in strz)
    """
    if tmpdir is not None:
        ensure_dir(tmpdir)
    path_in = tempfile.mktemp(prefix='vpn-clustering-in-', suffix='.h5', dir=tmpdir)
    path_out = tempfile.mktemp(prefix='vpn-clustering-out', suffix='.h5', dir=tmpdir)

    # Save distance matrix to temporary file (we might want to reuse...)
    with h5py.File(path_in) as h5:
        h5['distances'] = D

    # Let's assume that he used all parameters as default (clustering anyway converges quickly)
    # Unfortunatelly people keep running stuff using system's rng seed
    # So I guess Florian did do that too

    julia_program = """
    import HDF5
    import Clustering
    srand(%d);
    D = HDF5.h5read("%s", "distances");
    result = Clustering.kmedoids(D, %d);
    HDF5.h5write("%s", "/labels", result.assignments);
    HDF5.h5write("%s", "/medoids", result.medoids);
    """ % (seed, path_in, k, path_out, path_out)

    if julia_exe is None:
        julia_exe = check_output(['which','julia']).strip()
    print('using Julia at %r' % julia_exe)
    if 0 != check_call([julia_exe, '-e', julia_program.strip()]):
        raise Exception('Julia error (you need to debug, sorry)')

    with h5py.File(path_out) as h5:
        labels, medoids = h5['labels'], h5['medoids']
        os.unlink(path_in)
        os.unlink(path_out)
        return labels[()], medoids[()]


def run_one(path,
            cache_path=CLUSTER_RUNS_CACHE_PATH,
            clusterers=(kmedoids_julia, kmedoids_kpp, kmedoids),
            distance=dicedist,
            repeats=tuple(range(10)),
            ks=tuple(range(2, 30) + range(30, 151, 5)),
            recompute=False):
    """Ad-hoc function to compute many partitions for varying clusterer, k, seed."""

    # TODO: make generic, whatamise, more fine-grained cache

    dset = ExpressionDataset(path=path)
    # print('DSET:', dset.expression_dset, dset.neuropil)
    # print('Lines:', dset.lines())
    # print('Voxels:', dset.voxels().tolist())
    X = None
    D = None

    for clusterer, k, repeat in product(clusterers, ks, repeats):

        needs_D = clusterer.__name__.startswith('kmedoids')  # ad-hoc

        result_id_dict = {
            'dset': dset.expression_dset,
            'neuropil': dset.neuropil,
            'clusterer': clusterer.__name__,
            'k': k,
            'distance': distance.__name__ if needs_D else None,
            'repeat': repeat,
        }

        result_id = ','.join('%s=%s' % (key, value) for key, value in sorted(result_id_dict.items()))
        assert len(result_id) < 254, 'A file name length cannot exceed 254 characters in ext4, hash'
        result_cache_path = op.join(ensure_dir(cache_path), '%s.pkl' % result_id)

        if not op.isfile(result_cache_path) or recompute:

            if X is None:
                X = dset.Xarray()  # N.B., we use dense, so account for 1-10GB of memory per run
            print(result_id)
            # print('%d voxels, %d lines' % X.shape)

            #
            # print('Preprocessing')
            # PCA before clustering/visualisation might be useful in this dataset
            # For the usual reasons PCA is usually very useful:
            #  - decorrelates
            #  - removes noise
            #  - reduces dimensionality
            #    (while keeping, most of the time, the essence of the dataset)
            #
            # pca = PCA(n_components=0.95)  # or pca = TruncatedSVD() for sparse, iterative SVD
            # X = pca.fit_transform(X)
            #

            if needs_D and D is None:
                print('Distance computation...')
                start = time()
                D = distance(X)
                print('\tTaken %.2f seconds to compute a %d x %d distance matrix' %
                      (time() - start, D.shape[0], D.shape[1]))
            #
            # Other options...
            # Euclidean, for example, if we do PCA or do not use thresholds...
            # D = pairwise_distances(X, metric='euclidian', n_jobs=4)
            # D = square_form(pdist(X, metric='dice'))
            # print(D.shape)
            #

            start = time()
            labels, centers = clusterer(X, D, k, repeat)

            clustering = {
                'labels': labels,
                'centers': np.array(centers),
                'taken_s': time() - start,
            }
            clustering.update(result_id_dict)

            pd.to_pickle(clustering, result_cache_path)


def compute_all(cache_path=CLUSTER_RUNS_CACHE_PATH, repeats=tuple(range(10)), ks=(40, 60)):
    for dset_path in ExpressionDataset.all_datasets():
        run_one(path=dset_path, cache_path=cache_path, repeats=repeats, ks=ks)


def mean_sim(labels, cluster_sim=metrics.adjusted_rand_score):
    # Use viinv, metrics.adjusted_rand_score, metrics.adjusted_mutual_info_score... as cluster_sim
    return np.array([cluster_sim(c1, c2) for c1, c2 in combinations(labels, 2)]).mean()


def read_clusterings_cache(dset='CB1', neuropil='Optic_Glomeruli', basedir=CLUSTER_RUNS_CACHE_PATH,
                           drop_centers=False,
                           categoricals=('dset', 'neuropil', 'distance', 'clusterer')):
    all_pkls = glob(op.join(basedir, '*dset=%s*neuropil=%s*.pkl' % (dset, neuropil)))
    if 0 == len(all_pkls):
        return None
    df = pd.DataFrame([pd.read_pickle(pkl) for pkl in all_pkls])
    for categorical in categoricals:
        df[categorical] = df[categorical].astype('category')
    if drop_centers:
        del df['centers']
    return df


def read_all_clusterings_cache(drop_centers=True, recompute=False,
                               categoricals=('dset', 'neuropil', 'distance', 'clusterer')):
    cache_file = op.join(CLUSTER_RUNS_CACHE_PATH, 'all#with_centers=%r.pkl' % (not drop_centers))
    if recompute or not op.isfile(cache_file):
        dfs = [read_clusterings_cache(dset, neuropil, drop_centers=drop_centers, categoricals=())
               for dset, neuropil in product(get_all_datasets(), get_all_neuropils())]
        dfs = [df for df in dfs if df is not None]
        df = pd.concat(dfs)
        for categorical in categoricals:
            df[categorical] = df[categorical].astype('category')
        pd.to_pickle(df, cache_file)
    try:
        return pd.read_pickle(cache_file)
    except:  # quick and dirty account for old pandas versions
        return read_all_clusterings_cache(drop_centers=drop_centers, recompute=True,)


def save_kmedoid_dice_clusterings_to_nrrd(basedir=None, clusterer='kmedoids_kpp'):
    drop_centers = True
    df = read_all_clusterings_cache(drop_centers=drop_centers)
    df = df[df['distance'] == 'dicedist']
    df = df[df['clusterer'] == clusterer]
    for (dset, neuropil, k), gdf in df.groupby(['dset', 'neuropil', 'k']):
        filenames = util.get_filenames(dset, neuropil, basedir=basedir)
        hub = ExpressionDataset(neuropil_dir(dset, neuropil, basedir=basedir))
        voxels = hub.voxels()
        input_h5_fname = filenames['h5']
        for row_enum, row in gdf.iterrows():
            k = row['k']
            labels = row['labels']
            cluster_type = 'stability_K%03d_dicedist_%03d' % (k, row['repeat'])
            this_filenames = util.get_filenames(dset, neuropil, cluster_type=cluster_type, basedir=basedir)
            nrrd_fname = this_filenames['clustering_result_nrrd']
            if op.exists(nrrd_fname):
                raise RuntimeError('will not overwrite %r' % nrrd_fname)
            print('saving %r' % nrrd_fname)
            # save_to_nrrd(input_h5_fname, labels, nrrd_fname)
            save_to_nrrd_xyz(input_h5_fname, labels, voxels, nrrd_fname)


def relabel_to_0k(labels):
    l2i = {}

    def l2i_(l):
        if l not in l2i:
            l2i[l] = len(l2i)
        return l2i[l]

    return list(map(l2i_, labels))


if __name__ == '__main__':

    def clustering_distance_example():
        df = read_all_clusterings_cache(drop_centers=True)

        df.info()
        print(df.dset.unique())
        print(df.clusterer.unique())
        print(df.neuropil.unique())
        print(df.k.unique())

        medoids_df = df.query('clusterer == "kmedoids_julia"')
        grouped = medoids_df.groupby(['dset','neuropil','k'])

        fname = op.join(CLUSTER_RUNS_CACHE_PATH,'ARI.pkl')
        print('computing Adjusted Rand Index, will save to',fname)
        ARI_series = grouped['labels'].agg(mean_sim)
        ARI_series.to_pickle(fname)

    def silhouette_example(dset='CB1', neuropil='Optic_Glomeruli', seed=0, clusterer='kmedoids'):
        print('Reading dataset')
        X = ExpressionDataset(neuropil_dir(dset, neuropil)).Xarray()
        print('Reading clusterings')
        df = read_clusterings_cache(dset=dset, neuropil=neuropil)
        # Pick just one seed (you might want to change this...)
        df = df.query('repeat==%d and clusterer=="%s"' % (seed, clusterer))
        # Compute all the silhouettes
        for k, kdf in df.groupby('k'):
            print('Computing mean silhoutte for k=%d' % k)
            assert len(kdf) == 1
            # sklearn expects labels to be in [0...k-1]
            labels = np.array(relabel_to_0k(kdf['labels'].iloc[0]))
            print(k, silhouette_score(X, labels, metric='euclidean', random_state=0))  # sample_size=1000

    compute_all(cache_path=CLUSTER_RUNS_CACHE_PATH, ks=range(2, 30) + range(30, 151, 5))
    # clustering_distance_example()
    # silhouette_example()
    # save_kmedoid_dice_clusterings_to_nrrd()
    print('Done')

# Do also X.T: the more usual clustering of lines
