from __future__ import print_function

import glob
import os
import os.path as op
from array import array
from collections import Counter, MutableMapping
from itertools import product, chain
from operator import itemgetter

import h5py
import nrrd
import numpy as np
import pandas as pd
import glob
import json
import datetime
from scipy.sparse import csr_matrix

from braincode.dice import dicesim_from_dot

def ensure_writable_dir(path):
    """Ensures that a path is a writable directory."""
    def check_path(path):
        if not op.isdir(path):
            raise Exception('%s exists but it is not a directory' % path)
        if not os.access(path, os.W_OK):
            raise Exception('%s is a directory but it is not writable' % path)
    if op.exists(path):
        check_path(path)
    else:
        try:
            os.makedirs(path)
        except Exception:
            if op.exists(path):  # Simpler than using a file lock to work on multithreading...
                check_path(path)
            else:
                raise
    return path


def ensure_dir(path):
    return ensure_writable_dir(path)


def voxel_name_to_idx(vn):
    """convert a voxel name to numerical values

    >>> vn = "231_38_94"
    >>> voxel_name_to_idx(vn)
    [231, 38, 94]
    """
    idxs = vn.split('_')
    return map(int, idxs)

voxelid2coords = voxel_name_to_idx


def get_all_datasets():
    return ['CB1', 'T1']


def get_all_neuropils():
    return ['Optic_Glomeruli', 'Antennal_lobe', 'Central_Complex',
            'Mushroom_Body', 'SEZ']


def get_finished_cluster_types(dataset, neuropil, basedir=None):
    base_filenames = get_filenames(dataset, neuropil, basedir=basedir)
    h5_fname = base_filenames['h5']
    basedir = os.path.dirname(h5_fname)
    dirnames1 = glob.glob(os.path.join(basedir, 'K*_dicedist*'))
    dirnames2 = glob.glob(os.path.join(basedir, 'K*_mbk1_seed1'))
    dirnames = dirnames1 + dirnames2
    if len(dirnames) == 0:
        raise ValueError('no finished clusters')
    cluster_types = [os.path.split(dirname)[1] for dirname in dirnames]
    return cluster_types


def braincode_basedir(basedir=None):
    if basedir is None:
        basedir = os.environ.get('BRAINCODE_BASEDIR', None)
    if basedir is None:
        # relative to this file's location
        basedir = op.abspath(op.join(__file__, '..', '..', 'clustering-data'))
    return basedir


def dataset_dir(dataset, basedir=None):
    return op.join(braincode_basedir(basedir), dataset)


def neuropil_dir(dataset, neuropil, basedir=None):
    return op.join(dataset_dir(dataset, basedir=basedir), neuropil)


def get_filenames(dataset, neuropil, cluster_type=None, basedir=None, distances_base_dir=None):

    basedir = braincode_basedir(basedir)

    if distances_base_dir is None:
        distances_base_dir = os.environ.get('BRAINCODE_DISTANCES_CACHE', None)
    if distances_base_dir is None:
        distances_base_dir = basedir

    # There is only one of these files per dataset (CB1 and T1).
    result = {'id_driver_image_csv': os.path.join(basedir,
                                                  dataset,
                                                  'id_driver_image.csv')}

    source_data_dir = os.path.join(basedir, dataset, neuropil)
    distances_dir = os.path.join(distances_base_dir, dataset, neuropil)

    result['h5'] = os.path.join(source_data_dir, '%s_samples.h5' % (neuropil,))
    result['distance_npy'] = os.path.join(distances_dir, '%s_distance.npy' % (neuropil,))
    if cluster_type is not None:
        clustering_results_dir = os.path.join(source_data_dir, cluster_type)
        result['clustering_result_nrrd'] = os.path.join(clustering_results_dir, '%s_clusterimage.nrrd' % (neuropil,))
        result['fragment_info_raw_csv'] = os.path.join(clustering_results_dir,
                                                       'fragment_info_%s_%s_%s.csv' % (dataset, neuropil, cluster_type))
        result['fragment_info_filtered_csv'] = os.path.join(clustering_results_dir,
                                                            'fragment_info_filtered_%s_%s_%s.csv' %
                                                            (dataset, neuropil, cluster_type))
        result['cluster_info_json'] = os.path.join(clustering_results_dir,
                                                   'cluster_info_%s_%s_%s.json' % (dataset, neuropil, cluster_type))
        result['cluster_stats_intra_csv'] = os.path.join(clustering_results_dir, 'cluster_stats_intra_%s_%s_%s.csv' % (
            dataset, neuropil, cluster_type))
        result['cluster_stats_inter_csv'] = os.path.join(clustering_results_dir, 'cluster_stats_inter_%s_%s_%s.csv' % (
            dataset, neuropil, cluster_type))
        result['medoids_distance_npy'] = os.path.join(clustering_results_dir, 'medoids_distance_%s_%s_%s.npy' % (
            dataset, neuropil, cluster_type))
        result['medoids_quality_png'] = os.path.join(clustering_results_dir,
                                                     'medoids_quality_%s_%s_%s.png' % (dataset, neuropil, cluster_type))
        result['medoids_quality_svg'] = os.path.join(clustering_results_dir,
                                                     'medoids_quality_%s_%s_%s.svg' % (dataset, neuropil, cluster_type))

        result['full_base'] = '%s_%s_%s' % (dataset, neuropil, cluster_type)
    return result


def h5_coord_to_nrrd(vals, step):
    return np.round(vals / float(step)).astype(np.int) - 1  # transformed to nrrd coordinate


def calculate_nrrd_to_h5_LUTs(h5_data):
    """compute a large look-up-table (LUT) to convert downsampled NRRD coordinates to original position_keys"""
    all_voxel_names = np.array(h5_data['positionKeys'][()].strip(',').split(','))
    stepXY = h5_data['stepXY'][()][0]
    stepZ = h5_data['stepZ'][()][0]

    idxs = np.array([voxel_name_to_idx(vn) for vn in all_voxel_names])
    xs = np.unique(idxs[:, 0])
    ys = np.unique(idxs[:, 1])
    zs = np.unique(idxs[:, 2])

    rs = h5_coord_to_nrrd(xs, stepXY)
    ss = h5_coord_to_nrrd(ys, stepXY)
    ts = h5_coord_to_nrrd(zs, stepZ)

    xidx = {}
    yidx = {}
    zidx = {}
    for r, x in zip(rs, xs):
        xidx[r] = x

    for s, y in zip(ss, ys):
        yidx[s] = y

    for t, z in zip(ts, zs):
        zidx[t] = z

    return xidx, yidx, zidx


def get_original_locations_from_cluster_id(cluster_id, nrrd_data, luts):
    """given a cluster_id, return all locations belonging

    The locations are a zero-indexed location into the original
    (non-downsampled) dataset.
    """
    result = []
    location_cond = nrrd_data == cluster_id
    locations = np.nonzero(location_cond)
    for (loc0, loc1, loc2) in zip(*locations):
        pk0 = luts[0][loc0]
        pk1 = luts[1][loc1]
        pk2 = luts[2][loc2]
        result.append((pk0, pk1, pk2))
    return result


def get_position_keys_from_cluster_id(cluster_id, nrrd_data, luts):
    """given a cluster_id, return all position_keys belonging"""

    def loc2key(location):
        """convert a location (zero-indexed) to a position key (one-indexed)"""
        key = '_'.join(['%s' % (loc,) for loc in location])
        return key

    tmp = get_original_locations_from_cluster_id(cluster_id, nrrd_data, luts)
    result = [loc2key(el) for el in tmp]
    return result


def get_k_from_cluster_type(cluster_type):
    # Assume all of these are of the form Kk_whatever
    # We actually do not need k for anything else than sanity checking...
    # Some whatami contracts could have been useful here.
    return int(cluster_type.partition('_')[0][1:])


def get_original_clustering(dataset='CB1', neuropil='Antennal_lobe', clusterer_or_k=60):
    """
    Returns two pandas dataframes (clusters_df, medoids_df) with the original k-medoids clusterings.
      - clusters_df containes one cluster per row with
    The dataframes also contains provenance information to link the "downsampled"
    space (where clustering was computed) and the "original" image space
    (where we usually render the results).

    These are the original clusterings in the paper supplementary,
    computed by Florian and explored for biological meaning in detail
    by Karin, Laszlo and co.

    Historical note: Florian curiously provided upsampled voxel IDs for its
    downsampled sparse expression matrix, while only provided the final cluster
    assignments in a downsampled nrrd image file. Also I think he numbered voxels
    and clusters using 1-based numbering.
    This uses the downsampled->original and 0->1->0 functions Andrew hacked.
    These are not fully correct (if fully correctness can be achieved).
    They should be good enough, though, and they should at least map properly
    downsampled coordinates in the nrrd to the provided original coordinates
    in the expression data matrix.
    Using them here will also help to avoid any divergence from the already reported results.

    Parameters
    ----------
    dataset : string, default 'CB1'
      The expression dataset id, one of 'CB1' or 'T1' at the moment.

    neuropil : string, default 'Antennal_lobe'
      The neuropil where the clustering was done.

    clusterer_or_k : string or int, default 60
      If a string, this should correspond to a cluster_type as returned by `get_finished_cluster_types`.
      If an integer, we assume it is the number of clusters k and that the cluster type is
      'K%d_dicedist' % k.

    Returns
    -------
    A two tuple (clusters_df, medoids_df).
    - clusters_df is a pandas dataframe with k rows (one per cluster) and the following columns:
      - dataset: a constant string to identify the dataset that is being clustered
      - neuropil: a constant string to identify the neuropil that is being clustered
      - clusterer: a constant string identifying the clustering implementation used to infer clusters
      - cluster_id: an integer in [1, k] arbitrarily identifying the cluster
      - original_medoid_id: a string "x_y_z" identifying the medoid voxel in the original image
      - downsampled_medoid_id: a string "x_y_z" identifying the medoid voxel in the downsampled image
      - original_voxels_in_cluster: a list of strings "x_y_z" with all the voxels in the cluster, original space
        Note that these are not all the voxels that correspond to the cluster,
        just one representative per volume (or "clustered metavoxel").
      - downsampled_voxels_in_cluster: a list of strings "x_y_z" with all the voxels in the cluster, downsampled space
      - downsampled_medoid: A pandas Series with the expression vector of the voxel in the downsampled space.
        The index correspond to each line brainbase database id.
      - stepX, stepY, stepZ: constant ratios on how many voxels in the original space correspond to a voxel
        in the downsampled space.

    - medoids_df can be inferred from clusteres_df.
     It contains a cluster per row and the expression of each line in columns.
     Expression is in the (binarised, opened, downsampled) space.
     The index is cluster_id and the columns are named after the line brainbase database id.

    See also
    --------
    `braincode.calculate_cluster_stats.py`
    """
    # TODO: this should be cached and/or stored in a better format

    # Load the original data matrix, as a dense pandas dataframe
    # Note that even if the voxels are indexed from the original voxel id,
    # these are expression data in the (binarised, downsampled, thresholded) space.
    Xdownsampled = ExpressionDataset.dataset(dset=dataset, neuropil=neuropil).Xdf(index_type='string')

    # Also use some of Andrew get_me_stuff functions
    if isinstance(clusterer_or_k, int):
        k = clusterer_or_k
        cluster_type = 'K%d_dicedist' % k
    else:
        cluster_type = clusterer_or_k
        k = get_k_from_cluster_type(cluster_type)
    filenames = get_filenames(dataset, neuropil, cluster_type=cluster_type)

    with h5py.File(filenames['h5'], mode='r') as data:

        # Downsampled to original
        xdown2xorig, ydown2yorig, zdown2zorig = calculate_nrrd_to_h5_LUTs(data)
        stepX = stepY = data['stepXY'][0]
        stepZ = data['stepZ'][0]

        # Cluster assignments (labels) in the downsampled space
        cluster_assignments, _ = nrrd.read(filenames['clustering_result_nrrd'])
        cluster_ids = np.unique(cluster_assignments.ravel())
        assert len(cluster_ids) == k + 1
        assert cluster_ids[0] == 0
        assert cluster_ids[-1] == k
        cluster_ids = cluster_ids[1:]  # 0 means "no cluster"

        clustering_results = []
        for cluster_id in sorted(cluster_ids):
            # Find cluster members
            downsampled_voxels_in_cluster = np.vstack(np.where(cluster_assignments == cluster_id)).T
            original_voxels_in_cluster_ids = ['{x}_{y}_{z}'.format(x=xdown2xorig[xdown],
                                                                   y=ydown2yorig[ydown],
                                                                   z=zdown2zorig[zdown])
                                              for xdown, ydown, zdown in downsampled_voxels_in_cluster]
            downsampled_voxels_in_cluster_ids = ['{x}_{y}_{z}'.format(x=xdown, y=ydown, z=zdown)
                                                 for xdown, ydown, zdown in downsampled_voxels_in_cluster]
            Xcluster = Xdownsampled.loc[original_voxels_in_cluster_ids]

            # Find the medoid
            original_medoid_id = dicesim_from_dot(Xcluster, nan_to_one=True).sum().idxmax()
            downsampled_medoid_id = downsampled_voxels_in_cluster_ids[
                original_voxels_in_cluster_ids.index(original_medoid_id)]

            # Add the result to the table
            clustering_results.append(dict(
                dataset=dataset,
                neuropil=neuropil,
                clusterer='OriginalKMedoids_' + cluster_type,  # pfff
                cluster_id=cluster_id,
                original_medoid_id=original_medoid_id,
                downsampled_medoid_id=downsampled_medoid_id,
                original_voxels_in_cluster=original_voxels_in_cluster_ids,
                stepX=stepX,
                stepY=stepY,
                stepZ=stepZ,
                downsampled_voxels_in_cluster=downsampled_voxels_in_cluster_ids,
                downsampled_medoid=Xcluster.loc[original_medoid_id],

            ))

        clusters_df = pd.DataFrame(clustering_results)
        medoids_df = pd.DataFrame({cluster_id: medoid
                                   for cluster_id, medoid in zip(clusters_df.cluster_id,
                                                                 clusters_df.downsampled_medoid)}).T

        return clusters_df, medoids_df


def vt_name_to_bbweb_url(name):
    num = vt_name_to_num(name)
    return 'https://strawlab.org/fly-enhancer-redirect/v1/bbweb?vt=' + '%05d' % num


def vt_name_to_vdrc_url(name):
    num = vt_name_to_num(name)
    return 'https://strawlab.org/fly-enhancer-redirect/v1/vdrc?vt=' + '%05d' % num


def janelia_name_to_flylight_url(name):
    line = janelia_name_to_line(name)
    return 'https://strawlab.org/fly-enhancer-redirect/v1/flylight?line=' + line


def janelia_name_to_line(orig):
    assert orig.startswith('GMR')
    line = orig[2:]
    return line


def vt_name_to_num(orig):
    assert orig.startswith('VT')
    name = orig[2:]
    assert name.endswith('.GAL4@attP2')
    nums = name[:-11]
    num = int(nums, 10)
    assert orig == 'VT' + nums + '.GAL4@attP2'
    return num


def generate_set_cache(voxel_names, ids):
    set_cache = []
    for pk in voxel_names:
        tmp = ids[pk]
        if len(tmp.shape) == 0:
            set_cache.append(set())
        else:
            set_cache.append(set(tmp[()]))
    return set_cache


def dataset2dense(dataset, neuropil, basedir=None):
    """Reads a expression dataset into a numpy array.
    Inspired in calculate_cluster_stats.py, to test ExpressionDataset.
    """
    filenames = get_filenames(dataset, neuropil, basedir=basedir)
    data = h5py.File(filenames['h5'], mode='r')

    voxel_names = data['positionKeys'][()].strip(',').split(',')

    print('%r: %d positions' % (filenames['h5'], len(voxel_names)))
    ids = data['ids']
    N = len(voxel_names)

    # read sparse data
    set_cache = generate_set_cache(voxel_names, ids)
    # make a list of all driver ids
    all_drivers = set()
    for sc in set_cache:
        all_drivers.update(sc)

    all_drivers = list(all_drivers)
    all_drivers.sort()
    M = len(all_drivers)
    print('%d drivers' % M)

    print('calculating dense matrix of size %d x %d' % (N, M))
    dense_matrix = np.zeros((N, M), dtype=np.uint8)
    for i in range(N):
        sc = set_cache[i]
        for driver in sc:
            j = all_drivers.index(driver)
            dense_matrix[i, j] = 1

    return dense_matrix, all_drivers, voxel_names


class ExpressionDataset(object):
    """Data hub"""

    def __init__(self, path):
        super(ExpressionDataset, self).__init__()
        self.path = path
        self.expression_dset = op.basename(op.dirname(path))  # CB1, T1...
        self.neuropil = op.basename(path)  # e.g. Antennal_lobe
        self._samples_h5 = op.join(self.path, '%s_samples.h5' % self.neuropil)
        self._voxels = None
        self._lines = None
        self._X = None

    def _read_csr(self):
        """Converts and stores the data in the HDF5 files into a sparse matrix."""

        with h5py.File(self._samples_h5, mode='a') as h5:
            if 'csr' not in h5:
                print('Caching %s %s' % (self.expression_dset, self.neuropil))
                # Read original data into CSR
                indptr = array('i', [0])
                indices = array('i')
                voxels = []
                lines = {}

                def line2i(line):
                    if line not in lines:
                        lines[line] = len(lines)
                    return lines[line]

                for i, (voxel, on_lines) in enumerate(h5['ids'].items()):
                    if len(on_lines.shape) > 0:  # never expressed voxel (happens in T1/SEZ)
                        indices.extend(map(line2i, on_lines[()]))
                        indptr.append(indptr[-1] + len(on_lines))
                    else:
                        indptr.append(indptr[-1])
                    voxels.append(map(int, voxel.split('_')))

                # Cache to the hdf5 file
                csr = h5.create_group('csr')
                csr['voxels'] = np.array(voxels)  # dtype=np.intx
                csr['lines'] = np.array(list(map(itemgetter(0),
                                                 sorted(lines.items(), key=itemgetter(1)))))  # dtype=np.intx
                csr['indptr'] = indptr
                csr['indices'] = indices
                # if we could have something other than 1 or 0
                # (e.g. we would have not binarised the intensities)
                # we could save also need to save the data
                # csr['data'] = blah

            csr = h5['csr']
            self._voxels = csr['voxels'][:]
            self._lines = csr['lines'][:]
            indices, indptr = (csr['indices'][:], csr['indptr'][:])
            data = np.ones(len(indices), dtype=np.float32)  # note float32
            self._X = csr_matrix((data, indices, indptr),
                                 shape=(len(self._voxels), len(self._lines)))

    #
    # The dataset is small and it repays to have it represented as float32
    # It fits comfortably in RAM even if we represent it as dense.
    # So we can switch from dense to sparse representation to use whatever makes things faster or possible
    #

    def Xcsr(self):
        """Returns a dense num_voxels x num_lines csr matrix."""
        if self._X is None:
            self._read_csr()
        return self._X

    def Xdense(self):
        """Returns a num_voxels x num_lines dense scipy matrix."""
        return self.Xcsr().todense()

    def Xarray(self):
        """Returns a num_voxels x num_lines dense numpy array."""
        return self.Xcsr().toarray()

    def Xdf(self, index_type='multilevel'):
        """Returns a num_voxels x num_lines dense pandas DataFrame."""
        if index_type == 'multilevel':
            index = pd.MultiIndex.from_arrays(self.voxels().T, names=['x', 'y', 'z'])
        elif index_type == 'tuple':
            index = map(tuple, self.voxels())
        elif index_type == 'string':
            index = self.voxel_ids()
        else:
            raise ValueError('Possible values for index_type are ["multilevel", "string"], '
                             'not "%s"' % index_type)
        return pd.DataFrame(self.Xarray(),
                            index=index,
                            columns=self.lines(),
                            copy=False)

    def voxels(self):
        """Returns a numpy array with a row per voxel and ('x', 'y', 'z') columns."""
        # FIXME: document if this is 1-based or 0-based; make it 0-based if needed.
        if self._voxels is None:
            self._read_csr()
        return self._voxels

    def voxel_ids(self):
        """Returns a list with a string 'x_y_z' per voxel."""
        return ['_'.join(map(str, voxel)) for voxel in self.voxels()]

    def lines(self):
        """Returns a numpy array with the database id of the lines in the dataset."""
        if self._lines is None:
            self._read_csr()
        return self._lines

    def densities(self):
        df = self.Xdf()
        # How many voxels are "on" (in the region) for each line?
        # Note that because of our visualisation-oriented preprocessing,
        # this should in principle be 0.01 when use the whole brain images.
        # When selecting a region, it can be anything in [0, 1].
        voxels_per_line = df.sum(axis=0) / df.shape[1]
        # For how many lines was a voxel "on"? This is in [0, 1]
        lines_per_voxel = df.sum(axis=1) / df.shape[0]
        return voxels_per_line, lines_per_voxel

    @staticmethod
    def all_datasets(basedir=None):
        return [op.join(braincode_basedir(basedir), dset, neuropil)
                for dset, neuropil in product(get_all_datasets(), get_all_neuropils())]

    @staticmethod
    def dataset(dset='T1', neuropil='Central_Complex', basedir=None):
        return ExpressionDataset(op.join(braincode_basedir(basedir), dset, neuropil))


def test_expression_dset_load(dset='T1', neuropil='Central_Complex', basedir=None):
    expectedX, lines, voxels = dataset2dense(dset, neuropil)
    edset = ExpressionDataset.dataset(dset=dset, neuropil=neuropil, basedir=basedir)
    assert set(edset.lines()) == set(lines)
    assert set(edset.voxel_ids()) == set(voxels)
    assert expectedX.shape == edset.Xarray().shape
    # Take care of ordering issues... easier with pandas
    expectedX = pd.DataFrame(expectedX, columns=lines, index=voxels, copy=False)
    expectedX = expectedX.loc[edset.voxel_ids()][edset.lines()]
    # Make sure they are the same
    np.testing.assert_array_equal(edset.Xarray(), expectedX.values)
    print('No problems loading', dset, neuropil)

def get_csv_metadata(fname):
    with open(fname) as fd:
        first_line = fd.readline().strip()
    assert first_line.startswith('#')
    first_line = first_line[1:].strip()
    metadata = json.loads(first_line)
    if 'analysis_time' in metadata:
        metadata['analysis_time_parsed'] = datetime.datetime.strptime(metadata['analysis_time'],'%Y-%m-%dT%H:%M:%S.%f')
    return metadata

def check_disjoint(*dicts):
    repeated = sorted(key for key, counts in Counter(chain.from_iterable(*dicts)).items() if counts > 1)
    if 0 < len(repeated):
        raise ValueError('There are repeated keys: %r' % repeated)


def mergedicts(*dicts):
    """Merges dictionaries, keys in rightmost dictionaries take precedence when overriding."""
    return {k: v for k, v in chain.from_iterable(d.items() for d in dicts)}


def mergedisjointdicts(*dicts):
    """Merges dictionaries.
    Keys in rightmost dictionaries take precedence when overriding and no repeated keys are allowed.
    """
    check_disjoint(dicts)
    return mergedicts(*dicts)


def flattendict(d, parent_key='', sep='_'):
    """Flattens a dictionary with possible nested dictionaries compressing keys."""
    # Shamelessly stolen from:
    #   http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flattendict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":

    import doctest
    doctest.testmod()

    test_expression_dset_load(dset='T1', neuropil='Central_Complex')
    test_expression_dset_load(dset='T1', neuropil='SEZ')
    test_expression_dset_load(dset='CB1', neuropil='Optic_Glomeruli')
