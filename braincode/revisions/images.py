# coding=utf-8
"""Image processing tools."""
import os.path as op
import time
from collections import OrderedDict
from functools import reduce, partial
from itertools import chain

from scipy.ndimage import zoom

from braincode.revisions.config import NoopI
from config import numba

import h5py
import nrrd
import numpy as np
import pandas as pd
from py_amira_file_reader.read_amira import read_amira


# --- Image IO


class ParanoidImageReader(object):
    """
    Over-the-top, very strict checker for data read out of our CB1 and VT lines amira files.
    To answer Andrew's concerns on (us) missing present data we are not aware of, like "extra channels".
    """
    # FIXME: split this into Image and read_image_paranoid()

    def __init__(self, path):
        super(ParanoidImageReader, self).__init__()
        # Amira file path
        self.path = path
        # Fields
        self.amira_version = None    # Like 2.0
        self.amira_type = None       # Like 'AmiraMesh'
        self.amira_is_binary = None  # True or False
        self.bounding_box = None     # Like [xmin, xmax, ymin, ymax, zmin, zmax]
        self.coord_type = None       # Like 'uniform'
        self.data_type = None        # Like 'byte'
        self.shape = None            # Like [768, 768, 165]
        self.data = None             # The 3D tensor of volumetric data
        self.read_time_s = None      # The time taken to read the amira file
        # Read, check and cache (careful mem leaks!) the data in the amira file
        self._read_check_data()

    def _read_check_data(self):

        # Read the amira file
        start = time.time()
        amira_data = read_amira(self.path)
        self.read_time_s = time.time() - start

        # File size
        self.file_size_mb = op.getsize(self.path) / 1024 ** 2

        # Top level dictionary
        assert isinstance(amira_data, dict)
        assert len(amira_data) == 2

        # Amira file info
        info = amira_data['info']
        assert isinstance(info, dict)
        assert len(info) == 3
        assert set(info) == {'version', 'type', 'is_binary'}
        self.amira_version = info['version']
        self.amira_type = info['type']
        self.amira_is_binary = info['is_binary']

        # The "data" node
        data = amira_data['data']
        assert isinstance(data, list)
        assert len(data) == 4

        # Image size
        assert isinstance(data[0], dict)
        assert len(data[0]) == 1
        assert isinstance(data[0]['define'], dict)
        assert len(data[0]['define']) == 1
        assert isinstance(data[0]['define']['Lattice'], list)
        self.shape = np.array(data[0]['define']['Lattice'])
        assert self.shape.dtype <= np.int
        assert self.shape.shape == (3,)

        # Other parameters
        assert isinstance(data[1], dict)
        assert len(data[1]) == 1
        parameters = data[1]['Parameters']
        assert isinstance(parameters, dict)
        assert len(parameters) == 3 or len(parameters) == 4
        content_text = parameters['Content'][1:-1]       # 1:-1 to remove quotes
        self.bounding_box = np.array(parameters['BoundingBox'])
        assert self.bounding_box.dtype >= np.float
        assert self.bounding_box.shape == (6,)
        self.coord_type = parameters['CoordType'][1:-1]  # 1:-1 to remove quotes
        if len(parameters) == 4:  # CB1
            assert parameters['NRRD0004'] is None

        # Data type?
        assert isinstance(data[2], dict)
        assert len(data[2]) == 1
        assert isinstance(data[2]['Lattice'], dict)
        assert len(data[2]['Lattice']) == 1
        self.data_type = data[2]['Lattice'].keys()[0]
        assert data[2]['Lattice'][self.data_type] == 'Data'

        # Volumetric data
        assert isinstance(data[3], dict)
        assert len(data[3]) == 1
        self.data = data[3]['data']
        assert isinstance(self.data, np.ndarray)

        # Internal consistency checks
        assert (self.data.shape == self.shape).all()
        expected_content_text = self.content_text()
        assert content_text == expected_content_text

    def content_text(self):
        # Like '768x768x165 byte, uniform coordinates'
        return '{xsize}x{ysize}x{zsize} {data_type}, {coord_type} coordinates'.format(
            xsize=self.shape[0],
            ysize=self.shape[1],
            zsize=self.shape[2],
            data_type=self.data_type,
            coord_type=self.coord_type
        )

    def as_dict(self, exclude=('data',), flatten=False):
        if exclude is None:
            exclude = ()
        fields = [field for field in ('path',
                                      'read_time_s',
                                      'file_size_mb',
                                      'amira_version',
                                      'amira_type',
                                      'amira_is_binary',
                                      'bounding_box',
                                      'coord_type',
                                      'data_type',
                                      'shape',
                                      'data')
                  if field not in exclude]
        d = OrderedDict([(field, getattr(self, field)) for field in fields])

        if flatten:
            if 'bounding_box' in d:
                d['xmin'], d['xmax'], d['ymin'], d['ymax'], d['zmin'], d['zmax'] = d['bounding_box']
                del d['bounding_box']
            if 'shape' in d:
                d['xvoxels'], d['yvoxels'], d['zvoxels'] = d['shape']
                del d['shape']

        return d


# --- Functions to manipulate image masks

class VoxelLabels(object):
    """
    Assign a single named label to each voxel of an image.

    Parameters
    ----------
    labels_df : pd.DataFrame
      A dataframe with a label per row and at least the following columns:
        - name: the name of the label
        - id: the numeric id of the region, used in voxel2labelid

    voxel2labelid : numpy array like
      An array containing the single label id for each voxel

    label_groups : dictionary or None
      A dictionary linking "label group name" to a list of labels
    """
    def __init__(self, labels_df, voxel2labelid, label_groups=None):
        super(VoxelLabels, self).__init__()
        assert labels_df.name.nunique() == len(labels_df)
        assert labels_df.id.nunique() == len(labels_df)
        self.labels_df = labels_df
        self.voxel2labelid = voxel2labelid
        self.label_groups = {}
        self.add_label_groups(label_groups)

    def labels(self, *labels):
        """
        Returns a list with all the available "singleton" labels, sorted alphanumerically.

        Parameters
        ----------
        labels : strings
          The names of the labels / label groups for which we want to retrieve the labels.
          If None, all the labels are returned.
          See `VoxelLabels.label_group_names`.
        """
        all_labels = sorted(self.labels_df.name.unique())
        if 0 == len(labels):
            return all_labels
        singleton_labels = sorted(set(chain.from_iterable(self.labels_in_group(name)
                                                          for name in labels)))
        if set(singleton_labels) > set(all_labels):
            raise Exception('Unknown labels %r' % sorted(set(singleton_labels) - set(all_labels)))
        return singleton_labels

    def label_group_names(self):
        """Returns a sorted list of labels group names."""
        return sorted(self.label_groups)

    def labels_in_group(self, label_group_name):
        """Returns the labels in the specified label set."""
        return self.label_groups.get(label_group_name, (label_group_name,))

    def add_labels_group(self, label_group_name, *label_names):
        """Adds a new label group (a name for a group of labels).
        N.B. label redifinition is forbidden ATM.
        """
        # Check that we are not redefining an existing label
        if label_group_name in self.label_groups:
            raise ValueError('The label %s is already in the label groups.')
        if label_group_name in self.labels():
            raise ValueError('The label %s is the name of a singleton label.')
        # Check that all label_names are known
        present_labels = set(self.labels(*label_names))
        label_names = set(label_names)
        wrong_labels = label_names - present_labels
        # Check that we are not overriding
        if len(wrong_labels) > 0:
            raise ValueError('Unknown labels %r' % sorted(wrong_labels))
        try:
            already_present = set(self.label_groups[label_group_name])
            if label_names != already_present:
                raise ValueError('The label set %s is already defined, and different' % label_group_name)
        except KeyError:
            pass
        self.label_groups[label_group_name] = tuple(sorted(label_names))

    def add_label_groups(self, label_groups):
        """Adds label groups from the `label_groups` dictionary {label_name->label_list}."""
        if label_groups is not None:
            for label_group, labels in label_groups.items():
                self.add_labels_group(label_group, *labels)

    def mask(self, *labels):
        """
        Returns a mask with the voxels that pertain to the specified labels "unmasked" (that is, False).

        The mask follows the numpy conventions in which "False" entries indicates that the voxel
        label is in labels (is "unmasked") while "True" indicates that the voxel is
        masked, that is, is not in labels. See:
          http://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html#what-is-a-masked-array

        Parameters
        ----------
        labels : strings
          A list of labels.
          Valid labels are in `self.labels` and possible label groups in `self.label_group_names`.

        Returns
        -------
        The mask, a 3D boolean array of masked voxels.
        """
        labels = self.labels(*labels)
        codes = self.labels_df[self.labels_df.name.isin(labels)].id.unique()
        fast = True
        if fast:
            mask = reduce(lambda acc, code: (self.voxel2labelid == code) | acc,
                          codes,
                          np.zeros_like(self.voxel2labelid, dtype=np.bool))
            return ~mask

        # This is, curiously, way slower than the vanilla implementation...
        return np.in1d(self.voxel2labelid,
                       codes,
                       invert=True,
                       assume_unique=False).reshape(self.voxel2labelid.shape)

    def copy(self, mask=None, downsampler=None):
        """Deep copies this VoxelLabel object, possibly further processing the label assignments.

        Parameters
        ----------
        mask : numpy boolean array
           Masked entries will be assigned to label 0

        downsampler : function(assignments) -> assignments
          A function that maps assignments to a new space.
          For example, a function that (up/down) samples label assignments.
        """
        voxel2labelid = self.voxel2labelid
        if mask is not None:
            voxel2labelid = masked2value(mask, image=voxel2labelid, masked_value=0)
        if downsampler is not None:
            voxel2labelid = downsampler(voxel2labelid)
        if voxel2labelid is self.voxel2labelid:
            voxel2labelid = voxel2labelid.copy()
        return VoxelLabels(labels_df=self.labels_df.copy(),
                           voxel2labelid=voxel2labelid,
                           label_groups=self.label_groups.copy())

    def to_hdf5(self, h5_path, dataset_path='regions', **dataset_opts):
        """
        Writes the region map and metadata to a dataset in an hdf5 file.

        Parameters
        ----------
        h5_path : string
          The path to the hdf5.

        dataset_path : string, default 'regions'
          The path to the dataset inside the hdf5 file.

        dataset_opts : options to store the map
          Options for dataset creation via h5py. For example: compression='gzip'.
          See http://docs.h5py.org/en/latest/high/dataset.html
        """
        h5_path = op.expanduser(h5_path)
        with h5py.File(h5_path, 'a') as h5:
            # write map
            h5.create_dataset(dataset_path, data=self.voxel2labelid, **dataset_opts)
            # write label information
            attrs = h5[dataset_path].attrs
            for column in self.labels_df.columns:
                if self.labels_df[column].dtype == object:
                    attrs[column] = self.labels_df[column].values.astype('S')
                else:
                    attrs[column] = self.labels_df[column].values
            attrs['columns'] = self.labels_df.columns.values.astype('S')
            # write label sets
            for label_group_name, labels in self.label_groups.items():
                attrs['ls=%s' % label_group_name] = np.array(sorted(labels), dtype='S')

    @classmethod
    def from_hdf5(cls, h5_path, dataset_path='regions'):
        """
        Reads the region map and metadata from a dataset in an hdf5 file.

        Parameters
        ----------
        h5_path : string
          The path to the hdf5.

        dataset_path : string, default 'regions'
          The path to the dataset inside the hdf5 file.

        Returns
        -------
        A `Voxel2Labels` object.
        """
        h5_path = op.expanduser(h5_path)
        with h5py.File(h5_path, 'r') as h5:
            # read the map
            voxel2labelid = h5[dataset_path][()]
            # read the regions info dataframe
            attrs = h5[dataset_path].attrs
            columns = attrs['columns']
            data = {column: attrs[column] for column in columns if not column.startswith('ls=')}
            labels_df = pd.DataFrame(data=data)[columns]
            # read label sets
            label_groups = {label_group_name[3:]: list(attrs[label_group_name]) for label_group_name in attrs
                            if label_group_name.startswith('ls=')}
            return cls(labels_df, voxel2labelid, label_groups)

    @classmethod
    def from_clusters(cls, img_shape, labels, present_voxels):
        voxel2i, i2voxel = voxel2i2voxel(img_shape)
        cluster_image = np.zeros(img_shape, dtype=choose_min_int_dtype(labels.max()))
        cluster_ids = np.unique(labels)
        for cluster_id in cluster_ids:
            voxels = i2voxel(present_voxels[np.where(labels == cluster_id)[0]])
            cluster_image[voxels] = cluster_id + 1
        return VoxelLabels(labels_df=pd.DataFrame({'name': cluster_ids, 'id': cluster_ids}),
                           voxel2labelid=cluster_image)

    def to_nrrd(self, nrrd_path):
        # we should save also metadata and allow roundtrip
        nrrd.write(nrrd_path, self.voxel2labelid)

    @property
    def shape(self):
        return self.voxel2labelid.shape

    @property
    def dtype(self):
        return self.voxel2labelid.dtype


def masked_to_value(mask,
                    image=None,
                    masked_value=0,
                    dtype=None,
                    inplace=False):
    """
    Makes masked entries in image to have masked_value.

    Parameters
    ----------
    mask : boolean numpy array
      Entries with true value will mark which values to change in the image

    image : numpy array or None, default None
      The image to change the masked values in.
      Must have the same shape as `mask`.
      If None, a new zeros array with the same shape as `mask` and provided `dtype` will be created.

    masked_value : scalar, default 0
      The value that will be set to all the masked values in the image.

    dtype : numpy dtype or None
      The dtype of the new image.
      If None, no dtype is enforced.
      If None and `image` is None, dtype becomes the smallest integer type that can represent `masked_value`.

    inplace : boolean, default False
      If False, a copy of `image` is done and returned.

    Returns
    -------
    The image with the masked values changed to `masked_value`
    """

    # Create new image?
    if image is None:
        dtype = dtype if dtype is not None else choose_min_int_dtype(masked_value)
        image = np.zeros(shape=mask.shape, dtype=dtype)
        inplace = True

    # Check shapes
    if image.shape != mask.shape:
        raise ValueError('The shape of the image (%r) does not match the shape of the mask (%r)' %
                         (image.shape, mask.shape))

    # Avoid side effects?
    if not inplace:
        image = image.copy()

    # Ensure dtype?
    if dtype is not None:
        image = image.astype(dtype, copy=False)

    # Change requested values
    image[mask] = masked_value

    return image

masked2value = masked_to_value


def percentile_mask(img, q=99, invert=False, min_threshold=0):
    """Returns a mask with everything over the percentile q unmasked."""
    threshold = max(np.percentile(img, q=q), min_threshold)
    if invert:
        return img >= threshold, {'threshold': threshold}
    return img < threshold, {'threshold': threshold}


@numba.jit(nopython=True)
def _byte_freqs(data, max_val=255):
    counts = np.zeros(max_val + 1, dtype=np.uint64)
    for x in data:
        counts[x] += 1
    return counts


def percentile_from_freqs(freqs, q=99):
    assert 0 <= q <= 100
    cumsum = np.cumsum(freqs)
    where = np.where(cumsum / cumsum[-1] >= q / 100)[0]
    return where[0] if 0 < len(where) else None


def non_zero_counter(image, stat_name='pre_non_zero'):
    return {stat_name: (image > 0).sum()}


def histogrammer(image, stat_name='pre_distro', use_numba=True, as_list=False):
    max_val = 1 if image.dtype <= np.bool else np.iinfo(image.dtype).max
    dtype = choose_min_int_dtype(max_val=max_val)
    if not use_numba or isinstance(numba, NoopI):
        counts = np.zeros(max_val + 1, dtype=np.uint64)
        present, present_counts = np.unique(image.astype(dtype), return_counts=True)
        counts[present] = present_counts
    else:
        counts = _byte_freqs(image.ravel().astype(dtype), max_val=max_val)
    counts = counts.tolist() if as_list else counts
    return {stat_name: counts}


# --- Utility functions

def voxel2i2voxel(shape, i_base=0):
    """
    Returns two functions (voxel2i, i2voxel).

    - voxel2i takes voxel coordinates arrays (either a single voxel or a list of len(shape) arrays)
      and returns the corresponding index in the C-order raveled array.

    - i2voxel takes indices in the C-order raveled array (either a single integer or a list of integers)
      and returns a single tuple or len(shape) arrays with the coordinates in the image

    Parameters
    ----------
    shape : int tuple
      The shape of the image we want to index the voxels from

    i_base : int, default 0
      Indices (the "i" in i2voxel and voxel2i) will be i_base numbered.
      Usually values that make sense are 0 and 1

    Examples
    --------
    >>> voxel2i, i2voxel = voxel2i2voxel((10, 2, 5))
    >>> voxel2i([0, 1, 0])
    5
    >>> i2voxel(5)
    (0, 1, 0)
    >>> voxel2i([[0, 1, 2], [1, 1, 1], [0, 3, 4]])
    array([ 5, 18, 29])
    >>> i2voxel([5, 18, 29])
    (array([0, 1, 2]), array([1, 1, 1]), array([0, 3, 4]))
    """
    # create the maps voxel -> voxel_index and line -> line_index
    # because we have a grid, we can just use simple formulae and not heavy maps
    i2voxel = partial(np.unravel_index, dims=shape, order='C')
    voxel2i = partial(np.ravel_multi_index, dims=shape, order='C')
    if 0 != i_base:
        return lambda v: voxel2i(v) + i_base, lambda i: i2voxel(i - i_base)
    return voxel2i, i2voxel


def choose_min_int_dtype(max_val,
                         dtypes=(np.ubyte, np.uint16, np.uint32, np.uint64)):
    """Returns the smallest numpy dtype that can represent values lesser or equal than `max_val`."""
    for dtype in dtypes:
        if max_val <= np.iinfo(dtype).max:
            return dtype
    return None


# Image resizing helpers

def nn_voxel2rescaled(original_img_shape, rescaled_shape):
    """Given a downsampled shape and an upscale factor,
    Returns an array co that correspond to the index
    This assumes nearest-neighbor (up/down)sampling.

    Examples
    --------
    Image we have a clustering in the downsampled space
    >>> img_shape = 10, 10, 3
    >>> upscale = 30, 30, 9
    >>> clustering_down = np.random.RandomState(0).randint(0, high=60, size=img_shape)

    To find the corresponding clustering in an upsampled space assuming nearest neighbor interpolation
    >>> voxel_correspondence = nn_voxel2rescaled(clustering_down.shape, rescaled_shape=upscale)
    >>> clustering_up = clustering_down.ravel()[voxel_correspondence].reshape(upscale)
    """
    return (zoom(np.arange(np.prod(original_img_shape)).reshape(original_img_shape),
                 zoom=rescaled_shape, order=0, mode='constant').
            ravel())


def nn_rescale(image, new_shape):
    """Rescales an image using nearest neighbor interpolation."""
    return zoom(image, zoom=new_shape, order=0, mode='constant')


class NNResizer(object):
    def __init__(self, shape1, shape2):
        super(NNResizer, self).__init__()
        self.shape1 = shape1
        self.shape2 = shape2
        self._one2two = nn_voxel2rescaled(shape2, shape1)
        self._two2one = nn_voxel2rescaled(shape1, shape2)
        # Warning, these two can be big arrays...

    def resize(self, image):
        if image.shape == self.shape1:
            return image.ravel()[self._one2two].reshape(self.shape2)
        elif image.shape == self.shape2:
            return image.ravel()[self._two2one].reshape(self.shape1)
        else:
            raise ValueError('The image shape %r is not one of %r' % (image.shape, [self.shape1, self.shape2]))
