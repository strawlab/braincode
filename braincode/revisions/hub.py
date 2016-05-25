# coding=utf-8
from __future__ import print_function, division
from future.utils import string_types

import os
import os.path as op
import subprocess
import functools

import h5py
import nrrd
import numpy as np
import pandas as pd

from braincode.revisions.config import braincode_dir
from braincode.revisions.images import ParanoidImageReader, VoxelLabels
from braincode.util import mergedicts


def _my_viewer(use_fiji=False):
    """Finds imagej and its envvars for us."""
    from socket import gethostname
    if gethostname() in ('noisy', 'str22', 'strall'):
        if use_fiji:
            envvars = {
                '_JAVA_OPTIONS': '-Dawt.useSystemAAFontSettings=on '
                                 '-Dswing.aatext=true '
                                 '-Dswing.defaultlaf=com.sun.java.swing.plaf.gtk.GTKLookAndFeel'
            }
            command = ['ImageJ-linux64', '--ij2']
            return command, envvars
        # ImageJ keeps failing opening nrrds from the command line (it works if using the menus)
        # A problem with our nrrds?
        # So we switch to 3dslicer:
        #   https://www.slicer.org/ (AUR: 3dslicer)
        # The command line command is rich on options (read them).
        # It takes long to load ATM, explore how to reduce the time and maybe always use the same instance
        # Also we could use its python facilities
        #   https://www.slicer.org/slicerWiki/index.php/Documentation/Nightly/Developers/
        #   Python_scripting#Running_a_CLI_from_Python
        return ['Slicer', '--launcher-no-splash'], {}
    return ['ImageJ-linux64'], {}


def memoize(obj):
    # This implementation modified from
    # https://wiki.python.org/moin/PythonDecoratorLibrary to detect
    # hash collisions (important, given the relatively bad hashing
    # function).
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        inputs = (args, kwargs)
        key = str(args) + str(kwargs)  # could make a better hashing function
        if key not in cache:
            output = obj(*args, **kwargs)
            cache[key] = (inputs, output)
        original_inputs, original_output = cache[key]
        assert original_inputs == inputs, 'hash collision'
        return original_output
    return memoizer


class Hub(object):

    image_opener_command, image_opener_env = _my_viewer()

    def __init__(self, root):
        super(Hub, self).__init__()
        self.root = root
        self.dataset = op.basename(self.root)

    # --- Factories

    @staticmethod
    def datasets():
        return 'T1', 'CB1'

    @staticmethod
    def hubs():
        return map(Hub.hub, Hub.datasets())

    @staticmethod
    def hub(dataset_or_hub='T1'):
        """Returns a Hub for the dataset."""
        if isinstance(dataset_or_hub, Hub):
            return dataset_or_hub
        if isinstance(dataset_or_hub, string_types):
            if dataset_or_hub not in Hub.datasets():
                raise ValueError('Unknown dataset %r. Valid values: %r' %
                                 (dataset_or_hub, Hub.datasets()))
            return Hub(op.join(op.expanduser(braincode_dir()), dataset_or_hub))
        raise ValueError('Dataset_or_hub must be a dataset name or a Hub instance')

    @staticmethod
    def t1():
        """Vienna Tiles dataset hub."""
        return Hub.hub('T1')

    @staticmethod
    def cb1():
        """Complete Brain 1 (Janelia dataset) hub."""
        return Hub.hub('CB1')

    # --- Image identification and retrieval

    def image2driver_df(self):
        """Returns a dataframe with the following columns:
          - db_id: the internal brainbase database id of the image
          - dataset: the dataset ID (usually constant CB1 or T1)
          - file_name: the amira file name
          - channel: the channel name of the amira file (usually what was expressed)
          - driver: the flybase id for the driver line
        """
        image2driver_csv = op.join(self.root, 'image2driver.csv')
        return pd.read_csv(image2driver_csv, index_col=False,
                           dtype={'channel': object, 'db_id': int})

    def image2driver_df_description(self):
        """Returns a text description of the image2driver.csv file."""
        image2driver_csv_description = op.join(self.root, 'image2driver.csv' + '.description')
        with open(image2driver_csv_description, 'r') as reader:
            return reader.read()

    def image_names(self, keep_ext=False):
        """Returns a list with all the image names in the dataset.
        This is the "canonical" index for the images.
        For example, the columns in expression matrices correspond to lines in this order.
        """
        if keep_ext:
            return sorted(self.image2driver_df().file_name)
        return sorted(self.image2driver_df().file_name.apply(lambda fn: op.splitext(fn)[0]).values)

    def dbids(self):
        """Return the database ids of the images with the canonical order imposed by `image_names`."""
        return (self.image2driver_df().
                set_index('file_name').
                loc[self.image_names(keep_ext=True)].
                db_id.
                values)

    @memoize
    def image_path(self, name=None, index=None, dbindex=None, pipeline=None):
        """Returns the path to an original image."""
        if sum([x is not None for x in (name, index, dbindex)]) != 1:
            raise ValueError('name=%r, index=%r, dbindex=%r, but one and only one must be None' %
                             (name, index, dbindex))
        if index is not None:
            name = self.image_names()[index]
        if dbindex is not None:
            names = self.image2driver_df().query('db_id == %r' % dbindex)
            if 0 == len(names):
                raise Exception('No image can be found for dbindex=%r' % dbindex)
            if 1 < len(names):
                raise Exception('Multiple images can be found for dbindex=%r' % dbindex)
            name = names.iloc[0].file_name
        if pipeline is None:
            if not name.endswith('.am'):
                name += '.am'
            return op.join(self.root, 'originals', 'images', name)
        else:
            if not name.endswith('.nrrd'):
                name += '.nrrd'
            return op.join(self.root, 'images', 'pipeline=%s' % pipeline, name)

    def image(self, name=None, index=None, dbindex=None, pipeline=None):
        """
        Returns a 3D numpy array containing an image in this dataset.

        If pipeline is None, data for an original image is given.
        If pipeline is not None, an image after passing by a pipeline is given.

        Images can be addressed by:
           - name (like "AR10062801L4Sum01" or "AR10062801L4Sum01.am"),
           - canonical index as defined by `image_names` (e.g. 0)
           - database id (e.g. 32100)
        """
        path = self.image_path(name=name, index=index, dbindex=dbindex, pipeline=pipeline)
        # An original image
        if path.endswith('.am'):
            return ParanoidImageReader(path).data
        # A downsampled image
        return nrrd.read(path)[0]

    def view(self, name=None, index=None, dbindex=None, pipeline=None, block=False):
        """Opens an image in a viewer, probably fiji.
        The viewer can be changed by manipulating `Hub.image_opener_command` and `Hub.image_opener_env`.
        """
        path = self.image_path(name=name, index=index, dbindex=dbindex, pipeline=pipeline)
        env = mergedicts(os.environ.copy(), self.image_opener_env)
        print('Opening %s' % path)
        if block:
            subprocess.call(self.image_opener_command + [path], env=env)
        else:
            subprocess.Popen(self.image_opener_command + [path], env=env)

    def num_lines(self):
        """Returns the number of lines in the dataset (alias for `num_images`)."""
        return len(self.image_names())

    def num_images(self):
        """Returns the number of images in the dataset."""
        return self.num_lines()

    # --- Template, regions and image stats

    def shape(self, pipeline=None):
        """Returns the shape (x_size, y_size, z_size) of the images after passing by a pipeline."""
        if pipeline is None:
            with h5py.File(op.join(self.root, 'template_regions.h5'), 'r') as h5:
                return h5['template'].shape
        return self.image(index=0, pipeline=pipeline).shape

    def num_voxels(self, pipeline=None):
        """Returns the number of voxels in the registered images."""
        return np.prod(self.shape(pipeline=pipeline))

    def regions(self, pipeline=None):
        """
        Returns a `VoxelLabels` instance with the original template neuropil segmentation.
        :rtype: VoxelLabels
        """
        vl = VoxelLabels.from_hdf5(op.join(self.root, 'template_regions.h5'))
        if pipeline is not None:
            vl = vl.copy(downsampler=self.downsampler(pipeline))
        return vl

    def template(self, pipeline=None):
        """Returns a three tuple (template (3D array), bounding box, bounding box description."""
        with h5py.File(op.join(self.root, 'template_regions.h5'), 'r') as h5:
            template = h5['template'][()]
            bounding_box = h5['template'].attrs['bounding_box']
            bounding_box_description = h5['template'].attrs['bounding_box_description']
            template = self.downsampler(pipeline=pipeline)(template)
            return template, bounding_box, bounding_box_description

    def bounding_box(self):
        """
        Returns the bounding box of the original template.
        This is [xmin, xmax, ymin, ymax, zmin, zmax], units are micrometers.
        """
        with h5py.File(op.join(self.root, 'template_regions.h5'), 'r') as h5:
            return h5['template'].attrs['bounding_box']

    def voxel_size(self, pipeline=None):
        """Returns (a close approximation) of the size of a voxel, in micrometers.
        Note that if pipeline is not None, not all the voxels might represent the same volume size.
        """
        with h5py.File(op.join(self.root, 'template_regions.h5'), 'r') as h5:
            xmin, xmax, ymin, ymax, zmin, zmax = h5['template'].attrs['bounding_box']
            xsize, ysize, zsize = self.shape(pipeline=pipeline)
            # The images are a uniform 3D grid of voxels
            return ((xmax - xmin) / xsize,
                    (ymax - ymin) / ysize,
                    (zmax - zmin) / zsize)

    # --- Image processing pipelines

    def original_pipeline(self):
        return {'CB1': 'cb1_orig_wannabe', 'T1': 't1_orig_wannabe'}.get(self.dataset, None)

    def pipelines(self):
        """Returns a list of available image pipelines (that is, image preprocessing workflows)."""
        return sorted(pipeline[len('pipeline='):]
                      for pipeline in os.listdir(op.join(self.root, 'images')))

    def pipeline_stats(self, *pipelines):
        """Returns a pandas dataframe with the available stats for the pipelines."""
        from braincode.revisions.pipelineit import consolidate_stats
        return pd.concat(consolidate_stats(pipeline=pipeline, dest_dir=op.dirname(self.root))
                         for pipeline in pipelines)

    @staticmethod
    def all_pipelines_stats(hubs=None,
                            id_columns=('dataset', 'image', 'pipeline', 'what'),
                            ignore_columns=('image_id',),
                            drop_na=True,
                            categoricals=True):
        """
        Munges all known pipelines stats for the provided hubs into a tidy pandas dataframe.

        If `hubs` is None, uses all known hubs.

        Returns a dataframe with a stat per row, `ignore_columns` removed,
        `id_columns` kept and 4 extra columns:
         - step_num: the order of the step in the pipeline, -1 for overall stats
         - step_name: the name of the step, 'overall' for full pipeline stats
         - stat: the name of the stat
         - value: the value of the stat

        If `categoricals` is True, id_columns and possibly step_name and stat are casted
        as categorical columns in the dataframe.

        If `drop_na` is True, it is assumed that missings only happen because of concat
        (that is, there is no possible missing value as stat value) and drops rows
        accordingly.
        """

        hubs = Hub.hubs() if hubs is None else hubs

        # Read the pipeline stats from the logs, concatenate
        stats_df = (pd.concat(map(lambda hub: hub.pipeline_stats(*hub.pipelines()), hubs)).
                    rename(columns=lambda col: col[7:] if col.startswith('stats__') else col))

        # Remove ignored columns
        ignore_columns = [] if ignore_columns is None else list(ignore_columns)
        stats_df.drop(ignore_columns, inplace=True, axis=1)

        # Melt
        stats_df = pd.melt(stats_df,
                           id_vars=id_columns,
                           var_name='step_stat',
                           value_name='value')

        # Make 3 columns out of step info
        def break_step_stat_name(step_stat_name):
            if step_stat_name.startswith('step__'):
                _, step_num, step_name, step_stat = step_stat_name.split('__')
                return int(step_num), step_name, step_stat
            return -1, 'overall', step_stat_name
        broken_df = pd.DataFrame(stats_df.step_stat.apply(break_step_stat_name).values.tolist(),
                                 columns=['step_num', 'step_name', 'stat'])
        del stats_df['step_stat']
        stats_df = pd.concat((stats_df, broken_df), axis=1)

        # Remove missings (assume that missing only happens due to concat)
        if drop_na:
            stats_df = stats_df.dropna(subset=['value'], how='any', axis=0)

        # Relevant column order
        column_order = list(id_columns) + ['step_num', 'step_name', 'stat', 'value']
        stats_df = stats_df[column_order]

        # Categoricals
        if categoricals:
            categorical_columns = list(id_columns) + ['step_name', 'stat']
            for column in categorical_columns:
                stats_df[column] = stats_df[column].astype('category')

        return stats_df

    @staticmethod
    def downsampler(pipeline=None):
        if pipeline is None:
            return lambda x: x
        from braincode.revisions.pipelineit import find_downsampler
        return find_downsampler(pipeline)

    # --- Data matrices: sparse
    #  Maybe coming: dense (dataframe), off-core with quick random access to voxels

    def Xcsr(self, pipeline=None):
        """Returns a pair (Xcsr, image_shape).

        - Xcsr is a sparse matrix with all the images and voxels after the dataset have been processed by the pipeline.
        Voxels are per rows, voxel actual coordinates are like in `revisions.images.voxel2i2voxel`.
        Images are per columns, sorted as in `image_names()`.
        The dtype can be anything from bool to float.

        - Image shape is the shape of each original image in a column.
        To recreate an original matrix array: X[:, 3].reshape(image_shape).astype(np.uint8).
        """
        from braincode.revisions.pipelineit import Xcsr
        if pipeline is None:
            pipeline = self.original_pipeline()
        return Xcsr(pipeline=pipeline, dest_dir=op.dirname(self.root))

    # --- Clustering results

    def _clusterings_path(self):
        # TODO: this should be relative to the dataset dir
        return op.join(op.dirname(self.root), 'clusterings')

    def _clusterings_group_path(self, clusterings_group):
        path = op.join(self._clusterings_path(), clusterings_group)
        if not op.isdir(path):
            raise Exception('Cannot find clusterings group %s' % clusterings_group)
        return path

    def clusterings_groups(self):
        return sorted(path for path in os.listdir(self._clusterings_path())
                      if op.isdir(op.join(self._clusterings_path(), path)))

    def clusterings_df(self,
                       clusterings_group='clusterings-florian',
                       reconsolidate=False,
                       verbose=False,
                       regions_to_string=True,
                       only_this_dataset=True,
                       n_jobs=1):
        from clustering import MBKClusteringResult
        path = self._clusterings_group_path(clusterings_group)
        cached_pickle = op.join(path, 'clusterings_df.pickle')
        if reconsolidate or not op.isfile(cached_pickle):
            df = MBKClusteringResult.df_from_path(path, verbose=verbose, n_jobs=n_jobs)
            df.to_pickle(cached_pickle)
        df = pd.read_pickle(cached_pickle)
        if regions_to_string:
            df['regions'] = ['__'.join(regions) for regions in df['regions']]
        if only_this_dataset:
            df = df.query('dataset == "%s"' % self.dataset)
        return df

    def add_clustering_column(self,
                              clusterings_df,
                              clusterings_group='clusterings-florian',
                              inplace=False,
                              column='clustering'):
        # Silly API, get smaller and sweeter...
        from clustering import MBKClusteringResult
        path = self._clusterings_group_path(clusterings_group)
        if not inplace:
            clusterings_df = clusterings_df.copy()
        clusterings_df[column] = MBKClusteringResult.from_df(clusterings_df, path)
        return clusterings_df

    def clusterings(self, query, clusterings_group='clusterings-florian'):
        if isinstance(query, string_types):
            cdf = self.clusterings_df(clusterings_group=clusterings_group).query(query)
        else:
            cdf = query  # Assume a clusterings pandas dataframe
        return self.add_clustering_column(cdf, clusterings_group=clusterings_group)


if __name__ == '__main__':
    for dataset in Hub.datasets():
        hub = Hub.hub(dataset_or_hub=dataset)
        # Get the stats for all pipelines
        print(hub.pipelines())
        print(hub.pipeline_stats(*hub.pipelines()))
        # Use data from a pipeline
        pipeline = hub.original_pipeline()
        image = hub.image(index=3, pipeline=pipeline)
        # Roundtrip an image from the expression matrix
        Xcsr, img_shape = hub.Xcsr(pipeline=pipeline)
        image_roundtrip = Xcsr[:, 3].toarray().reshape(img_shape).astype(np.uint8)
        assert (image == image_roundtrip).all()
        # View the image in 3DSlicer
        hub.view(index=3, pipeline=hub.original_pipeline())
