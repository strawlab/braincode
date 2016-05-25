# coding=utf-8
"""Reproduce original image processing pipelines and create our own."""
from __future__ import division, print_function
from future.utils import string_types

import gzip
import json
import os.path as op
import time
from array import array
from functools import partial
from string import Template

import h5py
import nrrd
import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening, zoom, grey_opening
from scipy.sparse import csr_matrix

from braincode.revisions.config import braincode_dir
from braincode.revisions.hub import Hub
from braincode.revisions.images import percentile_mask, histogrammer
from braincode.util import ensure_dir, mergedisjointdicts, flattendict
from whatami import whatable, what2id


# These will collect image histograms before and after each step
pre_freqs = partial(histogrammer, stat_name='pre_histogram', as_list=True)
post_freqs = partial(histogrammer, stat_name='post_histogram', as_list=True)


@whatable
class Pipeline(object):
    # Probably we could just use/bridge sklearn APIs and their Pipeline class...

    def __init__(self,
                 steps,
                 save_stats=False,
                 pre_stats=pre_freqs,
                 post_stats=post_freqs):
        super(Pipeline, self).__init__()
        self.steps = steps
        self.save_stats = save_stats
        self.pre_stats = pre_stats
        self.post_stats = post_stats
        self.num_processed_ = 0
        self.stats_ = {}

    def apply(self, image, image_id=None, verbose=False):
        # Image id
        image_id = ('%d#%s' % (self.num_processed_, image_id)) if image_id is not None else str(self.num_processed_)
        # Log start
        if verbose:
            print('\tStart all', image_id)
        # Initialise stats
        stats = {'image_id': image_id, 'step_stats': []}
        full_start = time.time()
        for step_name, step in self.steps:
            # Initialise step stats
            step_start = time.time()
            step_stats = {'name': step_name}
            # Log start
            if verbose:
                print('\tStart step', image_id, step_name)
            # Compute pre-step stats on the image
            if self.pre_stats is not None:
                step_stats = mergedisjointdicts(step_stats, self.pre_stats(image))
            # Run the step, save its stats
            image = step(image)
            if isinstance(image, tuple):
                image, processing_stats = image
                step_stats = mergedisjointdicts(step_stats, processing_stats)
            # Compute post-step stats on the image
            if self.post_stats is not None:
                step_stats = mergedisjointdicts(step_stats, self.post_stats(image))
            # Time taken
            step_stats['taken_s'] = time.time() - step_start
            # Save step stats
            stats['step_stats'].append(step_stats)
            # Log done
            if verbose:
                print('\tDone step', image_id, step_name, '%.2fs' % step_stats['taken_s'])
        # Total time taken to process the image
        stats['taken_s'] = time.time() - full_start
        # Save the stats to the pipeline diccionary
        if self.save_stats:
            self.stats_ = mergedisjointdicts(self.stats_, stats)
        # Update processed count
        self.num_processed_ += 1
        # Log end
        if verbose:
            print('\tDone all')
        # Bye
        return image, stats

    def copy(self):
        return self.__class__(steps=self.steps)

    def time(self, step_name):
        return self._times[step_name]

#
# --- Original image processing pipeline reproduction wannabe
#  From the paper:
#   1- Thresholding to get a 1% voxels per image
#   2- Morphological opening with a 3x3x3 kernel to reduce clutter
#   3- To decrease the effects of registration error and image acquisition noise
#      and to speedup computations, images were binned into larger voxels,
#      typically a 3x3x3 downsampling.
#

DOWNSAMPLER_CB1 = partial(zoom, zoom=1 / 3, order=0, mode='constant')
DOWNSAMPLER_CB1_ALT = partial(zoom, zoom=1 / 2, order=0, mode='constant')
DOWNSAMPLER_T1 = partial(zoom, zoom=(1 / 5, 1 / 5, 1 / 4), order=0, mode='constant')
DOWNSAMPLER_T1_ALT = partial(zoom, zoom=1 / 3, order=0, mode='constant')

PIPELINE_CB1 = Pipeline((
    # polemic thresholding
    ('binarise1', partial(percentile_mask, q=99, invert=True, min_threshold=1)),
    # salt and pepper removal - hopefully I get it right with the structure
    ('open333', partial(binary_opening, structure=np.ones((3, 3, 3)))),
    # downsampling by nns interpolation, hopefully exactly what was done as originally
    ('downsample333', DOWNSAMPLER_CB1),
    # mask out of regions; let's assume this was done after the rest with a downsampled mask...
))

PIPELINE_CB1_ALT1 = Pipeline((
    # no thresholding
    # some noise reduction
    ('gopen333', partial(grey_opening, size=(3, 3, 3))),
    # downsampling by nns interpolation, a bit less aggressive than originally
    ('downsample222', DOWNSAMPLER_CB1_ALT),
))

PIPELINE_CB1_ALT2 = Pipeline((
    # no thresholding
    # no noise reduction
    # downsampling by nns interpolation, a bit less aggressive than originally
    ('downsample222', DOWNSAMPLER_CB1_ALT),
))

PIPELINE_T1 = Pipeline((
    # polemic thresholding
    ('binarise1', partial(percentile_mask, q=99, invert=True, min_threshold=1)),
    # salt and pepper removal - hopefully I get it right with the structure
    ('open333', partial(binary_opening, structure=np.ones((3, 3, 3)))),
    # downsampling by nns interpolation, hopefully exactly what was done as originally
    ('downsample554', DOWNSAMPLER_T1),
    # mask out of regions; let's assume this was done after the rest with a downsampled mask...
))

PIPELINE_T1_ALT1 = Pipeline((
    # no thresholding
    # some noise reduction
    ('gopen333', partial(grey_opening, size=(3, 3, 3))),
    # downsampling by nns interpolation, a bit less aggressive than originally
    ('downsample333', DOWNSAMPLER_T1_ALT),
))

PIPELINE_T1_ALT2 = Pipeline((
    # no thresholding
    # no noise reduction
    # downsampling by nns interpolation, a bit less aggressive than originally
    ('downsample222', DOWNSAMPLER_T1_ALT),
))

PIPELINES = {
    'cb1_orig_wannabe': ('CB1', PIPELINE_CB1),
    'cb1_alt1': ('CB1', PIPELINE_CB1_ALT1),
    'cb1_alt2': ('CB1', PIPELINE_CB1_ALT2),
    't1_orig_wannabe': ('T1', PIPELINE_T1),
    't1_alt1': ('T1', PIPELINE_T1_ALT1),
    't1_alt2': ('T1', PIPELINE_T1_ALT2),
}


def find_pipeline(pipeline_name):
    return PIPELINES[pipeline_name] + (find_downsampler(pipeline_name),)


def find_downsampler(pipeline_name):
    return PIPELINES[pipeline_name][1].steps[-1][1]


def pipelineit(hub='CB1',
               image_name_or_index='CB1_GMR_9G08_AE_01_31-fA01b_C100418_20100419213620609_02_warp_m0g80c8e1e-1x26r301',
               pipeline='cb1_orig_wannabe',
               verbose=False):
    """Passes an image on a dataset through a pipeline."""

    if isinstance(hub, string_types):
        hub = Hub.hub(hub)

    img = hub.image(image_name_or_index)

    if isinstance(pipeline, string_types):
        dataset, pipeline = PIPELINES[pipeline]
        assert dataset == hub.dataset

    img, stats = pipeline.apply(img, image_id=image_name_or_index, verbose=verbose)

    return img, pipeline, stats


def _dest_dir(pipeline='cb1_orig_wannabe', dest_dir=None):
    dataset, _ = PIPELINES[pipeline]
    if dest_dir is None:
        dest_dir = braincode_dir()
    return ensure_dir(op.join(op.expanduser(dest_dir), dataset, 'images', 'pipeline=%s' % pipeline))


def process_and_save(pipeline='cb1_orig_wannabe',
                     start=0, step=300, force=False, verbose=True,
                     dest_dir=None):

    dataset, _ = PIPELINES[pipeline]
    dest_dir = _dest_dir(pipeline=pipeline, dest_dir=dest_dir)

    hub = Hub.hub(dataset)
    for imagenum in range(start, hub.num_lines(), step):
        image_name = hub.image_names()[imagenum]
        dest_image = op.join(dest_dir, image_name + '.nrrd')
        dest_json = op.join(dest_dir, image_name + '.json.gz')
        if force or not op.isfile(dest_json):
            if verbose:
                print('Dataset: {dataset}; image: {image}'.format(dataset=dataset, image=image_name))
            image, pipeline, stats = pipelineit(hub, image_name_or_index=image_name, pipeline=pipeline, verbose=verbose)
            nrrd.write(dest_image, image.astype(np.uint8))
            # should add options spacing / nms / voxel size...
            # should add np.ubyte to nrrd datatypes
            with gzip.open(dest_json, 'w') as writer:
                info = {'what': what2id(pipeline), 'stats': stats}
                json.dump(info, writer, indent=2)
            if verbose:
                print('DONE Dataset: {dataset}; image: {image}'.format(dataset=dataset, image=image_name))


# --- Generate job scripts for the cluster

IMAGE_PIPELINE_CLUSTER_SCRIPT_TEMPLATE = Template("""#!/usr/bin/env bash
#
# Note that we also do not specify resource (memory, time) limits.
# At the moment the cluster is quite idle, so no big competition in the queues.
#
# SGE submission options
#$$ -N ${pipeline}.${start}.${step}   # Set the job name
#$$ -j y                              # Join stderr and stdout
#$$ -cwd                              # Go to current working directory

START=`date +%Y/%m/%d\ %H:%M:%S`
echo "Started at $$START"

cd ~/braincode/braincode
time python -u braincode/revisions/pipelineit.py process-and-save \\
--pipeline ${pipeline} \\
--start ${start} --step ${step}

END=`date +%Y/%m/%d\ %H:%M:%S`
echo "Started at $$START"
echo "Ended at $$END"
echo "DONE"

""")


def generate_cluster_jobs(dest_dir=op.expanduser('~'),
                          step_size=150,
                          template=IMAGE_PIPELINE_CLUSTER_SCRIPT_TEMPLATE):
    for pipeline in PIPELINES:
        for start in range(step_size):
            fn = '{pipeline}_{start}_{step_size}.sh'.format(pipeline=pipeline,
                                                            start=start,
                                                            step_size=step_size)
            with open(op.join(ensure_dir(dest_dir), fn), 'w') as writer:
                writer.write(template.substitute(pipeline=pipeline,
                                                 start=start,
                                                 step=step_size))


# --- Consolidate datasets


def consolidate_stats(pipeline='cb1_orig_wannabe', force=False, dest_dir=None):
    """Returns a pandas dataframe with all the stats collected during processing."""
    # Where to read data from
    dest_dir = _dest_dir(pipeline=pipeline, dest_dir=dest_dir)
    # Hub, to make sure we have everything
    dataset, _ = PIPELINES[pipeline]
    hub = Hub.hub(dataset)
    # Destination file
    stats_pickle = op.join(dest_dir, 'consolidated_stats.pickle')

    # We want numpy arrays instead of python lists
    def list_to_array(v):
        return np.array(v) if isinstance(v, tuple) else v

    def step_stats_to_dict(step_stats):
        result = {}
        for i, stats in enumerate(step_stats):
            name = '%d__%s' % (i, stats['name'])
            result[name] = {k: v for k, v in stats.items() if k != 'name'}
        return result

    if force or not op.isfile(stats_pickle):
        rows = []
        for i, image_name in enumerate(hub.image_names()):
            print(i, image_name)
            json_path = op.join(dest_dir, image_name + '.json.gz')
            with gzip.open(json_path, 'r') as reader:
                row = json.load(reader)
                row['stats']['step'] = step_stats_to_dict(row['stats']['step_stats'])
                del row['stats']['step_stats']
                row = {k: list_to_array(v) for k, v in flattendict(row, sep='__').items()}
                row.update(dataset=dataset, image=image_name, pipeline=pipeline)
                rows.append(row)
        # Maybe add categoricals here, but they are small dataframes
        pd.DataFrame(rows).to_pickle(stats_pickle)

    return pd.read_pickle(stats_pickle)


def Xcsr(pipeline='cb1_orig_wannabe', force=False, dest_dir=None):
    # Consolidates a dataset of images into a CSR sparse matrix
    # with fast random access to voxel data
    # (as opposed to fast random access to image data).
    #
    # This should just take an stream of images
    #
    # Eventually this should be hidden in a "voxel provider" object,
    # as in-memory sparse won't cut it in the general case.
    # And of course, be part of the hub
    #

    dataset, _ = PIPELINES[pipeline]
    dest_dir = _dest_dir(pipeline=pipeline, dest_dir=dest_dir)
    hdf5_file = op.join(dest_dir, 'consolidated.h5')
    group_name = '{dataset}_{pipeline}'.format(dataset=dataset, pipeline=pipeline)
    hub = Hub.hub(dataset)

    if force or not op.isfile(hdf5_file):

        # --- Read original data into a CSR sparse matrix
        #  Eventually we should support something else than binary data
        indptr = array('i', [0])
        indices = array('i')
        img_shape = None
        for i, image_name in enumerate(hub.image_names()):
            print(i, image_name)
            img, _ = nrrd.read(op.join(dest_dir, image_name + '.nrrd'))
            on_voxels = np.nonzero(img.ravel())[0]
            if len(on_voxels) > img.size * 0.5:
                print('WARNING: %s is quite dense (%d%% non-zero)...' %
                      (image_name, 100 * len(on_voxels) / img.size))
            if len(on_voxels.shape) > 0:
                indices.extend(on_voxels)
                indptr.append(indptr[-1] + len(on_voxels))
            else:  # never expressed line...
                indptr.append(indptr[-1])
            if img_shape is None:
                img_shape = img.shape
            else:
                if img_shape != img.shape:
                    raise Exception('Image %s has a different shape than the rest' % image_name)

        # --- Make voxels to be the fastest accessed object, matrix be num_voxels x num_lines
        #   N.B. frombuffer avoids costly data copying
        #   For example, h5py uses "array-has-buffer-interface unaware" asarray.
        #   Obviously we could just write incrementally to the hdf5 (or use jagged...)
        #   However, this in-memory transposition is very fine for aggresive
        #   data reduction pipelines like the one originally used in the paper.
        indices = np.frombuffer(indices, dtype=np.int32)
        indptr = np.frombuffer(indptr, dtype=np.int32)
        data = np.ones(len(indices), dtype=np.bool)
        X = csr_matrix((data, indices, indptr),
                       shape=(len(indptr) - 1, np.prod(img_shape))).T.tocsr()

        # --- Store
        with h5py.File(hdf5_file, 'w') as h5:  # Maybe we want open mode 'a', e.g. if we allow to save X.T too
            g = h5.create_group(group_name)
            g.attrs['shape'] = X.shape
            g.attrs['img_shape'] = img_shape
            g.attrs['dtype'] = 'bool'
            g.create_dataset('indices', dtype=np.int32, data=X.indices)
            g.create_dataset('indptr', dtype=np.int32, data=X.indptr)
            # Maybe compress? Of course, it would make random read from disk costly.
            # Note that here we abuse that data is binary... should be parameterised or inferred

        return X, img_shape

    with h5py.File(hdf5_file, 'r') as h5:
        # Here it is very easy to have a random-voxel-access interface
        # It is also very easy to check for non-set / 0-length objects using indptr
        # Implement
        g = h5[group_name]
        shape = g.attrs['shape']
        img_shape = g.attrs['img_shape']
        indptr = g['indptr'][()]
        indices = g['indices'][()]
        data = np.ones(len(indices), dtype=np.bool)
        return csr_matrix((data, indices, indptr), shape=shape), img_shape


if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([process_and_save, generate_cluster_jobs, Xcsr, consolidate_stats])
    parser.dispatch()
