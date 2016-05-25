# coding=utf-8
from __future__ import division, print_function

import glob
import hashlib
import os.path as op
import time
from functools import partial
from itertools import product, chain
from string import Template

import argh
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from future.utils import string_types
from sklearn.cluster import MiniBatchKMeans

from braincode.revisions.hub import Hub
from braincode.revisions.images import VoxelLabels
from braincode.revisions.pipelineit import Xcsr, DOWNSAMPLER_CB1, DOWNSAMPLER_T1
from braincode.util import ensure_dir
from whatami import What, id2what
from whatami.misc import MAX_EXT4_FN_LENGTH
from whatami.wrappers.what_sklearn import whatamise_sklearn

# Make sklearn whatami aware
whatamise_sklearn()


# Where clusterings will live
# ATM a directory both Florian and the cluster know about
CLUSTERINGS_ROOT = '~/strawscience/santi/andrew/clustering-paper/forFLORIAN/clusterings'


# --- Clustering proof of concept

@argh.arg('-k', '--ks', nargs='+', type=int)
@argh.arg('-s', '--seeds', nargs='+', type=int)
@argh.arg('-r', '--regions', nargs='+', type=str)
def ducktape_kmeans(dataset='CB1',
                    pipeline=None,
                    expression_threshold=5,
                    template_threshold=1,
                    regions='Optic_Glomeruli',
                    seeds=(0, 1, 2, 3, 5),
                    ks=tuple(range(2, 200, 2)),
                    minibatch_ratio=5,
                    downsampler=None,
                    dest_h5='~/test-ducktape.h5',
                    force=False):
    # TODO: accept a factory (data, context) -> clusterer

    dest_h5 = op.abspath(op.expanduser(dest_h5))
    ensure_dir(op.dirname(dest_h5))

    if pipeline is None:
        pipeline = 'cb1_orig_wannabe' if dataset == 'CB1' else 't1_orig_wannabe'

    coords = dict(
        dataset=dataset,
        pipeline=pipeline,
        expression_threshold=expression_threshold,
        template_threshold=template_threshold,
        regions=regions,
        minibatch_ratio=minibatch_ratio,
        downsampler=downsampler,
    )

    print('Loading expression matrix')
    X, img_shape = Xcsr(pipeline=pipeline)

    # --- Voxel selection

    hub = Hub.hub(dataset)

    # To downsample masks
    # We really need to define an API for UpDownSamplers
    if downsampler is None:
        downsampler = DOWNSAMPLER_CB1 if dataset == 'CB1' else DOWNSAMPLER_T1

    # Template mask
    template, _, _ = hub.template()
    template_mask = downsampler(template) < template_threshold
    # Regions mask
    if tuple(regions) == ('full',):
        regions = None
    if isinstance(regions, string_types):
        regions = [regions]
    if regions is None:
        regions = ()
    regions_mask = downsampler(hub.regions().mask(*regions))
    # Only keep what is kept in both
    kept_voxels = ~(template_mask | regions_mask).ravel()
    # Report...
    num_voxels_in_template = (~template_mask).sum()
    print('Voxels in template: %d' % num_voxels_in_template)
    num_voxels_in_regions = (~regions_mask).sum()
    print('Voxels in regions: %d' % num_voxels_in_regions)
    num_voxels_in_regions_not_in_template = (template_mask & ~regions_mask).sum()
    print('Voxels in regions, not in template: %d' % num_voxels_in_regions_not_in_template)
    # Expression mask
    ever_expressed = np.array(X.sum(axis=1)).ravel() >= expression_threshold
    kept_voxels &= ever_expressed
    kept_voxels = np.where(kept_voxels)[0]
    # Select kept voxels
    X = X[kept_voxels, :]
    # Final report...
    num_voxels_expressed = ever_expressed.sum()
    print('Voxels ever expressed: %d' % num_voxels_expressed)
    print('Kept voxels: %d' % len(kept_voxels))

    # -- Perform the clusterings
    for k, seed in product(ks, seeds):

        # This implementation of MBK does not include center sparsification
        # (ala L1 min in the original paper).
        mbk = MiniBatchKMeans(random_state=seed,
                              n_clusters=k,
                              batch_size=int(X.shape[0] / minibatch_ratio),
                              verbose=False,
                              # init
                              init='k-means++',  # Here we could be clever and pass neuropil centers...
                              n_init=3,
                              init_size=None,
                              max_iter=500,
                              # stopping criteria
                              max_no_improvement=50,
                              tol=0.0,
                              # center reassignment policy
                              reassignment_ratio=0.01)

        this_coords = coords.copy()
        this_coords['clusterer'] = mbk
        # this_coords['clusterer'] = mbk.get_params()
        # these could be extracted from clusterer, for convenience
        this_coords['k'] = k
        this_coords['seed'] = seed

        what = What('ducktape_kmeans', conf=this_coords)
        whatid = what.id()
        print(whatid)

        ensure_dir(op.dirname(dest_h5))
        with h5py.File(dest_h5, 'a') as h5:
            if force or whatid not in h5:
                start = time.time()
                mbk.fit(X)
                g = h5.create_group(whatid)  # impressed that h5 can eat these, otherwise just trim + hash
                g.create_dataset('centers', data=mbk.cluster_centers_, compression='gzip', compression_opts=5)
                g.create_dataset('labels', data=mbk.labels_, compression='gzip', compression_opts=5)
                g.create_dataset('voxels', data=kept_voxels, compression='gzip', compression_opts=5)
                g.attrs['num_voxels_in_template'] = num_voxels_in_template
                g.attrs['num_voxels_in_regions'] = num_voxels_in_regions
                g.attrs['num_voxels_in_regions_not_in_template'] = num_voxels_in_regions_not_in_template
                g.attrs['num_voxels_expressed'] = num_voxels_expressed
                g.attrs['img_shape'] = img_shape
                g.attrs['whatid'] = whatid
                g.attrs['inertia'] = mbk.inertia_
                g.attrs['last_iteration'] = mbk.n_iter_
                g.attrs['taken_s'] = time.time() - start
            else:
                print('Already done, skipping...')


_CLUSTER_CLUSTER_SCRIPT_TEMPLATE = Template("""#!/usr/bin/env bash
#
# Note that we also do not specify resource (memory, time) limits.
# At the moment the cluster is quite idle, so no big competition in the queues.
#
# SGE submission options
#$$ -N ${jobid}
#$$ -j y
#$$ -cwd


START=`date +%Y/%m/%d\ %H:%M:%S`
echo "Started at $$START"

time python -u ~/braincode/braincode/braincode/revisions/clustering.py ducktape-kmeans \\
--dataset ${dataset} \\
--regions ${regions} \\
--seeds ${seeds} \\
--ks ${ks} \\
--expression-threshold ${expression_threshold} \\
--template-threshold ${template_threshold} \\
--dest-h5 "${dest_h5}"

END=`date +%Y/%m/%d\ %H:%M:%S`
echo "Started at $$START"
echo "Ended at $$END"
echo "DONE"

""")

REGIONS = (
    'full',
    'Antennal_lobe_L',
    'Antennal_lobe_R',
    'Antennal_lobe',
    'Optic_Glomeruli_L',
    'Optic_Glomeruli_R',
    'Optic_Glomeruli',
    'Mushroom_Body_L',
    'Mushroom_Body_R',
    'Mushroom_Body',
    'SEZ',
    'Central_Complex',
)

DATASETS_PIPELINES = (
    ('CB1', 'cb1_orig_wannabe'),
    ('T1', 't1_orig_wannabe'),
)

MIN_K = 2
MAX_K = 250
STEP_K = 3

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

THRESHOLDS = (
    # expression (how many lines is a voxel expressed), template
    (5, 1),  # Minimum expressed in 5 lines, expressed with at leas minimum intensity in the template
    (0, 0),  # No voxel filtered out
)


def generate_cluster_scripts():
    for region, (dataset, pipeline), (expression_t, template_t) in product(REGIONS, DATASETS_PIPELINES, THRESHOLDS):
        for start, seed in product(range(STEP_K), SEEDS):
            jobid = ('ducktape_kmeans('
                     'dataset={dataset},'
                     'pipeline={pipeline},'
                     'regions={region},'
                     'seed={seed},'
                     'expression_threshold={expression_threshold},'
                     'template_threshold={template_threshold},'
                     'start={start},end={end},step={step}'
                     ')'.format(dataset=dataset,
                                pipeline=pipeline,
                                region=region,
                                seed=seed,
                                expression_threshold=expression_t,
                                template_threshold=template_t,
                                start=start, step=STEP_K, end=MAX_K))

            dest_hdf5 = '~/braincode/clusterings/' + jobid + '.h5'

            command = _CLUSTER_CLUSTER_SCRIPT_TEMPLATE.substitute(
                jobid=jobid,
                dataset=dataset,
                regions=region,
                seeds=seed,
                ks=' '.join(map(str, range(MIN_K + start, MAX_K + 1, STEP_K))),
                dest_h5=dest_hdf5,
                pipeline=pipeline,
                expression_threshold=expression_t,
                template_threshold=template_t,
                start=start,
                step=STEP_K,
            )
            with open(op.expanduser('~/%s.sh' % jobid), 'w') as writer:
                writer.write(command)


# --- Generation of clusterings for Florian - might just be the clusterings we end up using for k-bounding in the paper


@argh.arg('-k', '--ks', nargs='+', type=int)
@argh.arg('-r', '--regions', nargs='+', type=str)
def multithreaded_ducktape_kmeans(dataset='CB1',
                                  regions='Antennal_lobe_L',
                                  ks=tuple(range(2, 120)),
                                  seed=0,
                                  n_jobs=-2,
                                  dest_dir=CLUSTERINGS_ROOT):
    # TODO: bring back image pipeline configurability into the mix

    ensure_dir(op.expanduser(dest_dir))

    flo_duct_tape = partial(ducktape_kmeans,
                            expression_threshold=0,
                            template_threshold=0)

    # Mimic parallel n_jobs semantics
    if 0 == n_jobs:
        n_jobs = cpu_count()
    if 0 > n_jobs:
        n_jobs += cpu_count() + 1

    # Balance k accross jobs (larger ks are much more expensive to compute)
    k_sets = [tuple(ks[start::n_jobs]) for start in range(n_jobs)]

    # Takes care of too long hdf5 file names
    def not_too_long_fn(k_set):
        fn = 'clusterer=mbk1__dataset=%s__regions=%s__ks=%r__seed=%r' % (dataset, regions, k_set, seed)
        if len(fn) > MAX_EXT4_FN_LENGTH:
            k_set = hashlib.sha224(repr(k_set)).hexdigest()
            fn = 'clusterer=mbk1__dataset=%s__regions=%s__ks=%r__seed=%r' % (dataset, regions, k_set, seed)
        return fn + '.h5'

    Parallel(n_jobs=n_jobs)(delayed(flo_duct_tape)
                            (dataset=dataset,
                             regions=regions,
                             seeds=(seed,),
                             ks=k_set,
                             dest_h5=op.join(dest_dir, not_too_long_fn(k_set)))
                            for k_set in k_sets)
    print('Done')


_CLUSTER_MULTITHREADED_SCRIPT_TEMPLATE = Template("""#!/usr/bin/env bash
#
# Note that we also do not specify resource (memory, time) limits.
# At the moment the cluster is quite idle, so no big competition in the queues.
#
# SGE submission options
#$$ -N ${jobid}
#$$ -pe smp ${n_jobs}
#$$ -j y
#$$ -cwd


START=`date +%Y/%m/%d\ %H:%M:%S`
echo "Started at $$START"

time python -u ~/braincode/braincode/braincode/revisions/clustering.py multithreaded-ducktape-kmeans \\
--dataset ${dataset} \\
--regions ${regions} \\
--seed ${seed} \\
--ks ${ks} \\
--n-jobs ${n_jobs}

END=`date +%Y/%m/%d\ %H:%M:%S`
echo "Started at $$START"
echo "Ended at $$END"
echo "DONE"

""")


def generate_multithreaded_cluster_scripts(script_template=_CLUSTER_MULTITHREADED_SCRIPT_TEMPLATE,
                                           regions=REGIONS,
                                           seeds=SEEDS,
                                           ks=tuple(range(2, 121)),
                                           k_set_step=5,
                                           n_jobs=6):
    k_sets = [ks[start::k_set_step] for start in range(k_set_step)]
    for region, dataset, k_set, seed in product(regions, ('CB1', 'T1'), k_sets, seeds):
        jobid = ('multithreaded_ducktape_kmeans('
                 'dataset={dataset},'
                 'regions={region},'
                 'seed={seed},'
                 'ks={ks}'
                 ')'.format(dataset=dataset,
                            region=region,
                            seed=seed,
                            ks=hashlib.md5(' '.join(map(str, k_set))).hexdigest()))

        command = script_template.substitute(
            jobid=jobid,
            dataset=dataset,
            regions=region,
            seed=seed,
            ks=' '.join(map(str, k_set)),
            n_jobs=n_jobs
        )
        with open(op.expanduser('~/%s.sh' % jobid), 'w') as writer:
            writer.write(command)


# --- Generate one nrrd per clustering (warning, file count explosion)

def nrrds_for_florian(clusterer='mbk1', path=CLUSTERINGS_ROOT, force=False):
    def process_result(h5, result_id):
        what = id2what(result_id)
        dataset = what['dataset']
        neuropil = what['regions'][0]
        seed = what['seed']
        k = what['k']
        fn = 'clusterer={clusterer}__dataset={dataset}__region={neuropil}__k={k}__seed={seed}'.format(
            clusterer=clusterer,
            dataset=dataset,
            neuropil=neuropil,
            k=k,
            seed=seed)
        dest_nrrd = op.join(ensure_dir(op.join(path, 'nrrds', dataset, neuropil)), '%s.nrrd' % fn)
        if force or not op.isfile(dest_nrrd):
            g = h5[result_id]
            labels = g['labels'][()]
            voxels = g['voxels'][()]
            img_shape = g.attrs['img_shape'][()]
            vl = VoxelLabels.from_clusters(img_shape, labels, voxels)
            vl.to_nrrd(dest_nrrd)

    path = op.expanduser(path)
    h5_paths = sorted(glob.glob(op.join(path, 'clusterer=%s*.h5' % clusterer)))
    for h5_path in h5_paths:
        with h5py.File(h5_path) as h5:
            result_ids = sorted(h5.keys())
            print('Saving %d clustering nrrds from %s' % (len(result_ids), op.basename(h5_path)))
            for result_id in result_ids:
                process_result(h5, result_id)


# --- Results management


def _index_one_clusterings_hdf5(cls, h5_path, verbose=False):
    if verbose:
        print(h5_path)
    with h5py.File(h5_path, 'r') as h5:
        return [cls(h5_path, result_id).to_dict() for result_id in h5.keys()]


class MBKClusteringResult(object):
    """Result reader ad-hoc for minibatch k-means clustering results.
    Lazy on hdf5 / result_id coordinates.
    """

    _INTERESTING_KEYS = (
        'img_shape',
        'num_voxels_in_template',
        'num_voxels_expressed',
        'num_voxels_in_regions',
        'num_voxels_in_regions_not_in_template',
        'taken_s',
        'last_iteration',
        'inertia',
        # 'whatid',
    )

    _INTERESTING_WHATID_KEYS = {
        'dataset': 'dataset',
        'regions': 'regions',
        'pipeline': 'pipeline',
        'minibatch_ratio': 'minibatch_ratio',
        'template_threshold': 'template_threshold',
        'expression_threshold': 'expression_threshold',
        'k': ('clusterer', 'n_clusters'),
        'seed': ('clusterer', 'random_state'),
    }

    def __init__(self, h5_path, result_id):
        super(MBKClusteringResult, self).__init__()
        self.h5_path = h5_path
        self.result_id = result_id

    def _read_dataset(self, dataset):
        with h5py.File(self.h5_path, 'r') as h5:
            return h5[self.result_id][dataset][()]

    def present_labels(self):
        return np.unique(self.labels())

    def labels(self):
        return self._read_dataset('labels')

    def centers(self):
        return self._read_dataset('centers')

    def voxels(self):
        return self._read_dataset('voxels')

    def image_shape(self):
        return self.attrs()['img_shape']

    def image(self):
        return VoxelLabels.from_clusters(self.image_shape(), self.labels(), self.voxels())

    def attrs(self):
        with h5py.File(self.h5_path, 'r') as h5:
            return dict(h5[self.result_id].attrs)

    def whatid(self):
        return self.attrs()['whatid']

    def to_dict(self, keys=_INTERESTING_KEYS, whatid_keys=_INTERESTING_WHATID_KEYS):
        attrs = self.attrs()
        result = {key: attrs[key] for key in keys}
        what = id2what(attrs['whatid'])
        result.update({keyname: what[key] for keyname, key in whatid_keys.items()})
        result['clusterer'] = 'mbk1'
        result['h5_file'] = op.basename(self.h5_path)
        result['result_id'] = self.result_id
        return result

    @classmethod
    def from_df(cls, df, root_dir, h5_path_column='h5_file', result_id_column='result_id'):
        return [cls(h5_path=op.join(root_dir, h5_path), result_id=result_id)
                for h5_path, result_id in zip(df[h5_path_column], df[result_id_column])]

    @classmethod
    def df_from_path(cls, root_dir, globpattern='*.h5', verbose=False, n_jobs=1):
        hdf5s = sorted(glob.glob(op.join(root_dir, globpattern)))
        results = (Parallel(n_jobs=n_jobs, pre_dispatch='5*n_jobs')
                   (delayed(_index_one_clusterings_hdf5)(cls, h5_path, verbose=verbose)
                    for h5_path in hdf5s))
        return pd.DataFrame(list(chain.from_iterable(results)))


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([ducktape_kmeans, generate_cluster_scripts,
                         multithreaded_ducktape_kmeans, generate_multithreaded_cluster_scripts,
                         nrrds_for_florian])
    parser.dispatch()

# For relatively small datasets, do:
#   AffinityPropagation in smaller datasets
#   Try density based clusterers too (particularly slow-like dbscan)
#   Spectral, agglomerative and what not
