#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import os.path as op
import json
import collections
from StringIO import StringIO
import datetime

import pandas as pd
import numpy as np
import scipy.stats

from braincode.util import get_filenames, get_all_datasets, get_all_neuropils, \
    get_finished_cluster_types, vt_name_to_num, vt_name_to_vdrc_url, \
    vt_name_to_bbweb_url, janelia_name_to_flylight_url, ensure_dir
from braincode.fragments_per_cluster_step1_compute import get_fragment_cache_fname

from braincode.revisions.hierarchical import AgglomeratedClusteringResult
from braincode.revisions.filenames import (get_hierarchy_dir,
    get_hierarchy_file_prefix)
from braincode.revisions.config import BRAINCODE_DATA_CACHE_DIR
from braincode.revisions.hub import Hub
from braincode.revisions.pipelineit import Xcsr

FORCE_REWRITE = bool(int(os.environ.get('FORCE_REWRITE','0')))

def get_fragment_pre_csv_fname(dataset,region,cluster_type):
    fragment_cache_dir = ensure_dir(BRAINCODE_DATA_CACHE_DIR)
    fn = 'fragment_statistics__{dataset}__{region}__{cluster_type}.csv'.format(
            dataset=dataset,
            region=region,
            cluster_type=cluster_type,
        )
    return op.join(fragment_cache_dir,fn)

def run_analysis(neuropil,dataset,cluster_type):
    hub = Hub.hub(dataset)
    if dataset == 'CB1':
        pipeline = 'cb1_orig_wannabe'
    else:
        assert dataset == 'T1'
        pipeline = 't1_orig_wannabe'
    full_csr_matrix, img_shape = Xcsr(pipeline=pipeline)
    dbids = hub.dbids()
    full_brain_full_num_voxels, full_brain_num_drivers = full_csr_matrix.shape
    assert len(dbids) == full_brain_num_drivers
    assert full_brain_full_num_voxels == np.prod(img_shape)

    dbid_to_count = pd.Series( np.array(full_csr_matrix.sum(axis=0)).ravel(), index=dbids)
    full_brain_num_voxels = len(np.nonzero(full_csr_matrix.sum(axis=1))[0])

    analysis_time = datetime.datetime.now().isoformat()

    filenames = get_filenames( dataset, neuropil, cluster_type )
    hs_name_raw_pre = get_fragment_pre_csv_fname(dataset=dataset,
        region=neuropil, cluster_type=cluster_type)

    if os.path.exists(hs_name_raw_pre):
        if not FORCE_REWRITE:
            return
    fragment_cache_fname = get_fragment_cache_fname(dataset=dataset,
        region=neuropil, cluster_type=cluster_type)

    ids_fname = filenames['id_driver_image_csv']
    print('reading %r'%ids_fname)
    id_driver_image_df = pd.read_csv(ids_fname, sep=';')
    driver_id_to_driver_name = {}
    for i,driver_image_row in id_driver_image_df.iterrows():
        driver_id_to_driver_name[ driver_image_row['id'] ] = driver_image_row['driver']

    print('reading %r'%fragment_cache_fname)
    store = pd.HDFStore( fragment_cache_fname, mode='r' )
    df = store['fragment_stats']
    store.close()
    print('done reading')

    print('reorganizing data structures')
    num_expressing_voxels_in_region = {}
    num_voxels_in_region = {}
    everywhere_df = df[ df['cluster_id']==-1 ]
    for i,everywhere_row in everywhere_df.iterrows():
        num_expressing_voxels_in_region[ everywhere_row['driver_id'] ] = everywhere_row['num_expressing_voxels_this_cluster']
        num_voxels_in_region[ everywhere_row['driver_id'] ] = everywhere_row['num_voxels_this_cluster']

    q = collections.defaultdict(list)

    package_data_dir = get_hierarchy_dir(dataset=dataset,
        region=neuropil, cluster_type=cluster_type)
    fn = get_hierarchy_file_prefix(dataset=dataset,
        region=neuropil, cluster_type=cluster_type)
    agglomeration_json_path = op.join(package_data_dir, fn + '.json')
    agglom = AgglomeratedClusteringResult(df, agglomeration_json_path)
    for cluster_id, cluster_df in agglom.iter_clusters():
        if cluster_id==-1:
            continue
        for i,cluster_row in cluster_df.iterrows():
            driver_id = cluster_row['driver_id']

            q['cluster_id'].append( cluster_id )
            q['driver_id'].append( driver_id )
            q['expressing cluster'].append( cluster_row['num_expressing_voxels_this_cluster'] )
            q['expressing region'].append( num_expressing_voxels_in_region[ driver_id ] )
            q['expressing entire'].append( dbid_to_count.loc[ driver_id ] )
            q['total cluster'].append( cluster_row['num_voxels_this_cluster'] )
            q['total region'].append( num_voxels_in_region[ driver_id ] )
            q['total entire'].append( full_brain_num_voxels )

    qq = pd.DataFrame(data=q)
    print('done reorganizing data structures')

    buf = StringIO()
    metadata = {
        'analysis_time':analysis_time,
        'url':'https://strawlab.org/braincode',
        'neuropil':neuropil,
        'dataset':dataset,
        'cluster_type':cluster_type,
    }
    comment_line = '# '+json.dumps( metadata ) + '\n'
    buf.write(comment_line)
    qq.to_csv(buf,index=False)
    with open(hs_name_raw_pre,mode='w') as fd:
        fd.write(buf.getvalue())
    print('saved to %r'%hs_name_raw_pre)

if __name__=='__main__':
    for neuropil in get_all_neuropils():
        for dataset in get_all_datasets():
            cluster_types = get_finished_cluster_types( dataset, neuropil )
            for cluster_type in cluster_types:
                run_analysis(neuropil,dataset,cluster_type)
