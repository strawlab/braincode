#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import sys
import json
import errno
import collections
from StringIO import StringIO
import datetime

import pandas as pd
import numpy as np
import scipy.stats

from braincode.util import get_filenames, get_all_datasets, get_all_neuropils, \
    get_finished_cluster_types, vt_name_to_num, vt_name_to_vdrc_url, \
    vt_name_to_bbweb_url, janelia_name_to_flylight_url, get_csv_metadata

def filter_csv(neuropil,dataset,cluster_type):
    filter_parameters = "hypergeometric_p < 1e-40 and fraction > 0.2 and fold_enrichment > 2"
    # filter_parameters = "hypergeometric_p < 1e-40"
    # filter_parameters = "cluster_id == 1"

    filenames = get_filenames( dataset, neuropil, cluster_type )
    hs_name_filtered = filenames['fragment_info_filtered_csv']

    original_metadata = get_csv_metadata(filenames['fragment_info_raw_csv'])

    if os.path.exists(hs_name_filtered):
        derived_metadata = get_csv_metadata(hs_name_filtered)
        if original_metadata['analysis_time'] == derived_metadata['analysis_time']:
            if filter_parameters == derived_metadata['filter_parameters']:
                print('output %r exists with same data and parameters. skipping.' % hs_name_filtered)
                return

    print('reading %r' % filenames['fragment_info_raw_csv'] )
    qq = pd.read_csv(filenames['fragment_info_raw_csv'],comment='#')
    print('done reading')

    # remove space from column names
    forward = {
        'hypergeometric p': 'hypergeometric_p',
        'fold enrichment': 'fold_enrichment'}
    qq.rename(columns=forward,
        inplace=True)

    # filter for high significance
    hs = qq.query(filter_parameters)
    del qq # free memory

    # restore column names
    reverse = dict([(v,k) for (k,v) in forward.iteritems()])
    hs_sorted = hs.rename(columns=reverse)
    del hs # free memory

    buf = StringIO()
    metadata = {
        'analysis_time':original_metadata['analysis_time'],
        'url':'https://strawlab.org/braincode',
        'neuropil':neuropil,
        'dataset':dataset,
        'cluster_type':cluster_type,
        'filter_parameters': filter_parameters,
    }
    comment_line = '# '+json.dumps( metadata ) + '\n'
    buf.write(comment_line)
    hs_sorted.to_csv(buf,index=False)
    with open(hs_name_filtered,mode='w') as fd:
        fd.write(buf.getvalue())
    print('saved to %r'%hs_name_filtered)

if __name__=='__main__':
    for neuropil in get_all_neuropils():
        for dataset in get_all_datasets():
            cluster_types = get_finished_cluster_types( dataset, neuropil )
            for cluster_type in cluster_types:
                try:
                    filter_csv(neuropil,dataset,cluster_type)
                except IOError as err:
                    if err.errno == errno.ENOENT: # not found
                        print('WARNING: DID NOT FINISH %r BECAUSE OF MISSING FILE.' %
                            ((neuropil,dataset,cluster_type),),file=sys.stderr)
                        print('   %s' % err, file=sys.stderr)
                        continue
