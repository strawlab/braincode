#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import json
import collections
from StringIO import StringIO
import datetime

import pandas as pd
import numpy as np
import scipy.stats

from braincode.util import get_filenames, get_all_datasets, get_all_neuropils, \
    get_finished_cluster_types, vt_name_to_num, vt_name_to_vdrc_url, \
    vt_name_to_bbweb_url, janelia_name_to_flylight_url, get_csv_metadata
from braincode.fragments_per_cluster_step2_save_csv import get_fragment_pre_csv_fname

from braincode.revisions.config import BRAINCODE_PACKAGE_DIR

def process_csv(neuropil,dataset,cluster_type):
    analysis_time = datetime.datetime.now().isoformat()
    WITH_CHROMOSOME_LOCATION=True
    if WITH_CHROMOSOME_LOCATION:
        import track # install with "pip install track" (see http://bbcf.epfl.ch/bbcflib/tutorial_track.html)
        if dataset=='T1':
            dataset_track = track.load(os.path.join(BRAINCODE_PACKAGE_DIR,'VTs.bed'))
        elif dataset=='CB1':
            dataset_track = track.load(os.path.join(BRAINCODE_PACKAGE_DIR,'janeliaTiles.bed'))
        else:
            raise ValueError('uknown dataset %r' % dataset)

    filenames = get_filenames( dataset, neuropil, cluster_type )
    hs_name_raw = filenames['fragment_info_raw_csv']

    pre_csv_fname = get_fragment_pre_csv_fname(dataset=dataset,
        region=neuropil, cluster_type=cluster_type)
    original_metadata = get_csv_metadata(pre_csv_fname)

    if os.path.exists(hs_name_raw):
        derived_metadata = get_csv_metadata(hs_name_raw)
        if original_metadata['analysis_time_parsed'] < derived_metadata['analysis_time_parsed']:
            print('output %r exists but is newer, so not rewriting. skipping.' % hs_name_raw)
            return
        else:
            print('output %r exists but is older than input, so rewriting.' % hs_name_raw)

    ids_fname = filenames['id_driver_image_csv']
    print('reading %r'%ids_fname)
    id_driver_image_df = pd.read_csv(ids_fname, sep=';')
    driver_id_to_driver_name = {}
    for i,driver_image_row in id_driver_image_df.iterrows():
        driver_id_to_driver_name[ driver_image_row['id'] ] = driver_image_row['driver']

    print('reading %r' % pre_csv_fname )
    qq = pd.read_csv(pre_csv_fname, low_memory=False, comment='#')

    print('computing statistics')
    qq['expressing elsewhere_in_region'] = qq['expressing region'] - qq['expressing cluster']
    qq['total elsewhere_in_region'] = qq['total region'] - qq['total cluster']

    qq['fraction cluster'] = qq['expressing cluster']/qq['total cluster']
    qq['observed'] = qq['expressing cluster']
    if 0:
        # expected value is expression everywhere
        qq['expected freq'] = qq['expressing region']/qq['total region']
    else:
        # expected value is expression elsewhere
        qq['expected freq'] = qq['expressing elsewhere_in_region']/qq['total elsewhere_in_region']
    qq['expected'] = qq['expected freq'] * qq['total cluster']
    qq['fold enrichment'] = qq['observed']/qq['expected']
    qq['chi sq'] = (qq['observed'] - qq['expected'])**2 / qq['expected']
    qq['chi sq p'] = scipy.stats.chisqprob(qq['chi sq'].values, df=1 )
    if 1:
        hypergeometric_p = []
        for i,qq_row in qq.iterrows():
            if i%10000==0:
                print('%d of %d'%(i,len(qq)))
            # Variable names correspond to the scipy docs at
            # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
            # Comment names correspond to the wikipedia entry at
            # https://en.wikipedia.org/wiki/Hypergeometric_distribution
            M = qq_row['total region']      # wikipedia's N
            n = qq_row['expressing region'] # wikipedia's K
            N = qq_row['total cluster']      # wikipedia's n
            x = qq_row['expressing cluster'] # wikipedia's k
            rv = scipy.stats.hypergeom( M, n, N )
            hypergeometric_p.append( rv.pmf(x) )
        qq['hypergeometric p'] = hypergeometric_p
    qq['driver name'] = qq.apply( lambda row:
                                  driver_id_to_driver_name[ row['driver_id'] ],
                                  axis = 1 )
    print('done computing statistics')
    print('saving all %d rows'%(len(qq),))

    column_renames = collections.OrderedDict([
        # (old_name, new_name),
        ('cluster_id','cluster_id'),
        ('fraction cluster','fraction'),
        ('fold enrichment','fold enrichment'),
        ('driver name','driver name'),
        ('driver_id','driver_id'),
        ('hypergeometric p','hypergeometric p'),
        ('chi sq p','chi sq p'),
        ('expressing cluster','num positive voxels in cluster'),
        ('total cluster','num voxels in cluster'),
        ('expressing region','num positive voxels in region'),
        ('total region','num voxels in region'),
        ]
    )

    qq = qq[[old_name for old_name in column_renames]] # drop unused columns
    qq = qq.rename(columns=column_renames) # rename columns

    hs_sorted = qq.sort( ['cluster_id','fraction'], ascending=[True,False] )

    if WITH_CHROMOSOME_LOCATION:
        chrom = []
        chromStart = []
        chromEnd = []
        chromStrand = []
        ucsc_urls = []
        cached_sdata = {}

        for i,hs_row in hs_sorted.iterrows():
            if dataset=='T1':
                num = vt_name_to_num(hs_row['driver name'])
                name = 'VT%04d'%num
            else:
                assert dataset=='CB1'
                name = hs_row['driver name']
            sdata = cached_sdata.get(name,None)
            if sdata is None:
                # cache miss
                sdata = [r for r in dataset_track.search({'name':name},exact_match=True)]
                cached_sdata[name] = sdata
                if len(sdata)==0:
                    print("no entry for %r"%hs_row['driver name'])

            if len(sdata)==0:
                chrom.append(None)
                chromStart.append(None)
                chromEnd.append(None)
                chromStrand.append(None)
                ucsc_urls.append(None)
            else:
                if len(sdata)>1:
                    print('sdata')
                    print(sdata)
                    raise RuntimeError("more than one entry for %r"%hs_row['driver name'])
                trackrow = sdata[0]
                chrom.append(trackrow[0])
                chromStart.append(trackrow[1])
                chromEnd.append(trackrow[2])
                chromStrand.append(trackrow[5])
                ucsc_urls.append( 'http://genome.ucsc.edu/cgi-bin/hgTracks?db=dm3&position=' + chrom[-1] +'%3A' + str(chromStart[-1]) +'-' +str(chromEnd[-1]) )
        hs_sorted['chrom'] = chrom
        hs_sorted['chromStart'] = np.array(chromStart,dtype=object)  # don't let Pandas convert to float
        hs_sorted['chromEnd'] = np.array(chromEnd,dtype=object) # don't let Pandas convert to float
        hs_sorted['chromStrand'] = chromStrand
        hs_sorted['UCSC Genome Browser URL'] = ucsc_urls

    if dataset=='T1':
        hs_sorted['bbweb URL'] = hs_sorted.apply(lambda row: vt_name_to_bbweb_url(row['driver name']), axis=1)
        hs_sorted['VDRC URL'] = hs_sorted.apply(lambda row: vt_name_to_vdrc_url(row['driver name']), axis=1)
    else:
        assert dataset=='CB1'
        hs_sorted['FlyLight URL'] = hs_sorted.apply(lambda row: janelia_name_to_flylight_url(row['driver name']), axis=1)

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
    hs_sorted.to_csv(buf,index=False)
    with open(hs_name_raw,mode='w') as fd:
        fd.write(buf.getvalue())
    print('saved to %r'%hs_name_raw)

if __name__=='__main__':
    for neuropil in get_all_neuropils():
        for dataset in get_all_datasets():
            cluster_types = get_finished_cluster_types( dataset, neuropil )
            for cluster_type in cluster_types:
                process_csv(neuropil,dataset,cluster_type)
