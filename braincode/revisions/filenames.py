import os.path as op

from braincode.revisions.config import BRAINCODE_SUP_DATA_DIR, BRAINCODE_DATA_CACHE_DIR

def get_hierarchy_temp_dir(dataset, neuropil, cluster_type):
    return op.join(BRAINCODE_DATA_CACHE_DIR,
        'hierarchical_clustering',
        '{dataset}__{neuropil}__{cluster_type}'.format(
            dataset=dataset,
            neuropil=neuropil,
            cluster_type=cluster_type
        ))

def get_hierarchy_dir(dataset, region, cluster_type):
    return op.join(BRAINCODE_SUP_DATA_DIR,
        dataset,
        region,
        cluster_type)

def get_hierarchy_file_prefix(dataset, region, cluster_type):
    return 'hierarchical_clustering__{dataset}__{region}__{cluster_type}'.format(
            dataset=dataset,
            region=region,
            cluster_type=cluster_type,
        )

def get_nrrd_cache_dirname(dataset, region, cluster_type):
    return op.join(BRAINCODE_DATA_CACHE_DIR,
        'image_accumulation',
        '{dataset}__{region}__{cluster_type}'.format(
            dataset=dataset,
            region=region,
            cluster_type=cluster_type
        ))

def get_nrrd_filename(dataset, region, cluster_type, cluster_id, what='average'):
    dest_dir = get_nrrd_cache_dirname(dataset,region,cluster_type)
    return op.join(dest_dir, '%s_filtered__K%03d.nrrd' % (what,cluster_id) )


def get_driver_index_filename(dataset, singletons_only=True):
    if singletons_only:
        my_type = 'singletons'
    else:
        my_type = 'all_agglomerated'

    return op.join(BRAINCODE_DATA_CACHE_DIR,
        'driver_index',
        'driver_index__{my_type}__{dataset}.csv'.format(
            my_type=my_type,
            dataset=dataset,
        ))


def get_average_image_dirname(dataset, region, cluster_type):
    return op.join(BRAINCODE_DATA_CACHE_DIR,
        'average_images',
        '{dataset}__{region}__{cluster_type}'.format(
            dataset=dataset,
            region=region,
            cluster_type=cluster_type
        ))


def get_average_image_filename(dataset, region, cluster_type, cluster_id, axis):
    dest_dir = get_average_image_dirname(dataset,region,cluster_type)
    return op.join(dest_dir, 'average_filtered__K%03d_%s.png' % (cluster_id,axis) )
