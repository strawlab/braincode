# coding=utf-8
"""
Munge original data together to a more cohesive and uniform shape.

You probably want to run bootstrap_braincode_dir (at the end of this file).
This filters and puts together data for our two datasets (CB1 and T1):
  - in the strawlab central repository (images, templates and segmentations),
  - in Florian's HDF5 files (to find which images go into which dataset, CB1 or T1),
  - in Florian's CSV database dumps (linking images to driver lines).
All these also keep the original brainbase database ids for provenance.
"""
from __future__ import print_function, division

import glob
import sys
import os
import os.path as op
import shutil
from collections import namedtuple
from functools import partial
from itertools import product
from textwrap import dedent

import h5py
import nrrd
import numpy as np
import pandas as pd
from py_amira_file_reader.read_amira import read_amira

from braincode.revisions.config import braincode_dir, BRAINCODE_PACKAGE_DIR, STRAWLAB_ROOT
from braincode.revisions.images import VoxelLabels
# We probably should manage to remove these two imports.
# So ultimately we do not depend on Florian's HDF5s and CSVs.
from braincode.CAUTION.copy_florians_data import my_read_csv
from braincode.util import ExpressionDataset, get_all_datasets, get_all_neuropils, ensure_dir


# --- Original files and configurations
# Where original data and brainbase database query dumps are
# Downsample sizes From [T1|CB1]/.../Antennal_lobe_clusterimage.am
# At the moment the HDF5 files are read using braincode.util

_Originals = namedtuple('Originals', ['template',
                                      'amiras',
                                      'test_amira',
                                      'regions',
                                      'downsampled_size'])

_T1_ORIGINALS = _Originals(
    template=op.join(STRAWLAB_ROOT, 'brainwarp', 'T1V2template', 'T1.am'),
    amiras=op.join(STRAWLAB_ROOT, 'bbweb', 'internal', 'Data', 'channelimages'),
    test_amira='TP10100501L33Sum01.am',
    regions=op.join(STRAWLAB_ROOT, 'Laszlo', 'T1_Ito_new'),
    downsampled_size=(154, 154, 41))


_CB1_ORIGINALS = _Originals(
    template=op.join(STRAWLAB_ROOT, 'brainwarp', 'CB1_template', 'latest',
                     'Merged_highcontrast_BCimpr_final_notcompressed.am'),
    amiras=op.join(STRAWLAB_ROOT, 'bbweb', 'internal', 'Data', 'channelimages'),
    test_amira='CB1_GMR_9C12_AE_01_29-fA01b_C100120_20100120125741015_02_warp_m0g80c8e1e-1x26r301.am',
    regions=op.join(STRAWLAB_ROOT, 'Laszlo', 'CB1_Ito', 'ItoCb1Amira'),
    downsampled_size=(341, 171, 73))

_ID2NAME_CSV = op.join(op.dirname(BRAINCODE_PACKAGE_DIR), 'images_id_to_file.csv')
_NAME2DRIVER_CSV = op.join(op.dirname(BRAINCODE_PACKAGE_DIR), 'CAUTION', 'names_drivers.csv')


def dataset2human(dataset):
    return {'CB1': 'Janelia FlyLight', 'T1': 'Vienna Tiles'}[dataset]


# --- Original segmentations of the brains into "neuropils"
#  Format is not uniform between datasets, these functions take care of it.
#  The annotations on the CB1 dataset were inferred by "warping" these in the T1 dataset.
#  Florian knows the details.

_NEUROPILS = {
    # N.B. capitals to match supplemental dir names
    # When relevant, we did the original clusterings in the right hemisphere
    'SEZ': ['GNG', 'PRW', 'SAD', 'AMMC_L', 'AMMC_R'],
    'Mushroom_Body_L': ['MB_CA_L', 'MB_ML_L', 'MB_PED_L', 'MB_VL_L'],
    'Mushroom_Body_R': ['MB_CA_R', 'MB_ML_R', 'MB_PED_R', 'MB_VL_R'],
    'Mushroom_Body': ['MB_CA_L', 'MB_ML_L', 'MB_PED_L', 'MB_VL_L',
                      'MB_CA_R', 'MB_ML_R', 'MB_PED_R', 'MB_VL_R'],
    'Antennal_lobe_L': ['AL_L'],
    'Antennal_lobe_R': ['AL_R'],
    'Antennal_lobe': ['AL_L', 'AL_R'],
    'Optic_Glomeruli_L': ['AOTU_L', 'PVLP_L', 'PLP_L'],
    'Optic_Glomeruli_R': ['AOTU_R', 'PVLP_R', 'PLP_R'],
    'Optic_Glomeruli': ['AOTU_L', 'PVLP_L', 'PLP_L',
                        'AOTU_R', 'PVLP_R', 'PLP_R'],
    'Central_Complex': ['PB', 'EB', 'FB', 'NO']
}


def neuropil2human(neuropil):

    # The original in braincode-web/util.js

    prefix = 'Left ' if neuropil.endswith('L') else 'Right ' if neuropil.endswith('R') else ''

    n2h = {
        'Optic_Glomeruli': 'Optic Glomeruli (oVLNP)',
        'Mushroom_Body': 'Mushroom Body (MB)',
        'Central_Complex': 'Central Complex (CX)',
        'Antennal_lobe': 'Antennal Lobe (AL)',
        'SEZ': 'Subesophageal Zone (SEZ)',
    }

    for np_name, human_name in n2h.items():
        if neuropil.startswith(np_name):
            return prefix + human_name

    raise ValueError('Unknown region "%s"' % neuropil)


_REGIONNAME2ID_EXPECTATIONS = (
    ('GNG', 39),
    ('PRW', 40),
    ('SAD', 8),
    ('AMMC_L', 48),
    ('AMMC_R', 10),
    ('LO_L', 43),
    ('LO_R', 2),
    ('LOP_L', 58),
    ('LOP_R', 20),
    ('AOTU_L', 68),
    ('AOTU_R', 32),
    ('ME_L', 60),
    ('EB', 21),
)


def _parse_brain_region_descriptions(regions_path):
    # Read the region description
    # This has short descriptions for some acronyms
    regionid2description = {}
    with open(regions_path) as reader:
        text = reader.read().strip()
        for line in text.splitlines():
            region_id, _, region = line.partition(' ')
            region = region.strip()
            regionid2description[region_id] = region
    # Extend to Mushroom Body subregions
    regionid2description['MB_CA'] = \
        regionid2description['MB_ML'] = \
        regionid2description['MB_PED'] = \
        regionid2description['MB_VL'] = regionid2description['MB']
    return regionid2description


def check_region2id(name, expectation, region2id):
    if region2id[name] != expectation:
        raise Exception('Expected ID %d for regions %s, got %d' %
                        (expectation, name, region2id[name]))


def _generate_regions_nrrds(dataset, regions, sanity_checks_dir=None):
    """
    This function generates nrrd files to check by experts:
      - One with with all the region assignments.
      - One per neuropil
    These images go out of the release directory.
    """
    print('\tWriting regions nrrds for dataset %s' % dataset)
    if sanity_checks_dir is None:
        sanity_checks_dir = ensure_dir(op.abspath(op.join(braincode_dir(),
                                                          '..',
                                                          'regions-sanity')))
    # Write nrrd with all the assignments
    print('\t\tAll regions...')
    nrrd.write(op.join(sanity_checks_dir, '%s-regions.nrrd' % dataset),
               regions.voxel2labelid)
    # Write a nrrd per neuropil
    for neuropil in regions.labels() + regions.label_group_names():
        print('\t\t%s...' % neuropil)
        mask = regions.mask(neuropil)
        image = np.zeros_like(mask, dtype=np.uint8)
        image[~mask] = 255
        nrrd.write(op.join(sanity_checks_dir, '%s-%s.nrrd' % (dataset, neuropil)), image)


def _t1_regions_munger():

    amira_path = op.expanduser(op.join(_T1_ORIGINALS.regions,
                                       'composition_s2_t128_laszlo_AL_R_andL.am'))

    # Read the region description
    regions_path = op.join(op.dirname(amira_path),
                           'DrosophilaBrainRegions.terms.txt')
    region2description = _parse_brain_region_descriptions(regions_path)

    # Read the amira data file for voxel assignments and regions mappings
    amira_data = read_amira(amira_path)
    assert amira_data['info'] == {'version': '2.0', 'type': 'AmiraMesh', 'is_binary': True}
    data = amira_data['data']

    # Declared shape
    shape = tuple(data[0]['define']['Lattice'])
    assert shape == (768, 768, 165)

    parameters = data[1]['Parameters']

    # Region2id. Here 1 = exterior, 0 = unselected, anything higher means a named selected region (neuropil)
    region2id = {material: mid['Id'] - 1 for material, mid in parameters['Materials'].items()}

    # Sanity checks - Andrew provided these expectations after looking at the actual image
    for name, expected_id in _REGIONNAME2ID_EXPECTATIONS:
        check_region2id(name, expected_id, region2id)

    # We also have BoundingBox, CoordType, Content and Seeds->Slices
    # Plus data types declarations and data for two tensors:
    #   - data: the labels
    #   - probabilities: byte type, probably we do not need this

    # voxel2regionid
    voxel2regionid = data[4]['data']  # Like 768x768x165 array of codes (aka "labels")
    assert voxel2regionid.shape == shape

    # generate a table label -> id
    def LR_region2description(region):
        prefix = 'left ' if region.endswith('_L') else 'right ' if region.endswith('_R') else ''
        key = region if prefix == '' else region[:-2]
        return prefix + region2description.get(key, key)
    regions = [(region, region_id, LR_region2description(region))
               for region, region_id in sorted(region2id.items())]
    regions_df = pd.DataFrame(regions, columns=['name', 'id', 'description'])

    return VoxelLabels(regions_df, voxel2regionid, label_groups=_NEUROPILS)


def _cb1_regions_munger(clean_spurious_voxels=True):

    path = op.expanduser(_CB1_ORIGINALS.regions)

    # Read the region description
    region2description = _parse_brain_region_descriptions(op.join(path, 'DrosophilaBrainRegions.terms.txt'))

    # Read the map region_name -> label
    # Note, this can be found also in the .surf file, which seems the original
    # So maybe we should use that file instead
    params = op.join(path, 'DrosophilaBrainRegions.params.txt')

    def parse_line(line):
        region_name, _, region_id, _, r, g, b = line.strip().split()
        region_id = int(region_id)
        region_color = (float(r), float(g), float(b))
        return region_name, region_id, region_color

    with open(params, 'r') as reader:
        parsed_params = [parse_line(line) for line in reader if 0 < len(line.strip())]
    region2id = {region_name: region_id - 1 for region_name, region_id, _ in parsed_params}
    # N.B. we probably want to keep colors so we are consistent if we ever generate images

    # Sanity checks - Andrew provided these expectations after looking at the actual image
    for name, expected_id in _REGIONNAME2ID_EXPECTATIONS:
        check_region2id(name, expected_id, region2id)

    # Read the masks
    def fn2label(amira):
        return int(op.basename(amira)[len('NeuropilMask'):].split('.')[0])
    amiras = glob.glob(op.join(path, 'NeuropilMask*.am'))
    label2amira = {fn2label(amira) + 1: amira for amira in amiras}
    # N.B. +1 for making it 1-based, as in the parameters file; it seems that the files go that name
    # All this is too speculative, talk to K/L/F

    # Combine all these into one "labels" image
    voxel2region = None
    for region_id, amira in sorted(label2amira.items()):
        if voxel2region is None:
            voxel2region = read_amira(amira)['data'][3]['data']
            voxel2region[voxel2region == 1] = region_id
        else:
            data = read_amira(amira)['data'][3]['data']
            region_voxels = data == 1
            if ((voxel2region != 0) & region_voxels).any():
                raise Exception('Overlapping regions are not supported at the moment')
            voxel2region[region_voxels] = region_id
    voxel2regionid = voxel2region

    # generate a table label -> id
    def LR_region2description(region):
        prefix = 'left ' if region.endswith('_L') else 'right ' if region.endswith('_R') else ''
        key = region if prefix == '' else region[:-2]
        return prefix + region2description.get(key, key)
    regions = [(region, region_id, LR_region2description(region))
               for region, region_id in sorted(region2id.items())]
    regions_df = pd.DataFrame(regions, columns=['name', 'id', 'description'])

    # clean these pesky spurious voxels
    def remove_spurious_voxels_on_the_other_side(voxel2regionid, regions_df):
        """
        Clean spurious voxels on the other side in L/R segmentations.
        This happens apparently only in the right regions of CB1;
        I have just checked it in "AL_R", "Mushroom_Body_R" and "Optic_Glomeruli_R".
        I will assume it might happens in all the left/right regions.
        Note: this can only work in the CB1 dataset (which is centered).
        """
        # midx = int(np.round(voxel2regionid.shape[0] / 2))
        midx = 510  # Inferred as the mean x in segments joining the mean coordinates of L,R pairs in the dataset
        in_right = np.zeros_like(voxel2regionid, dtype=np.bool)
        in_right[midx:] = True  # Remember that left/right is from "fly perspective"
        # Probably we should just compute neuropil center, a histogram of distances and remove outliers
        # Or use properly fitted reflection planes
        for name, region_id in zip(regions_df.name, regions_df.id):
            if name.endswith('_L'):
                in_region = voxel2regionid == region_id
                spurious = in_region & ~in_right
                print('\t\t\t%s removed %d of %d (%.4f%%) voxels in the other side' %
                      (name, spurious.sum(), in_region.sum(), 100 * spurious.sum() / in_region.sum()))
                voxel2regionid[spurious] = 0
            if name.endswith('_R'):
                in_region = voxel2regionid == region_id
                spurious = in_region & in_right
                print('\t\t\t%s removed %d of %d (%.4f%%) voxels in the other side' %
                      (name, spurious.sum(), in_region.sum(), 100 * spurious.sum() / in_region.sum()))
                voxel2regionid[spurious] = 0
        return voxel2regionid

    if clean_spurious_voxels:
        print('\t\tWARNING: removing spurious voxels in the other side...')
        print('\t\tTHIS IS UNTESTED AND PROBABLY BROKEN AT THE MOMENT')
        voxel2regionid = remove_spurious_voxels_on_the_other_side(voxel2regionid, regions_df)

    return VoxelLabels(regions_df, voxel2regionid, label_groups=_NEUROPILS)

# dest_dir = ensure_dir(op.expanduser('~/clean-vs-not'))
# vl = _cb1_regions_munger(clean_spurious_voxels=False)
# vl.to_hdf5(op.join(dest_dir, 'original.h5'))
# _generate_regions_nrrds('CB1', vl, sanity_checks_dir=ensure_dir(op.join(dest_dir, 'original')))
# vl = _cb1_regions_munger(clean_spurious_voxels=True)
# vl.to_hdf5(op.join(dest_dir, 'clean.h5'))
# _generate_regions_nrrds('CB1', vl, sanity_checks_dir=ensure_dir(op.join(dest_dir, 'clean')))
# exit(22)

# --- Templates

def _read_template(amira_path):
    # Read
    amira_data = read_amira(amira_path)
    data = amira_data['data']
    shape = tuple(data[0]['define']['Lattice'])
    coord_type = data[1]['Parameters']['CoordType'][1:-1]
    bounding_box = np.array(data[1]['Parameters']['BoundingBox'])
    template = data[3]['data']

    # Dumbchecking stuff
    assert amira_data['info'] == {'version': '2.0', 'type': 'AmiraMesh', 'is_binary': True}
    assert coord_type == 'uniform'
    assert data[2]['Lattice']['byte'] in {'ScalarField', 'Data'}
    assert template.shape == shape

    return template, bounding_box


# --- Data release logic

def bootstrap_braincode_dir(id2name_csv=_ID2NAME_CSV,
                            name2driver_csv=_NAME2DRIVER_CSV,
                            bootstrap_dir=None,
                            reset=False,
                            ignore_originals=False,
                            clean_spurious_voxels=False,
                            generate_nrdds=False,
                            relative_symlinks=True,
                            log_duplicate_paths=False):
    """
    This function bootstraps "data release" directories for each of the datasets in our analysis.
    Ideally it should directly query the databases.
    At the moment, we use the files provided by Florian to bootstrap these simple, standalone data repositories.

    It generates the following hierarchy and files:
      bootstrap_dir
       |--T1  # dataset
         |--image2driver.csv                 # "master table" of dataset contents;
         |                                   # it links original db_id with file names and driver lines
         |--image2driver.csv.description     # description of the "master table"
         |--template_regions.h5              # the template image and its segmentation in neuropils are here
         |--template_regions.h5.description  # description
         |--original                         # where the original images are symlinked
           |--images                         # images will be stored
             |--TP10100501L33Sum02.am        # files with such names
             |--...
           |--template.am                    # the template image is here
           |--regions                        # region information for the image (e.g. neuropils)
    """

    # --- Database IDs for the images in our analysis
    #  These are needed to later link to actual filenames, and useful to keep provenance.
    #  Unfortunately this is backwards from Florian's hdf5 files ATM
    #  We do not assume all lines are in all neuropils, so iterate all regions.
    #  Maybe these are not all IDs we need; the way to go should be to run the query ourselves.

    db_ids = []
    for expression_dset, neuropil in product(get_all_datasets(), get_all_neuropils()):
        dset = ExpressionDataset.dataset(dset=expression_dset, neuropil=neuropil)
        db_ids.extend(zip(dset.lines(), [expression_dset] * len(dset.lines())))
    dbids_df = pd.DataFrame(data=sorted(set(db_ids)),
                            columns=['db_id', 'dataset'])
    dbids_df = dbids_df.set_index('db_id')

    # --- db_id -> amira file_name
    #  On a second CSV, we are given the link from a db_id to the amira file name
    #  Probably all these could be done with a simple database query

    # Columns are ['id', 'file_path', 'previews']
    id2file_df = pd.read_csv(id2name_csv, sep=';').sort_values('file_path')
    # Check id uniqueness
    assert len(id2file_df.id) == id2file_df.id.nunique()
    # Rename id -> db_id; file_path -> file_name
    id2file_df = id2file_df.rename(columns=lambda col: {'id': 'db_id', 'file_path': 'file_name'}.get(col, col))
    # Set db_id as the index
    id2file_df = id2file_df.set_index('db_id')
    # Files seems duplicated in the database
    # Fortunately, each file is only once in our analysis, so not a big deal
    if log_duplicate_paths:
        duplicated_paths = id2file_df[id2file_df.file_name.duplicated()].file_name.unique()
        print('Images in more than one entry:\n%s' % '\n'.join(duplicated_paths))

    # Merge
    image2driver_df = pd.merge(id2file_df, dbids_df, left_index=True, right_index=True, how='inner')
    assert len(image2driver_df) == 9484
    assert image2driver_df.file_name.nunique() == 9484

    # --- file_name -> driver line
    #  On a third text file, a weirdo DB dump, we link name to driver line and channel
    #  Andrew deals with it in CAUTION/copy_florians_data.py:my_read_csv

    # This funky function gives a dictionary like this:
    #   {name: {channel: (driver, line number in the CSV)}
    # We will just add name, channel and driver to our dataframe
    n2d = my_read_csv(name2driver_csv, verbose=False)

    # "name" can be inferred from the file_name easily (e.g. TP10100501L33Sum02.am -> TP10100501L33Sum)
    image2driver_df['name'] = image2driver_df.file_name.apply(lambda fp: op.splitext(fp)[0][:-2])

    # Add channel and driver to our dataframe, checking for uniqueness and ambiguities
    def name2driver(name):
        try:
            channel2driver = n2d[name]
            if len(channel2driver) == 0:
                raise Exception('No driver line found for name %s' % name)
            if len(channel2driver) > 1:
                raise Exception('Multiple channels/driver lines found for name %s' % name)
            channel, (driver, _) = channel2driver.popitem()
            return pd.Series({'driver': driver, 'channel': channel})
        except KeyError:
            raise Exception('Cannot find driver line for name %s' % name)
    # add name and driver as dataframe columns
    image2driver_df = image2driver_df.merge(image2driver_df.name.apply(name2driver), left_index=True, right_index=True)
    # make db_id a column
    image2driver_df = image2driver_df.reset_index()
    # reorder, drop name and previews
    image2driver_df = image2driver_df[['db_id', 'dataset', 'file_name', 'channel', 'driver']]
    image2driver_description = dedent("""
    The image2driver.csv file contains a table with an image per row and the following columns:
      - db_id: The ID of the image in the in-house database, for provenance tracking.
      - dataset: The dataset this image pertains to (e.g. T1 (Total Brain 1) or CB1 (Central Brain 1)).
      - file_name: The name of the amira file containing the image.
      - channel: The name of the channel in the image, usually indicating what was expressed (e.g. antiGFP).
      - driver: The driver line ID from flybase.
    """).strip()

    # --- Do release on a per-dataset basis

    bootstrap_dir = braincode_dir(bootstrap_dir, create=False)
    ensure_dir(bootstrap_dir)
    print('Bootstrapping braincode data in directory %s' % op.abspath(bootstrap_dir))

    for dataset, dataset_df in image2driver_df.groupby('dataset'):
        print('Releasing data for dataset %s' % dataset)
        dataset_dir = ensure_dir(op.join(bootstrap_dir, dataset))
        template_regions_h5 = op.join(dataset_dir, 'template_regions.h5')

        # Targeted cleanup
        if reset:
            print('\tRemoving directories and files, please wait...')
            shutil.rmtree(op.join(dataset_dir, 'originals'), ignore_errors=True)
            for fn in ['image2driver.csv', 'image2driver.csv.description',
                       'template_regions.h5', 'template_regions.h5.description']:
                try:
                    os.remove(op.join(dataset_dir, fn))
                except OSError:
                    pass

        # Save the image2driver csv file
        print('\tSaving image2driver.csv')
        image2driver_csv = op.join(dataset_dir, 'image2driver.csv')
        dataset_df.to_csv(image2driver_csv, index=False)
        dataset_df = dataset_df.reset_index(drop=True)
        roundtripped = pd.read_csv(image2driver_csv, index_col=False, dtype={'channel': object})
        assert dataset_df.reset_index(drop=True).equals(roundtripped)
        with open(image2driver_csv + '.description', 'w') as writer:
            writer.write(image2driver_description)

        # Get the location of the original files in strawscience
        originals = _T1_ORIGINALS if dataset == 'T1' else _CB1_ORIGINALS

        # Symlink image files
        if not ignore_originals:
            print('\tSymlinking original amiras')
            original_dest = ensure_dir(op.join(dataset_dir, 'originals', 'images'))
            for file_name in dataset_df.file_name:
                original = op.expanduser(op.join(originals.amiras, file_name))
                if not op.isfile(original):
                    print('WARNING: Cannot find image %s' % original)
                    continue
                dest = op.join(original_dest, file_name)
                if relative_symlinks:
                    original = op.relpath(original, original_dest)
                try:
                    os.symlink(original, dest)
                except OSError as err:
                    print('failing on symlink dest %r' % dest, file=sys.stderr)
                    raise

        # Symlink neuropil files
        if not ignore_originals:
            print('\tSymlinking anatomical regions directory')
            original = op.expanduser(originals.regions)
            if not op.isdir(original):
                print('WARNING: Cannot find regions %s' % original)
            original_dest = op.join(dataset_dir, 'originals', 'regions')
            if relative_symlinks:
                original = op.relpath(original, op.dirname(original_dest))
            os.symlink(original, original_dest)

        # Read regions and save to HDF5
        print('\tMunging regions, please wait...')
        regions_munger = (_t1_regions_munger if dataset == 'T1' else
                          partial(_cb1_regions_munger,
                                  clean_spurious_voxels=clean_spurious_voxels))
        regions = regions_munger()
        regions.to_hdf5(template_regions_h5, dataset_path='regions',
                        compression='gzip', compression_opts=5)

        # Generate template nrrds
        if generate_nrdds:
            print('Generating region nrrds')
            _generate_regions_nrrds(dataset, regions)

        # Symlink template
        if not ignore_originals:
            print('\tSymlinking amira template')
            original = op.expanduser(originals.template)
            if not op.isfile(original):
                print('\tWARNING: Cannot find template %s' % original)
            original_dest = op.join(dataset_dir, 'originals', 'template.am')
            if relative_symlinks:
                original = op.relpath(original, op.dirname(original_dest))
            os.symlink(original, original_dest)

        # Read template and save to hdf5 (probably we could just skip this, for completeness)
        print('\tMunging template, please wait...')
        template, bounding_box = _read_template(op.expanduser(originals.template))
        with h5py.File(template_regions_h5, 'a') as h5:
            dset = h5.create_dataset('template', data=template,
                                     compression='gzip', compression_opts=5)
            dset.attrs['bounding_box'] = bounding_box
            dset.attrs['bounding_box_description'] = 'Bounding box is [xmin, xmax, ymin, ymax, zmin, zmax] micrometers'

        # Save the description of the hdf5 file
        template_regions_description = dedent("""
        This hdf5 file contains the registration template and,
        more importantly, its segmentation into regions/neuropils.

        The template is in dataset "template".
        The values are the intensity of the template image.
        It is a (xsize x ysize x zsize) byte array with attributes:
          - bounding_box: [xmin, xmax, ymin, ymax, zmin, zmax]
          - bounding_box_description: includes units (micrometers)

        The regions is in dataset "regions".
        The values are region id.
        It is a (xsize x ysize x zsize) byte array of "region ids" with attributes:
          - name: an array with the name of each region
          - id: an array with a numeric identifier of each region
          - description: an array with a longer description of each region
          - columns: ['name', 'id', 'description']
          - many 'rs=' attributes, that map region set name to region set labels
        If using python, this can be conveniently read using
        `braincode.revisions.hub.VoxelLabels.from_hdf5`.
        """).strip()
        with open(template_regions_h5 + '.description', 'w') as writer:
            writer.write(template_regions_description)

        print('\t%s done' % dataset)


# Uncomment this to keep developing the clean_spurious_voxels filter
# bootstrap_braincode_dir(generate_nrdds=True, clean_spurious_voxels=True, reset=True)
# exit(22)

if __name__ == '__main__':
    import argh
    argh.dispatch_command(bootstrap_braincode_dir)
