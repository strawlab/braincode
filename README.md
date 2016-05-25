# braincode

This is the supplementary data and code for the paper:

Panser K, Tirian L, Schulze F, Villalba S, Jefferis GSXE, Bühler K, Straw AD.
(2016) Automatic segmentation of Drosophila neural compartments using GAL4
expression data reveals novel visual pathways. Current Biology.

Please check our website at https://strawlab.org/braincode for updates and an
interactive data browser. Updated versions of this source code may be found at https://github.com/strawlab/braincode.

This directory contains three subdirectories:

* `braincode` - the source code
* `clustering-data` - the results of the published clusterings and agglomerations
* `3d-models` - models of the segmented VPNs, optic glomeruli and so on.

## Clustering data

There are three types of data files included in the `clustering-data`
directory:

* Samples HDF5 file (`data/<dataset>/<region>/<region>_samples.h5`)
* Clustering result volume file (`data/<dataset>/<region>/<clustering_type>/<region>_clusterimage.nrrd`)
* Cross-reference file (`data/<dataset>/id_driver_image.csv`)

### Samples HDF5 file

These .h5 source files are in the
[HDF5](https://www.hdfgroup.org/HDF5/) format. They contain the
expression data used for the clustering. They have been subsampled
from the original registered confocal stacks and have been taken from
the brain region on which the clustering was performed.

Internally, the structure of each .h5 file is

```
/
├── ids
│   ├── 247_217_115
│   ├── 247_217_118
│   └── <x>_<y>_<z>
├── positionKeys
├── size
├── stepXY
└── stepZ
```

The `ids` group contains many individual datasets named `<x>_<y>_<z>`,
where `<x>`, `<y>`, and `<z>` are the coordinates of the sampled
voxel. Each dataset is a list of ids whereby each id corresponds to a
particular confocal stack and driver line. The criterion for an id to
be listed is that the expression in that stack must exceed a threshold
value in that voxel.

The `positionKeys` dataset is a comma separated string of all
considered positions within the given region. The coordinates are not
in the downsampled space.

The `size` dataset species the number of voxels in the original
(not-downsampled) volume.

The `stepXY` and `stepZ` datasets specify the amount by which the
resulting clustering result volume (`.nrrd` file) has been
downsampled.

### Clustering result volume file

These .nrrd files contain the volumetric coordinates of each cluster
identified by the clustering algorithm in the [NRRD
format](http://teem.sourceforge.net/nrrd/format.html). Each voxel is
assigned zero (for no cluster) or an integer that defintes the cluster
number to which the voxel belongs.

### Cross-reference file

This file contains the identity between an integer id, the file name
of a confocal stack corresponding to that id, and the driver line from
which the confocal stack was made.

## Source code

- `kmedoids_salspaugh.py` The kmedoids algorithm
- `dice.py` The dice coefficient algorithm
- `calculate_distance.py` Computes the voxel-to-voxel distance matrix for a given brain region
- `perform_clustering.py` Run the kmedoids clustering algorithm
- `util.py` Various utilities
- `plot_distance_matrix.py` Create a plot showing the voxel-to-voxel distance matrix for a given clustering result
- `fragments_per_cluster_step1_compute.py` Compute which driver lines are expressed in which clusters
- `fragments_per_cluster_step2_save_csv.py` Save CSV file with the data per driver line
- `fragments_per_cluster_step3_csv_to_json.py` Convert CSV file with all rows to a JSON file with only significant rows
- `calculate_cluster_stats.py` Measure distances between medoids (inter-cluster) and between voxels in cluster (intra-cluster)
- `save_cluster_info_json.py` Save cluster statistics
- `stability.py` Evaluate the stability of clusterings
