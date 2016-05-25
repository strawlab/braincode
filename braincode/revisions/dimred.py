# coding=utf-8
"""Some dimensionality reduction goodies, geared towards centers visualisation."""
from __future__ import division, print_function

# Avoid matplotlib crashes when sshing or in the cluster
import os
if not os.environ.get('DISPLAY'):
    print('DISPLAY NOT SET: USING AGG BACKEND')
    import matplotlib
    matplotlib.use('agg')

import os.path as op

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.manifold.t_sne import trustworthiness


def cluster_scatter_plot(xs, ys, labels, ax):

    # Add jitter or do some more clever stuff to avoid overplotting
    # For example:
    #   http://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
    # Andrew solves it nicely in the web, and there are many examples around
    # For example:
    #   http://bl.ocks.org/rpgove/10603627

    # invisible points for circle center touchups later
    ax.plot(xs, ys, '.', markersize=1, mfc='white', mec='none', mew=0.0)

    # large circle around text
    ax.plot(xs, ys, 'o', markersize=10, markerfacecolor=(0.1, 1, 0.1, 0.5), mec='black', mew=1.0)

    # text
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, label, horizontalalignment='center', verticalalignment='center')

    # other cosmetics
    ax.set_aspect('equal', 'datalim')
    for getter, setter in [(ax.get_xlim, ax.set_xlim),
                           (ax.get_ylim, ax.set_ylim)]:
        xmin, xmax = getter()
        xmid = (xmin + xmax) / 2
        xradius = (xmax - xmin) / 2
        xmin = xmid - xradius * 1.05
        xmax = xmid + xradius * 1.05
        setter(xmin, xmax)

    return ax


def tsne(D, medoids_df, dest_dir, fn):
    # Reproducing braincode/calculate_cluster_medoids_tSNE
    print('2D TSNE embedding plotting')
    tSNE = TSNE(n_components=2, perplexity=5,
                early_exaggeration=1.0, learning_rate=10.0,
                metric='precomputed', verbose=True, random_state=0)
    medoids2D = pd.DataFrame(tSNE.fit_transform(D), index=medoids_df.index)
    print('Trusty TSNE: %.2f' % trustworthiness(D.values,
                                                medoids2D.values,
                                                n_neighbors=5,
                                                precomputed=True))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    cluster_scatter_plot(medoids2D[0], medoids2D[1],
                         labels=map(str, medoids2D.index),
                         ax=ax)
    plt.savefig(op.join(dest_dir, fn + '.singletons.tsne.png'))

