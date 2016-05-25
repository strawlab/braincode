# coding=utf-8
import os
import os.path as op
from functools import partial
from braincode.util import ensure_dir, braincode_basedir

# --- Custom imports


class _Noop(object):

    def noop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.noop
_noop = _Noop()


class NoopI(object):
    def noopi(*args, **_):
        return args[0]

    def __getattr__(self, _):
        return self.noopi
_noopi = NoopI()

try:
    import mkl
except ImportError:
    mkl = _noop

try:
    import numba  # do not add to setup.py unless we strongly recommend conda
except ImportError:
    numba = _noopi

# --- Utils


def _find_best_candidate(candidates, ensure_exists=False):
    for candidate in candidates:
        if candidate is not None:
            if not ensure_exists or op.isdir(op.expanduser(candidate)):
                return candidate
    return None


def _dir_or_default(dire=None, default=None, create=False):
    dire = default if dire is None else dire
    return ensure_dir(dire) if create else dire


# --- Strawlab central repository

_STRAWLAB_ROOT_CANDIDATES = (
    os.environ.get('STRAWLAB_ROOT', None),  # always configurable...
    '~/strawscience',                       # cluster and laptops
    '/mnt/strawscience',                    # other machines
)

STRAWLAB_ROOT = _find_best_candidate(candidates=_STRAWLAB_ROOT_CANDIDATES)
strawlab_root = partial(_dir_or_default, default=STRAWLAB_ROOT)


# --- "Relocatable" project layout
#  See related braincode/util.py:braincode_basedir(); this is a rework wannabe

_BRAINCODE_DIR_CANDIDATES = (
    os.environ.get('BRAINCODE_DIR', None),  # always configurable...
    op.join(STRAWLAB_ROOT, 'santi', 'andrew', 'clustering-paper', 'braincode')
)


def braincode_dir(dire=None, create=False):
    if dire is None:
        dire = _find_best_candidate(_BRAINCODE_DIR_CANDIDATES, ensure_exists=False)
    dire = op.expanduser(dire)
    return ensure_dir(dire) if create else dire

# Overview of paths:
# - BRAINCODE_SUP_DATA_DIR: supplementary data gets directly packaged from here.
#     Do not save to here unless you want it packaged with the paper.
# - BRAINCODE_DATA_CACHE_DIR: base location for files used as intermediate steps
#     in the computational pipeline.

# A separate step takes files from these two directories and packages into a
# webserver assets directory.

BRAINCODE_CODE_DIR = op.abspath(op.join(op.dirname(__file__), '..', '..'))
BRAINCODE_PACKAGE_DIR = op.join(BRAINCODE_CODE_DIR, 'braincode')
BRAINCODE_ASSETS_DIR = op.join(BRAINCODE_CODE_DIR, 'assets')
BRAINCODE_DATA_CACHE_DIR = op.join(BRAINCODE_CODE_DIR, 'data-cache')
BRAINCODE_SUP_DATA_DIR = braincode_basedir() # e.g. '<path>/Panser_et_al_supplemental_data_and_code/clustering-data'
