# pylint: disable-msg=W0614,W0401,W0611,W0622

# flake8: noqa

__docformat__ = 'restructuredtext'

# check required packages

"""
hard_dependencies = ("numpy", "scipy", "pandas", "pims", "tqdm", "tifffile")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError(
            "Missing required dependencies {0}".format(missing_dependencies))
del hard_dependencies, dependency, missing_dependencies
"""

from imlib.base import ImageBase
from imlib.base_filters import ImageFilter
from imlib.base_lines import ImageLine
from imlib.base_features import ImageFeature

from imlib.imgfolder import ImgFolder
from imlib.imglog import ImgLog

from imlib.meta import Meta
from imlib.lineobject import LineObject

__version__ = '1.1.0.dev1'

__doc__ = """
imlib - image process library for IBM Nanobiotechnology team
=============================================================

**imlib** is a python package providing

Main Features
-------------

  - Folder structure
  - Process many images with frames
"""
