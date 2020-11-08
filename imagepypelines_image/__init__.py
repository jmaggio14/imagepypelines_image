# --------- retrieve the source directory for our standard images ---------
import pkg_resources
STANDARD_IMAGE_DIRECTORY = pkg_resources.resource_filename(__name__,
                                                        'data/standard_images')
"""the location where imagepypelines standard images are stored"""
del pkg_resources

from .data import standard_image_filenames
from .data import standard_image_gen
from .data import standard_images
from .data import get_standard_image

from .data import STANDARD_IMAGES
from .data import funcs

import sys

curr_module = sys.modules[__name__]
for img_name in STANDARD_IMAGES.keys():
	setattr(curr_module, img_name, getattr(funcs, img_name))

# ND 9/7/18 - delete these so that the imagepypelines namespace is not polluted
del sys, curr_module, funcs, STANDARD_IMAGES


from .blocks import *
from .Resize import Resize
