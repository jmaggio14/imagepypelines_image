# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imagepypelines/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imagepypelines
#
# Copyright (c) 2018-2020 Jeff Maggio, Nathan Dileas, Ryan Hartzell
#
import sys
from imagepypelines import error as iperror


def import_opencv():
    """Direct opencv imports are discouraged for imagepypelines developers
    because it is not automatically installed alongside imagepypelines, and
    therefore may cause confusing errors to users.

    This function will check if opencv is installed and import it if
    possible. If opencv is not importable, it will print out installation
    instructions.

    Returns:
        module: module reference to opencv
    """
    try:
        import cv2
    except ImportError:
        iperror("imagepypelines requires opencv to be installed separately!")
        iperror("see README: https://github.com/jmaggio14/imagepypelines")
        sys.exit(1)

    return cv2
