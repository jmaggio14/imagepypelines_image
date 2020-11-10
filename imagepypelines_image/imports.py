# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imagepypelines/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imagepypelines
#
# Copyright (c) 2018 - 2020 Jeff Maggio, Jai Mehra, Ryan Hartzell
#
import sys
from imagepypelines import get_master_logger


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
        get_master_logger().error("imagepypelines_image requires opencv to be installed separately")
        get_master_logger().error("try 'pip install imagepypelines_image[cv]' or install OpenCV from source (see https://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.html)")
        sys.exit(1)

    return cv2
