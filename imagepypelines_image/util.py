from .constants import CV2_INTERPOLATION_TYPES, NUMPY_TYPES
from imagepypelines import BlockError

"""
Helper functions that contain canned tests or checks that we will run
frequently
"""
def interpolation_type_check(interp):
    """
    checks to see if the interpolation type is one of the acceptable
    values specified in opencv, otherwise raises a BlockError
    """
    if interp not in CV2_INTERPOLATION_TYPES:
        raise BlockError("Invalid interpolation type")

    return True


def dtype_type_check(dtype):
    """
    checks to see if the interpolation type is one of the acceptable
    values specified in opencv, otherwise raises a BlockError
    """
    if dtype not in NUMPY_TYPES:
        raise BlockError("Invalid Numpy type")

    return True


def channel_type_check(channel_type):
    """checks if the channel_type is one of ("channels_first","channels_last"),
    otherwise raises a BlockError"""
    if channel_type not in ("channels_first","channels_last"):
        raise BlockError("invalid channel type, must be one of ('channels_first','channels_last')")
