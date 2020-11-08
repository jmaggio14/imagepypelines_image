from .constants import CV2_INTERPOLATION_TYPES, NUMPY_TYPES
from imagepypelines import BlockError
from .imports import import_opencv
cv2 = import_opencv()

"""
Helper functions that contain canned tests or checks that we will run
frequently
"""


INTERPS = {
            'nearest':cv2.INTER_NEAREST,
            'linear':cv2.INTER_LINEAR,
            'area':cv2.INTER_AREA,
            'cubic':cv2.INTER_CUBIC,
            'lanczos4':cv2.INTER_LANCZOS4,
            }


################################################################################
def interpolation_type_check(interp):
    """
    checks to see if the interpolation type is one of the acceptable
    values specified in opencv, otherwise raises a BlockError
    """
    if interp not in CV2_INTERPOLATION_TYPES:
        raise BlockError("Invalid interpolation type")

    return True

################################################################################
def dtype_type_check(dtype):
    """
    checks to see if the interpolation type is one of the acceptable
    values specified in opencv, otherwise raises a BlockError
    """
    if dtype not in NUMPY_TYPES:
        raise BlockError("Invalid Numpy type")

    return True

################################################################################
def channel_type_check(channel_type):
    """checks if the channel_type is one of ("channels_first","channels_last"),
    otherwise raises a BlockError"""
    if channel_type not in ("channels_first","channels_last"):
        raise BlockError("invalid channel type, must be one of ('channels_first','channels_last')")

################################################################################
def get_cv2_interp_type(interp):
    """fetches the cv2 constant associated with the string interpolation type"""
    if interp in INTERPS:
        return INTERPS[interp]

    raise RuntimeError(f"no interpolation type {interp}, must be one of f{INTERPS.keys()}")




# END
