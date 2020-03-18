# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imagepypelines/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imagepypelines
#
# Copyright (c) 2018-2020 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from .util import dtype_type_check, interpolation_type_check, channel_type_check
from .imports import import_opencv

import numpy as np
import imagepypelines as ip
cv2 = import_opencv()

DEFAULT_CHANNEL_TYPE = "channels_last"
"""default channel axis for all images, defaults to 'channels_last'"""


"""

# [x] class Centroid
# [ ] class ExpandDims
# [x] class Squeeze
# [x] class FrameSize
# [ ] class Viewer - with a frame_counter
# [x] class NumberImage
# [ ] class SwapAxes
# [ ] class ConvertColor
# [ ] class HistogramEnhance
# [ ] class HistogramMatch
# [ ] class HistogramEnhance
# [ ] class KeypointFactory???
# [ ] class HarrisCorner
# [ ] class FastCorner




"""
__all__ = [
            # UTIL
            'ChannelSplit',
            'MergerFactory',
            'RGBMerger',
            'RGBAMerger',
            'Merger3',
            'Merger4',
            'CastTo',
            # NORMALIZATION
            'NormAB',
            'Norm01',
            'NormDtype',
            'DisplaySafe',
            # FILTERING
            'ImageFFT',
            # 'IdealFreqFilter',
            ]

# ImageBlock
# ChannelSplit
# MergerFactory
# for
# CastTo
# Squeeze
# Dimensions
# FrameSize
# Centroid
# NumberImage
# NormAB
# Norm01
# NormDtype
# DisplaySafe
# ImageFFT
#

class ImageBlock(ip.Block):
    """Special Block made for imagery with a predefined IO inputs and useful
    properties

    """
    def __init__(self):
        """instantiates the ImageBlock"""
        # NOTE: add default input types
        super().__init__(batch_size="each")
        self.tags.add("imagery")
        self.h_axis = 0 # height axis
        self.w_axis = 1 # width axis
        self.b_axis = 2 # band axis


################################################################################
#                               Util
################################################################################
class ChannelSplit(ImageBlock):
    """splits images into separate component channels

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self):
        """Instantiates the object"""
        super().__init__()
        self.enforce('image', np.ndarray, [(None,None,None)])

    def process(self, image):
        """splits every channel in the image into separate arrays

        Args:
            image(np.ndarray): image to channel split
        """
        channels = tuple(image[:,:,ch] for ch in range(image.shape[self.b_axis]))
        return channels


################################################################################
# Array Merging
class BaseChannelMerger(ImageBlock):
    """combines independent channels into one Image

    Default Enforcement:
         image
            type: np.ndarray
            shapes: [(None,None)]

    Batch Size:
        "each"
    """
    def __init__(self):
        super().__init__()

        for arg in self.args:
            self.enforce(arg, np.ndarray, [(None,None)])

    def process(self, *images):
        """merge multiple images into one image with multiple channels

        Args:
            *images(variable length tuple of images):
        """
        return np.stack(images, axis=self.b_axis)


class MergerFactory(object):
    def __new__(cls, n_channels, channel_names=None):
        """fetches a new channel merger object

        Args:
            n_channels(int): number of channels
            channel_names(tuple,list,None): Optional, the names of the channels to
                be merged. If left as None, then undescriptive names will be
                generated.

        Returns:
            :obj:`ChannelMerger`: new channel merger class for the given number of
                channels
        """
        # check if channel names are the correct length
        if channel_names:
            if len(channel_names) != n_channels:
                msg = "'channel_names' must be a tuple of length %s" % n_channels
                raise RuntimeError(msg)
            args = channel_names
        else:
            args = ["channel%s" % i for i in range(n_channels)]

        base_str = \
"""{}) {}
   type: np.ndarray
   shapes: [(None,None)]"""

        doc = \
"""Combines independent channels into one Image

Default Enforcement:
     {}

Batch Size:
    "each"
""".format('\n\t'.join(base_str.format(i,cname) for i,cname in enumerate(channel_names)))

        cls_name = "ChannelMerge%s" % n_channels
        channel_merger = type(cls_name,
                                (BaseChannelMerger,),
                                 {'args':args,
                                 '__doc__':doc})

        return channel_merger

# generate example Merger Classes
RGBMerger = MergerFactory(3, ["Red", "Green", "Blue"])
RGBAMerger = MergerFactory(4, ["Red", "Green", "Blue","Alpha"])
Merger3 = MergerFactory(3)
Merger4 = MergerFactory(4)


################################################################################
class CastTo(ip.Block):
    """casts arrays to a given numpy dtypes

    Attributes:
        cast_type (:obj:`numpy.dtype`): np.dtype the final array is casted to

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self, cast_type):
        """Instantiates the object

        Args:
            cast_type (:obj:`numpy.dtype`): np.dtype to case the array to
        """
        # Arg Checking
        # cast_type must be a NUMPY type
        dtype_type_check(cast_type)

        # instance variables
        self.cast_type = cast_type
        super().__init__(batch_size="each")
        self.enforce('arr', np.ndarray, [(None,None),(None,None,None)])

    ############################################################################
    def process(self, arr):
        """casts array to given cast type

        Args:
            arr(:obj:`numpy.ndarray`): numpy array of any shape

        Returns:
            :obj:`numpy.ndarray`: array of same shape casted to new dtype
        """
        return arr.astype(self.cast_type)


################################################################################
class Squeeze(ip.Block):
    """Removes single dimension axes from the
    """
    def __init__(self):
        super().__init__()
        self.enforce('arr', np.ndarray, None)

    def process(self, arr):
        return np.squeeze(arr)


################################################################################
class Dimensions(ImageBlock):
    """Retrieves the dimensions of the image, including number of bands. If
    `bands_none_if_2d` is True, then grayscale images will return
    n_bands = None. Otherwise n_bands will be equal to 1 for grayscale imagery.

    Attributes:
        bands_none_if_2d(bool): whether or not to return n_bands = None for
            grayscale imagery instead of n_bands = 1.

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self, bands_none_if_2d=False):
        """Instantiates the object

        Args:
            bands_none_if_2d(bool): whether or not to return n_bands = None for
                grayscale imagery instead of n_bands = 1.
        """
        self.bands_none_if_2d = bands_none_if_2d
        super().__init__()
        self.enforce('image', np.ndarray, [(None,None),(None,None,None)])

    def process(self, image):
        """Retrieves the height, width, and number of bands in the image.

        if `bands_none_if_2d` is True, then grayscale images will return
        n_bands = None, otherwise n_bands = 1 for grayscale images.

        Notes:
            assume image bands are the last axis

        Args:
            image(np.ndarray): the input image

        Returns:
            (tuple): tuple containing:

                height(int): number of rows in image
                width(int): number of columns in image
                n_bands(int): number of bands in image
        """
        # GRAYSCALE CASE
        if image.ndim == 2:
            if self.bands_none_if_2d:
                n_bands = None
            else:
                n_bands = 1
        # MULTIBAND CASE
        else:
            n_bands = image.shape[self.b_axis]

        # fetch the height and width axes using the axes prop
        height = image.shape[self.h_axis]
        width = image.shape[self.w_axis]

        return height, width, n_bands


################################################################################
class FrameSize(ImageBlock):
    """Retrieves the size of the image frame

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self):
        """Instantiates the object"""
        super().__init__()
        self.enforce('image', np.ndarray, [(None,None),(None,None,None)])

    def process(self, image):
        """Retrieves the height and width of the image.

        Args:
            image(np.ndarray): the input image

        Returns:
            (tuple): tuple containing:

                height(int): number of rows in image
                width(int): number of columns in image
        """
        return image.shape[self.h_axis], image.shape[self.w_axis]


################################################################################
class Centroid(ImageBlock):
    """Retrieves the central pixel in the image, rounds down to integer
    if an image dimension has an odd length

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self):
        """Instantiates the object"""
        super().__init__()
        self.enforce('image', np.ndarray, [(None,None),(None,None,None)])

    def process(self, image):
        """Retrieves the central row and column in an image.

        Args:
            image(np.ndarray): the input image

        Returns:
            (tuple): tuple containing:

                center_h(int): central row index
                center_w(int): central col index
        """
        return image.shape[self.h_axis]//2, image.shape[self.w_axis]//2


################################################################################
class NumberImage(ImageBlock):
    """Numbers incoming images

    Attributes:
        text_origin(:obj:`tuple` of :obj:`int`):
        index(int): frame number index

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self, text_origin=(.9,.9), start_at=1):
        """Instantiates the object

        Args:
            text_origin(:obj:`tuple` of :obj:`float`): the origin of the frame
                number as fraction of the image dimensions.
                start_at(int): integer to start counting at
        """
        self.text_origin = text_origin
        self.index = start_at
        self.enforce('image', np.ndarray, [(None,None),(None,None,None)])

    def process(self, image):
        """Adds a number to the corner of an image

        Args:
            img (np.ndarray): image

        Returns:
            :obj:`numpy.ndarray`: numbered image
        """
        width = image.shape[self.w_axis]
        height = image.shape[self.h_axis]

        # make text and bounding rectangle
        text = str(self.index)
        rect_w = len(text) * 16
        rect_h = 16


        start = (self.text_origin[1]*width, self.text_origin[0]*height)
        end = (min(width, start[0]+rect_w), min(height, start[1]+rect_h))

        # draw black bounding rectangle
        image = cv2.rectangle(image,
                                start_point=loc,
                                end_point=end,
                                color=(0,0,0),
                                thickness=-1
                                )
        # draw the white text
        image = cv2.putText(image,
                            text=str(self.index),
                            org=start,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=.5,
                            color=(255,255,255),
                            thickness=2,
                            bottomLeftOrigin=False)
        self.index += 1


################################################################################
#                               Normalization
################################################################################
class NormAB(ip.Block):
    """normalizes to range [a,b] and then casts to given cast_type

    If a < b, then the histogram will just be scaled and shifted.
    If b < a, then the histogram will be flipped left-right, scaled, and shifted.

    Attributes:
        a (int,float): minimum of normalization range
        b (int,float): maximum of normalization range
        cast_type (:obj:`numpy.dtype`): np.dtype the final array is casted to. default
            is float64

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: None

    Batch Size:
        "each"
    """
    def __init__(self, a, b, cast_type=np.float64):
        # INPUT CHECKING
        if not isinstance(a, (int,float)):
            raise ip.BlockError("'a' must be an integer or float")

        if not isinstance(b, (int,float)):
            raise ip.BlockError("'b' must be an integer or float")

        # cast_type must be a NUMPY type
        dtype_type_check(cast_type)

        # instance variables
        self.a = a
        self.b = b
        self.cast_type = cast_type

        super().__init__(batch_size="each")

        self.enforce('arr', np.ndarray, None)

    ############################################################################
    def process(self, arr):
        """normalizes to given range and cast

        Args:
            arr(:obj:`numpy.ndarray`): array of any shape and type

        Returns:
            :obj:`numpy.ndarray`: normalized array of same shape and casted
                to the given type
        """
        arr = arr.astype(np.float64)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = (arr * (self.b - self.a)) + self.a
        return arr.astype(self.cast_type)


################################################################################
class Norm01(NormAB):
    """normalizes to [0,1] and then casts to given cast_type

    Can be used to prepare images for file output.
    Equivalent to a 0% histogram stretch.
    Works by converting to float64, then stretching/shifting, then quantizing.

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: None

    Batch Size:
        "each"

    """
    def __init__(self, cast_type=np.float64):
        super().__init__(0, 1, cast_type)


################################################################################
class NormDtype(NormAB):
    """normalizes to [dtype_min, dtype_max] and then casts to given cast_type

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: None

    Batch Size:
        "each"

    """
    def __init__(self, cast_type=np.float64):
        dtype_info = np.iinfo(cast_type)
        super().__init__(type_info.min, dtype_info.max, cast_type)


################################################################################
class DisplaySafe(NormAB):
    """normalizes to [0,255] and bins to a displayable bitdepth

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self):
        super().__init__(0, 255, cast_type=np.uint8)


################################################################################
#                               Filtering
################################################################################
class ImageFFT(ImageBlock):
    """Performs an FFT on each Image channel independently

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    # NOTE:
    #     Make another block with a batch_size = "all"
    def __init__(self):
        """instantiates the fft block"""
        # call super
        super().__init__(channel_type)

        # update block tags
        self.tags.add("filtering")

    ############################################################################
    def process(self, image):
        """applies the fft to the axes specified by 'channel_type'

        Args:
            images(np.ndarray): N channel image
        """
        return np.fft.ftt2(image, axes=(self.h_axis,self.w_axis))
#
#
# ################################################################################
# class IdealFreqFilter(ImageBlock):
#     """Calculates and applies an MTF to a given fft input. Does not perform an
#     fft interally, that must be done upstream.
#     """
#     def __init__(self):
#         pass
