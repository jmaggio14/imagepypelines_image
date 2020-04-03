# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imagepypelines/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imagepypelines
#
# Copyright (c) 2018-2020 Jeff Maggio, Ryan Hartzell, and collaborators
from .util import dtype_type_check, interpolation_type_check, channel_type_check
from .imports import import_opencv

import numpy as np
import imagepypelines as ip
import time
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
# [ ] class DrawPoints




"""
__all__ = [
            # Viewing
            'SequenceViewer',
            'NumberImage',
            'DisplaySafe',
            'ImageFFT',
            # Merging/splitting channels
            'ChannelSplit',
            'BaseChannelMerger',
            'MergerFactory',
            'RGBMerger',
            'RGBAMerger',
            'Merger3',
            'Merger4',
            # dtype and dimensions
            'CastTo',
            'Squeeze',
            'Dimensions',
            'FrameSize',
            'Centroid',
            # normalizations
            'NormAB',
            'Norm01',
            'NormDtype',
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
#                               Display
################################################################################
class SequenceViewer(ImageBlock):
    """Display the given images.

    Attributes:
        pause_for(int): the amount of time in milliseconds to pause
            between images

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self, pause_for=30):
        """Instantiates the SequenceViewer

        Arg:
            pause_for(int): the amount of time in milliseconds to pause
                between images. defaults to 30ms
        """
        self.pause_for = int(pause_for)
        super().__init__()
        self.enforce('image', np.ndarray, [(None,None),(None,None,None)])

    ############################################################################
    def preprocess(self):
        """opens the opencv image window"""
        self._open_window()

    ############################################################################
    def postprocess(self):
        """closes the opencv image window"""
        self._close_window()

    ############################################################################
    def process(self, image):
        """Displays the image in a window

        Args:
            img (np.ndarray): image

        Returns:
            None
        """
        cv2.imshow(self.id, image)
        cv2.waitKey(self.pause_for)

    ############################################################################
    def _open_window(self):
        """launches the opencv viewing window"""
        cv2.namedWindow(self.id, cv2.WINDOW_AUTOSIZE)

    ############################################################################
    def _close_window(self):
        """closes the opencv viewing window"""
        cv2.destroyWindow(self.id)


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

#         base_str = \
# """{}) {}
#    type: np.ndarray
#    shapes: [(None,None)]"""
#
#         doc = \
# """Combines independent channels into one Image
#
# Default Enforcement:
#      {}
#
# Batch Size:
#     "each"
# """.format('\n\t'.join(base_str.format(i,cname) for i,cname in enumerate(channel_names)))

        cls_name = "ChannelMerge%s" % n_channels
        channel_merger = type(cls_name,
                                (BaseChannelMerger,),
                                 {'args':args})

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
    """Numbers incoming images. Resets the number index before every process
    run

    Attributes:
        start_at(int): the index to start at for every processing run
        index(int): The image index. The number that will appear in the corner
            of the image
        font_data(dict): dictionary of keyword arguments for cv2.putText


    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self, start_at=1):
        """Instantiates the object

        Args:
            start_at(int): integer to start counting at
        """
        self.start_at = start_at
        self.index = self.start_at
        self.font_data = {'fontFace' : cv2.FONT_HERSHEY_PLAIN,
                            'fontScale' : 1.5,
                            'thickness' : 1,
                            }
        super().__init__()
        self.enforce('image', np.ndarray, [(None,None),(None,None,None)])


    def preprocess(self):
        """resets the number index"""
        self.index = self.start_at

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
        (rect_w, rect_h), _ = cv2.getTextSize(text, **self.font_data)

        end = (width-1,height-1)
        start = (end[0]-rect_w, end[1]-rect_h)
        # rect_color = (0,0,0) if image.ndim > 2 else 0
        # # draw black bounding rectangle
        image = cv2.rectangle(image,
                                start,
                                end,
                                (0,0,0),
                                -1
                                )
        # # draw the white text
        # text_color = (255,255,255) if image.ndim > 2 else 255
        text_org = (end[0]-rect_w, end[1])
        image = cv2.putText(image,
                            text,
                            text_org,
                            color=(255,255,255),
                            **self.font_data
                            )
        self.index += 1
        return image


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


################################################################################
#                               Enhancement
################################################################################

# def lut_factory(func, a=0, b=255):
#
#
# class BaseLUT(ImageBlock):
#     """creates a lookup table from the given function along the range x.
#     func will be called once for every element in the range
#
#
#     Attributes:
#         lut(np.ndarray): the lookup table
#         x_range(np.ndarray): the range over which this lookup table wil
#     """
#     def __init__(self, func, a=0, b=256):
#         # instantiate ImageBlock
#         super().__init__()
#
#         # generate the range
#         x_range = np.arange(a, b)
#
#         # create an empty array to populate
#         self.lut = np.zeros( x_range.size )
#
#         # populate the LUT
#         for i,x in enumerate(x_range.flat):
#             self.lut[i] = func(x)
#
#         # enforce images
#         self.enforce('image', np.ndarray, [(None,None),(None,None,None)])
#
#         self.func = func
#         self.x_range = x_range
#
#     def process(self, image):
#         return self.lut[image]









# ################################################################################
# class IdealFreqFilter(ImageBlock):
#     """Calculates and applies an MTF to a given fft input. Does not perform an
#     fft interally, that must be done upstream.
#     """
#     def __init__(self):
#         pass
