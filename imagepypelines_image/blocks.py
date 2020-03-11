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

__all__ = [
            'ChannelSplit',
            'MergerFactory',
            'RGBMerger',
            'RGBAMerger',
            'Merger3',
            'Merger4',
            'CastTo',
            'NormAB',
            'Norm01',
            'NormDtype',
            'DisplaySafe',
            # 'ImageFFT',
            # 'IdealFreqFilter',
            ]


class ImageBlock(ip.Block):
    """Special Block made for imagery with a predefined IO inputs and useful
    properties

    Attributes:
        channel_type(str): channel_type(str): channel_type, either
            "channels_first" or "channels_last"
    """
    def __init__(self, channel_type):
        """instantiates the ImageBlock

        Args:
            channel_type(str): channel_type(str): channel_type, either
                "channels_first" or "channels_last"
        """
        # check the channel type
        channel_type_check(channel_type)
        self.channel_type = channel_type

        # NOTE: add default input types

        super().__init__(batch_size="singles")
        self.tags.add("imagery")

    ############################################################################
    @property
    def axes(self):
        if self.channel_type == "channels_first":
            # (C,H,W) - we want to transform the last two axes
            return (-2,-1)
        elif self.channel_type == "channels_last":
            # (H,W,C) - we want to transform the first two axes
            return (-3,-2)

    @property
    def channel_axis(self):
        if self.channel_type == "channels_first":
            # (C,H,W) - we want the first axis
            return 0
        elif self.channel_type == "channels_last":
            # (H,W,C) - we want the last axis
            return -1


################################################################################
#                               Util
################################################################################
class ChannelSplit(ImageBlock):
    # NOTE: ADD EXAMPLES
    """splits the image into it's component channels"""
    def process(self, image):
        """splits every channel in the image into separate arrays

        Args:
            image(np.ndarray): image to channel split
        """
        n_channels = image.shape[self.channel_axis]
        if self.channel_type == "channels_last":
            channels = tuple(image[Ellipsis,ch] for ch in range(n_channels))
        else: # self.channel_type == channels_first
            # this line will have to be updated if batching is supported
            channels = tuple(image[ch,Ellipsis] for ch in range(n_channels))

        return channels


################################################################################
# Array Merging
class BaseChannelMerger(ImageBlock):
    """combines independent channels into one Image"""
    def process(self, *images):
        """merge multiple images into one image with multiple channels

        Args:
            *images(variable length tuple of images):
        """
        return np.stack(images, axis=self.channel_axis)


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
        cls_name = "ChannelMerge%s" % n_channels
        channel_merger = type(cls_name, (BaseChannelMerger,), {'args':args})
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

    Batch Size:
        "all"
    """
    def __init__(self, cast_type):
        # Arg Checking
        # cast_type must be a NUMPY type
        dtype_type_check(cast_type)

        # instance variables
        self.cast_type = cast_type
        super().__init__(batch_size="singles")

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

    Batch Size:
        "singles"
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

        super().__init__(batch_size="singles")

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
    """
    def __init__(self, cast_type=np.float64):
        super().__init__(0, 1, cast_type)


################################################################################
class NormDtype(NormAB):
    """normalizes to [dtype_min, dtype_max] and then casts to given cast_type
    """
    def __init__(self, cast_type=np.float64):
        dtype_info = np.iinfo(cast_type)
        super().__init__(type_info.min, dtype_info.max, cast_type)


################################################################################
class DisplaySafe(NormAB):
    """normalizes to [0,255] and bins to a displayable bitdepth"""
    def __init__(self):
        super().__init__(0, 255, cast_type=np.uint8)


################################################################################
#                               Filtering
################################################################################
# class ImageFFT(ImageBlock):
#     """Performs an FFT on each Image channel independently"""
#     # NOTE:
#     #     Make another block with a batch_size = "all"
#     def __init__(self, channel_type=DEFAULT_CHANNEL_TYPE):
#         """instantiates the fft block
#
#         Args:
#             channel_type(str): channel_type, either "channels_first" or
#                 "channels_last"
#         """
#         # call super
#         super().__init__(channel_type)
#
#         # update block tags
#         self.tags.add("filtering")
#
#     ############################################################################
#     def process(self, images):
#         """applies the fft to the axes specified by 'channel_type'
#
#         Args:
#             images(np.ndarray): N channel image
#         """
#         return np.fft.ftt2(image, axes=self.axes)
#
#
# ################################################################################
# class IdealFreqFilter(ImageBlock):
#     """Calculates and applies an MTF to a given fft input. Does not perform an
#     fft interally, that must be done upstream.
#     """
#     def __init__(self):
#         pass
