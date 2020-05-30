# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imagepypelines/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imagepypelines
#
# Copyright (c) 2018-2020 Jeff Maggio, Ryan Hartzell, and collaborators
from .util import dtype_type_check, interpolation_type_check, channel_type_check
from .imports import import_opencv


import numpy as np
import time
import matplotlib.pyplot as plt
from functools import wraps
cv2 = import_opencv()
import imagepypelines as ip


"""

# [x] class Centroid
# [x] class ExpandDims  --> Unsqueeze
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
# [ ] class DrawCircle
# [ ] class DrawRect


"""
__all__ = [
            # Viewing
            'SequenceViewer',
            'QuickView',
            'CompareView',
            'NumberImage',
            'DisplaySafe',
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
            'Unsqueeze',
            'Dimensions',
            'FrameSize',
            'Centroid',
            # normalizations
            'NormAB',
            'Norm01',
            'NormDtype',
            # Filtering
            'ImageFFT',
            'ImageIFFT'
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

    Attributes:
        order(str): The order of the image axes. Default order is 'HWC', which
            sets (self.h_axis, self.w_axis, self.c_axis) = (0, 1, 2). Allowed
            combinations are ['HWC', 'WHC', 'CWH', 'CHW']
        h_axis(int): Index of the height axis.
        w_axis(int): Index of the width axis.
        c_axis(int): Index of the channel axis.
    """
    def __init__(self, order="HWC"):
        """instantiates the ImageBlock"""
        # NOTE: add default input types
        super().__init__(batch_type="each")
        self.tags.add("imagery")

        # RH:: We need to standardize on vocabulary
        # rows = height; columns = width; depth = bands = channels
        # personally, I think we should alias these with getters and provide a
        # setter utility for axes_order/order a.k.a self.order = "HWC" ->(0,1,2)
        # As a rule, channel axis never ends up in the middle
        order = order.upper()
        if order not in ('HWC','WHC','CWH','CHW'):
            raise ValueError("Value of 'order' keyword argument must be \
                            one of ['HWC','WHC','CWH','CHW']")

        self.h_axis = order.find('H') # height axis
        self.w_axis = order.find('W') # width axis
        self.c_axis = order.find('C') # channel axis
        self.rc_axes = (self.h_axis, self.w_axis)

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
    def __init__(self, pause_for=30, order="HWC"):
        """Instantiates the SequenceViewer

        Arg:
            pause_for(int): the amount of time in milliseconds to pause
                between images. defaults to 30ms
        """
        self.pause_for = int(pause_for)
        super().__init__(order=order)
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
        """Displays the image in an opencv window

        Args:
            image (np.ndarray): image

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
# MATPLOTLIB VIEWERS
class BaseMatplotlibViewer(ImageBlock):
    """Image Viewer that uses matplotlib internally. Nearly always guarenteed
    to work, but timing will be less accurate especially for short timeframes

    This viewer will work with online sphinx-generated examples

    Attributes:
        pause_for(int): the amount of time in milliseconds to pause
            between images
        fig(matplotlib.pyplot.Figure): Figure object for this viewer
        close_fig(bool): whether or not to close the matplotlib figure after
            processing is done. defaults to False

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self, pause_for=500, close_fig=False, order="HWC"):
        """Instantiates the Matplotlib Viewer

        Arg:
            pause_for(int): the amount of time in milliseconds to pause
                between images. defaults to 500ms
            close_fig(bool): whether or not to close the matplotlib figure after
                processing is done. defaults to False
        """
        self.pause_for = pause_for
        self.close_fig = close_fig
        self.fig = None
        self.timer = ip.Timer()
        super().__init__(order=order)

    def preprocess(self):
        self.fig = plt.figure()
        # make this figure interactive
        plt.ion()
        # display it
        plt.show()

    def postprocess(self):
        """closes the matplotlib figure"""
        if self.close_fig:
            plt.close(self.fig)

################################################################################
class QuickView(BaseMatplotlibViewer):
    @wraps(BaseMatplotlibViewer.__init__)
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.enforce('image', np.ndarray, [(None,None),(None,None,None)])

    def process(self, image):
        """Displays the image in a matplotlib figure

        Args:
            image (np.ndarray): image

        Returns:
            None
        """
        # show the image
        plt.imshow( cv2.cvtColor(image, cv2.COLOR_RGB2BGR) )
        # pause after converting to seconds
        plt.pause(self.pause_for / 1000.0)

################################################################################
class CompareView(BaseMatplotlibViewer):
    """Image Viewer that uses matplotlib internally to compare 2 images.
    Nearly always guarenteed to work, but timing will be less accurate
    especially for short timeframes

    This viewer will work with online sphinx-generated examples

    Attributes:
        pause_for(int): the amount of time in milliseconds to pause
            between images

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]
        2) image2
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]
    Batch Size:
        "each"
    """
    @wraps(BaseMatplotlibViewer.__init__)
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.enforce('image', np.ndarray, [(None,None),(None,None,None)])
        self.enforce('image2', np.ndarray, [(None,None),(None,None,None)])

    def preprocess(self):
        self.fig, self.axes = plt.subplots(1, 2)

        # make this figure interactive
        plt.ion()
        # display it
        plt.show()

    def process(self, image, image2):
        """Displays the image in a matplotlib figure

        Args:
            image (np.ndarray): image
            image2 (np.ndarray): second image

        Returns:
            None
        """
        # show the image
        self.axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.axes[0].set_title('Image1')
        self.axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))
        self.axes[1].set_title('Image2')

        plt.pause(self.pause_for / 1000.0)



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
    def __init__(self, order="HWC"):
        """Instantiates the object"""
        super().__init__(order=order)
        self.enforce('image', np.ndarray, [(None,None,None)])

    def process(self, image):
        """splits every channel in the image into separate arrays

        Args:
            image(np.ndarray): image to channel split
        """
        channels = tuple(image[:,:,ch] for ch in range(image.shape[self.c_axis]))
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
    def __init__(self, order="HWC"):
        super().__init__(order=order)

        for arg in self.args:
            self.enforce(arg, np.ndarray, [(None,None)])

    def process(self, *images):
        """merge multiple images into one image with multiple channels

        Args:
            *images(variable length tuple of images):
        """
        return np.stack(images, axis=self.c_axis)


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
        super().__init__(batch_type="each")
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
    """Removes single dimension axes from the array
    """
    def __init__(self):
        super().__init__(batch_type="each")
        self.enforce('arr', np.ndarray, None)

    def process(self, arr):
        return np.squeeze(arr)

class Unsqueeze(ip.Block):
    """Adds single dimension to array at specified  position
    """
    def __init__(self, axis):
        super().__init__(batch_type="each")
        self.enforce('arr', np.ndarray, None)

        self.axis = axis

    def process(self, arr):
        return np.expand_dims(arr, axis=self.axis)


################################################################################
class Dimensions(ImageBlock):
    """Retrieves the dimensions of the image, including number of channels. If
    `channels_none_if_2d` is True, then grayscale images will return
    n_channels = None. Otherwise n_channels will be equal to 1 for grayscale imagery.

    Attributes:
        channels_none_if_2d(bool): whether or not to return n_channels = None for
            grayscale imagery instead of n_channels = 1.

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    def __init__(self, channels_none_if_2d=False, order="HWC"):
        """Instantiates the object

        Args:
            channels_none_if_2d(bool): whether or not to return n_channels = None for
                grayscale imagery instead of n_channels = 1.
        """
        self.channels_none_if_2d = channels_none_if_2d
        super().__init__(order=order)
        self.enforce('image', np.ndarray, [(None,None),(None,None,None)])

    def process(self, image):
        """Retrieves the height, width, and number of channels in the image.

        if `channels_none_if_2d` is True, then grayscale images will return
        n_channels = None, otherwise n_channels = 1 for grayscale images.

        Notes:
            assume image channels are the last axis

        Args:
            image(np.ndarray): the input image

        Returns:
            (tuple): tuple containing:

                height(int): number of rows in image
                width(int): number of columns in image
                n_channels(int): number of channels in image
        """
        # GRAYSCALE CASE
        if image.ndim == 2:
            if self.channels_none_if_2d:
                n_channels = None
            else:
                n_channels = 1
        # MULTIBAND CASE
        else:
            n_channels = image.shape[self.c_axis]

        # fetch the height and width axes using the axes prop
        height = image.shape[self.h_axis]
        width = image.shape[self.w_axis]

        return [height, width, n_channels]


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
    def __init__(self, order="HWC"):
        """Instantiates the object"""
        super().__init__(order=order)
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
    def __init__(self, order="HWC"):
        """Instantiates the object"""
        super().__init__(order=order)
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
    def __init__(self, start_at=1, order="HWC"):
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
        super().__init__(order=order)
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

        super().__init__(batch_type="each")

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
    #     Make another block with a batch_type = "all"
    def __init__(self, order="HWC"):
        """instantiates the fft block"""
        # call super
        super().__init__(order=order)

        # update block tags
        self.tags.add("filtering")

    ############################################################################
    def process(self, image):
        """applies the fft to each channel'

        Args:
            images(np.ndarray): N channel image
        """
        image = np.fft.fft2(image, axes=self.rc_axes)
        return np.fft.fftshift(image, axes=self.rc_axes)


################################################################################
class ImageIFFT(ImageBlock):
    """Performs an IFFT on each Image channel independently

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None), (None,None,None)]

    Batch Size:
        "each"
    """
    # NOTE:
    #     Make another block with a batch_type = "all"
    def __init__(self, order="HWC"):
        """instantiates the ifft block"""
        # call super
        super().__init__(order=order)

        # update block tags
        self.tags.add("filtering")

    ############################################################################
    def process(self, image):
        """applies the ifft to each channel'

        Args:
            images(np.ndarray): N channel image
        """
        image = np.fft.fftshift(image, axes=self.rc_axes)
        return np.fft.ifft2(image, axes=self.rc_axes)



################################################################################
#                               Input/Output
################################################################################
#
# class Decoder(ImageBlock):
#     def __init__(self
# class Encoder(ImageBlock):
#     def __init__(self
# class Writer(ImageBlock):
#     def __init__(self
# class Reader(ImageBlock):
#     def __init__(self
# class VideoWriter(ImageBlock):
#     def __init__(self

#
#

# CONVERT TO
################################################################################
# def convert_to(fname, format, output_dir=None, no_overwrite=False):
#     """converts an image file to the specificed format
#     "example.png" --> "example.jpg"
#
#     Args:
#         fname (str): the filename of the image you want to convert
#         format (str): the format you want to convert to, acceptable options are:
#             'png','jpg','tiff','tif','bmp','dib','jp2','jpe','jpeg','webp',
#             'pbm','pgm','ppm','sr','ras'.
#         output_dir (str,None): optional, a directory to save the reformatted
#             image to. Default = None
#         no_overwrite (bool): whether of not to prevent this function from
#             overwriting a file if it already exists. see
#             :prevent_overwrite:~`imagepypelines.prevent_overwrite` for
#             more information
#
#     Returns:
#         str: the output filename that the converted file was saved to
#     """
#     # eg convert .PNG --> png if required
#     format = format.lower().replace('.','')
#
#     if format not in IMAGE_EXTENSIONS:
#         raise TypeError("format must be one of {}".format(IMAGE_EXTENSIONS))
#
#     file_path, ext = os.path.splitext(fname)
#     if output_dir is None:
#         out_name = file_path + '.' + format
#     else:
#         basename = os.path.basename(file_path)
#         out_name = os.path.join(output_dir, basename + '.' + format)
#
#     if no_overwrite:
#         # check if the file exists
#         out_name = prevent_overwrite(out_name)
#
#     img = cv2.imread(fname)
#     if img is None:
#         raise RuntimeError("Unable to open up file {}".format(fname))
#
#     cv2.imwrite(out_name, img)
#
#     return out_name


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





################################################################################
#                                   Classes
################################################################################
# class CameraCapture(object):
#     """
#     object used to talk to pull imagery from UVC camera (webcam)
#
#     Args:
#         cam (int) = 0:
#             The camera's numerical index (on linux, this number is at the end of
#             the camera's file path eg: "/dev/video0")
#
#         fourcc (str) = "MJPG":
#              the codec used to encode images off the camera. Many UVC
#              camera device achieve highest frame rates with MJPG
#
#     Attributes:
#         cap (cv2.VideoCapture): the cv2 camera object
#         fourcc (str): the fourcc codec used for this camera
#         frame_number (int): the number of frame retrieval attempts
#
#     """
#
#     def __init__(self, cam=0, fourcc="MJPG"):
#         assert isinstance(cam, int), "cam' must be str or int"
#         assert isinstance(fourcc, str), "'fourcc' must be str"
#
#         # openning the camera
#         self.cam = int(cam)
#         self.open()
#
#         # setting the codec
#         self._changeable_settings = {
#             "width": cv2.CAP_PROP_FRAME_WIDTH,
#             "height": cv2.CAP_PROP_FRAME_HEIGHT,
#             "fps": cv2.CAP_PROP_FPS,
#             "brightness": cv2.CAP_PROP_BRIGHTNESS,
#             "contrast": cv2.CAP_PROP_CONTRAST,
#             "hue": cv2.CAP_PROP_HUE,
#             "gain": cv2.CAP_PROP_GAIN,
#             "exposure": cv2.CAP_PROP_EXPOSURE,
#             "fourcc": cv2.CAP_PROP_FOURCC}
#         self.change_setting('fourcc', fourcc)
#         self.fourcc = fourcc
#
#     def open(self):
#         self.cap = cv2.VideoCapture(self.cam)
#         self.frame_number = 0
#
#     def retrieve(self):
#         """
#         reads an image from the capture stream, returns a static debug
#         frame if it fails to read the frame
#
#         Args:
#             None
#
#         Returns:
#             np.ndarray: image frame from the Capture Stream
#         """
#         status = False
#         self.frame_number += 1
#
#         if self.cap.isOpened():
#             status, frame = self.cap.read()
#
#         elif not status or not self.cap.isOpened():
#             debug_message = "unable to read frame {0}, is camera connected?"\
#                 .format(self.frame_number)
#             raise CameraReadError(debug_message)
#
#         return frame
#
#     def metadata(self):
#         """
#         grabs all metadata from the frame using the metadata properties
#         and outputs it in an easy to use dictionary. also adds key
#         "capture_time", which is the time.time() at the time the metadata
#         is collected
#         WARNING - what metadata is available is dependent on what
#         camera is attached!
#
#         Args:
#             None
#
#         Returns:
#             dict: dictionary containing all metadata values
#         """
#         metadata = {
#             "width": self.__get_prop(cv2.CAP_PROP_FRAME_WIDTH),
#             "height": self.__get_prop(cv2.CAP_PROP_FRAME_HEIGHT),
#             "fps": self.__get_prop(cv2.CAP_PROP_FPS),
#             "contrast": self.__get_prop(cv2.CAP_PROP_CONTRAST),
#             "brightness": self.__get_prop(cv2.CAP_PROP_BRIGHTNESS),
#             "hue": self.__get_prop(cv2.CAP_PROP_HUE),
#             "gain": self.__get_prop(cv2.CAP_PROP_GAIN),
#             "exposure": self.__get_prop(cv2.CAP_PROP_EXPOSURE),
#             "writer_dims": (self.__get_prop(cv2.CAP_PROP_FRAME_HEIGHT),
#                             self.__get_prop(cv2.CAP_PROP_FRAME_WIDTH)),
#             "fourcc": self.fourcc,
#             "fourcc_val": self.__get_prop(cv2.CAP_PROP_FOURCC),
#             "capture_time": time.time(),
#             "frame_number": self.frame_number
#         }
#         return metadata
#
#     def change_setting(self, setting, value):
#         """changes a setting on the capture object
#         acceptable
#
#         Args:
#             setting (str): The setting to modify. Must be one of
#                 [width,height,fps,contrast,brightness,hue,gain,
#                 exposure,writer_dims,fourcc,fourcc_val,
#                 capture_time,frame_number]
#             value (variable): The value to switch the setting to
#
#         Returns:
#             None
#         """
#         if setting not in self._changeable_settings:
#             raise ValueError("settings must be one of {0}"
#                     .format(self._changeable_settings.keys()))
#
#
#         flag = self._changeable_settings[setting]
#         if setting == 'fourcc':
#             value = cv2.VideoWriter_fourcc(*value)
#         ret = self.cap.set(flag, value)
#         return ret
#
#     def __get_prop(self, flag):
#         """
#         gets a camera property
#         wrapper for VideoCapture.get function
#
#         Args:
#             flag (opencv constant): flag indicating what metadata to get
#
#         Returns:
#             the camera property requested
#         """
#
#         return self.cap.get(flag)
#
#     def close(self):
#         self.cap.release()
#
#
#
# class Emailer(object):
#     """
#     Goal is to build an object which can be used to automatically send emails
#     after a test or run completes.
#     """
#
#     def __init__(self,
#                 sender,
#                 recipients,
#                 subject="noreply: imagepypelines automated email",
#                 server_name='smtp.gmail.com',
#                 server_port=465):
#         self.subject = subject
#
#         # TODO verify that recipients are valid here
#         # ND: what is the rationale here?
#         # JM: for a line in get_msg: self.current_msg['To'] = ', '.join(self.recipients)
#         # my thinking a list or a single address can be passed in, it's admittedly a lil awk
#         if isinstance(recipients, str):
#             recipients = [recipients]
#
#         self.sender = sender
#         self.recipients = recipients
#         self.subject = subject
#         self.current_msg = None
#
#         self.server_name = server_name
#         self.server_port = server_port
#
#     def get_msg(self):
#         """
#         returns the current email message or creates a new one if one
#         is not already queued
#         """
#         if self.current_msg is not None:
#             return self.current_msg
#
#         self.current_msg = MIMEMultipart('alternative')
#         self.current_msg['Subject'] = self.subject
#         self.current_msg['To'] = ', '.join(self.recipients)
#         self.current_msg['From'] = self.sender
#         return self.current_msg
#
#     def attach(self, filename):
#         """
#         attaches a file to the email message
#         """
#         msg = self.get_msg()
#
#         if not os.path.isfile(filename):
#             iperror("file '{}' does not exist or is inaccessible,\
#                             skipping attachment!".format(filename))
#             return
#
#         with open(filename, 'rb') as fp:
#             msg.attach(fp.read())
#
#     def body(self, text):
#         """
#         sets the body of the current email message
#         """
#         if not isinstance(text, str):
#             iperror("unable to set body because text must be a str,\
#                     currently".format(type(text)))
#             return
#
#         msg = self.get_msg()
#         msg.attach(MIMEText(text, 'plain'))
#
#     def send(self, password=None):
#         """
#         sends the current message and clears the template so a new
#         message can be created
#         """
#         if password is None:
#             password = getpass.getpass()
#
#         msg = self.get_msg()
#
#         server = smtplib.SMTP_SSL(self.server_name, self.server_port)
#         server.ehlo()
#         server.login(self.sender, password)
#         server.send_message(msg)
#
#         self.current_msg = None
#
#     def close(self):
#         pass
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.close()
#
#     def __del__(self):
#         self.close()
#
#
# # class ImageWriter(object):
# #     """
# #     Class that operates as a system that saves single frames to a
# #     specified output directory.
# #
# #     every new frame passed in will be saved in the following format:
# #         output_dir/000001example_filename.png
# #         output_dir/000002example_filename.png
# #         output_dir/000003example_filename.png
# #         ...
# #
# #     automatic resizing is also available
# #
# #     Args:
# #         output_dir (str):
# #             path to output directory that images will be saved to
# #         base_filename (str): default is 'image.png'
# #             filename common among all images, these will be incremented
# #             numerically with each new image saved
# #         size (tuple,None): Default is None
# #             size of the image if forced resizing is desired, or
# #             None if raw write is desired
# #         interpolation (cv2 interpolation type):
# #             Default is cv2.INTER_NEAREST
# #             interpolation method to be used if resizing is desired
# #
# #     """
# #
# #     def __init__(self, output_dir,
# #                          base_filename="image.png",
# #                          size=None,
# #                          interpolation=cv2.INTER_NEAREST):
# #
# #         assert isinstance(base_filename, str), "'base_filename' must be str"
# #         imagepypelines.util.interpolation_type_check(interpolation)
# #
# #         self.base_filename = base_filename
# #         self.size = size
# #         self.interpolation = interpolation
# #         self.image_number = 0
# #
# #     def write(self, frame):
# #         """
# #         writes an image frame to the specificed directory, forces
# #         resizing if specified when the class is instantiated
# #
# #         Args:
# #             frame (np.ndarray): frame to be saved to the output_dir
# #
# #         Returns:
# #             None
# #         """
# #         self.image_number += 1
# #         image_number = imagepypelines.util.make_numbered_prefix(self.image_number,6)
# #         out_filename = os.path.join(self.output_dir,
# #                                     image_number + self.base_filename)
# #
# #         if not isinstance(self.size, type(None)):
# #             frame = cv2.resize(frame,
# #                                dsize=(self.size[1], self.size[0]),
# #                                interpolation=self.interpolation)
# #
# #         cv2.imwrite(filename, frame)
# #
# #
# # class VideoWriter(object):
# #     """
# #     a wrapper class for the cv2 Video Writer:
# #     https://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html#videowriter-fourcc
# #
# #     This class will take a series of single frame imagery and
# #     """
# #
# #     def __init__(self, filename="out_video.avi", fps=30.0, fourcc="XVID"):
# #         self.filename = imagepypelines.util.prevent_overwrite(filename)
# #         self._fourcc = fourcc
# #         self._fourcc_val = cv2.VideoWriter_fourcc(*self._fourcc)
# #         self._fps = float(fps)
# #         self.__is_initialized = False
# #
# #     def __init(self, size):
# #         """
# #         opens and initializes the videowriter
# #         """
# #         imagepypelines.info("initializing the VideoWriter...")
# #         self._h, self._w = size
# #         self.video_writer_kwargs = {"filename": self.filename,
# #                                     "fourcc": self._fourcc_val,
# #                                     "fps": self._fps,
# #                                     "frameSize": (self._w, self._h)
# #                                     }
# #         self.writer = cv2.VideoWriter(**self.video_writer_kwargs)
# #         self.__is_initialized = True
# #
# #     def write(self, frame):
# #         """
# #         writes a frame to the video file.
# #         automatically opens a video writer set to the input frame size
# #
# #         Args:
# #             frame (np.ndarray): input frame to save to file
# #
# #         Returns:
# #             None
# #         """
# #         if not self.__is_initialized:
# #             size = imagepypelines.frame_size(frame)
# #             self.__init(size)
# #
# #         if not self.writer.isOpened():
# #             self.writer.open(**self.video_writer_kwargs)
# #
# #         self.writer.write(frame)
# #
# #     def release(self):
# #         """
# #         closes the video writer
# #
# #         Args:
# #             None
# #
# #         Returns:
# #             None
# #         """
# #         self.writer.release()




# ################################################################################
# class IdealFreqFilter(ImageBlock):
#     """Calculates and applies an MTF to a given fft input. Does not perform an
#     fft interally, that must be done upstream.
#     """
#     def __init__(self):
#         pass
