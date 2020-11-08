from .util import dtype_type_check,
                    interpolation_type_check,
                    channel_type_check,
                    get_cv2_interp_type,
from .imports import import_opencv
from .blocks import ImageBlock

cv2 = import_opencv()
import numpy as np
import imagepypelines as ip


class Resize(ImageBlock):
    """splits images into separate component channels

    Attributes:
        w_scale_type(str): type of scaling used for image width, either
            "proportional" or "absolute"
        h_scale_type(str): type of scaling used for image height, either
            "proportional" or "absolute"
        h_param(int,float): vertical scale or absolute height to resize
            image to
        w_param(int,float): horizontal scale or absolute height to resize
            image to
        interp(str): interpolation type for resizing. One of
            'nearest', 'linear', 'area', 'cubic', 'lanczos4'

    Default Enforcement:
        1) image
            type: np.ndarray
            shapes: [(None,None,None),(None,None)]
            notes: image must be ordered [height,width,channels]

    Batch Size:
        "each"
    """
    def __init__(self, h=None, w=None, scale_h=None, scale_w=None, interp='nearest'):
        """Instantiates the object

        Args:
            w(None,int): width to scale image to, must be None is scale_w is
                defined
            h(None,int): height to scale image to, must be None is scale_h is
                defined
            scale_h(None,float): vertical scale for the image, must be None
                if 'h' is defined
            scale_w(None,float): horizontal scale for the image, must be None
                if 'w' is defined
            interp(str): interpolation type for image scaling, must be one of:
                'nearest', 'linear', 'area', 'cubic', 'lanczos4'

        """
        super().__init__(order="HWC")

        # make sure either h or scale_h is defined
        if (h is None) and (scale_h is None):
            raise ValueError("'h' or 'scale_h' must be defined")

        # make sure either w or scale_w is defined
        if (w is None) and (scale_w is None):
            raise ValueError("'w' or 'scale_w' must be defined")

        # make sure only h or scale_h is defined
        if (not h is None) and (not scale_h is None):
            raise ValueError("only 'h' or 'scale_h' can be defined")

        # make sure only w or scale_w is defined
        if (not w is None) and (not scale_w is None):
            raise ValueError("only 'w' or 'scale_w' can be defined")

        # set w instance variables
        if w is None:
            self.w_scale_type = 'proportional'
            self.w_param = scale_w
        else:
            self.w_scale_type = 'absolute'
            self.w_param = w

        # set h instance variables
        if h is None:
            self.h_scale_type = 'proportional'
            self.h_param = scale_h
        else:
            self.h_scale_type = 'absolute'
            self.h_param = h


        self.__cv2_interp = get_cv2_interp_type(interp)
        self.interp = interp

        self.enforce('image', np.ndarray, [(None,None,None),(None,None)])

    def process(self, image):
        """Resizes the image to the specified dimensions

        Args:
            image(np.ndarray): image to resize, must be shaped
                [height,width,channels]

        Returns:
            np.ndarray: resized image
        """
        # get h dimension
        if self.h_scale_type == "proportional":
            new_h = round(self.h_param * image.shape[0], 0)
        else:
            new_h = self.h_param

        # get w dimension
        if self.w_scale_type == "proportional":
            new_w = round(self.w_param * image.shape[1], 0)
        else:
            new_w = self.w_param

        return cv2.rezize(image, (new_w,new_h), interpolation=self.__cv2_interp)








    # END
