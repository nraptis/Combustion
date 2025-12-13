from enum import Enum, auto

class ConvolutionEdgeBehavior(Enum):
    """
    Defines how convolution handles samples that fall outside the bitmap.
    """

    TRIM = auto()
    """
    Do not sample outside the bitmap.
    Output image is smaller (valid convolution).
    """

    COPY = auto()
    """
    Clamp coordinates to the nearest valid pixel.
    (Edge pixels are repeated outward.)
    """

    BLACK = auto()
    """
    Treat out-of-bounds samples as black (0,0,0,0).
    """
