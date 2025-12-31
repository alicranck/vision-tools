import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

from .types import ImageHandle, Sequence, Union, Tuple, IMG_MODES


def load_image_pil(img_handle: ImageHandle, mode="RGB") -> Image.Image:

    if isinstance(img_handle, np.ndarray):
        verify_image(img_handle, [np.uint8], (0, 255))
        return Image.fromarray(img_handle, mode=mode)

    elif isinstance(img_handle, Image.Image):
        verify_image(img_handle, allowed_modes=IMG_MODES)
        return img_handle.convert(mode=mode)
        
    # assume `input_handle` is a string, Path or file-like object
    elif isinstance(img_handle, (str, Path, io.BytesIO, io.IOBase)):
        verify_image(Image.open(img_handle), allowed_modes=IMG_MODES)
        return Image.open(img_handle).convert(mode=mode)
        
    else:
        raise NotImplementedError(f"load_image_pil() does not support input_handle of type {type(img_handle)}")
    

def load_image_opencv(image_handle: ImageHandle, as_mask: bool = False,) -> np.ndarray:

    if isinstance(image_handle, np.ndarray):
        img = image_handle

    elif isinstance(image_handle, Image.Image):
        img = np.array(image_handle)

    else:
        img = cv2.imread(image_handle) # assume it's a path-like object
        if img.ndim == 3 and img.shape[2] != 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if as_mask:
        if img.ndim == 3 and img.shape[2] != 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.where(img > 0.5, 1, 0).astype(np.uint8)

    assert verify_image(img, allowed_dtypes=[np.uint8], value_range=(0, 255))

    return img

def verify_image(image: Union[np.ndarray, Image.Image], allowed_dtypes: Sequence = None,
                  value_range: Tuple = None,
                     ndim: int = None, allowed_modes: Sequence = None):

    if isinstance(image, Image.Image):
        if allowed_modes is not None:
            assert image.mode in allowed_modes, f"Expected image mode to be one of {allowed_modes} but got " \
                                                f"{image.mode} instead"
        image = np.array(image)

    if allowed_dtypes is not None:
        assert image.dtype in allowed_dtypes, f"Expected image dtype to be one of {allowed_dtypes} but got " \
                                              f"{image.dtype} instead"

    if value_range is not None:
        assert image.min() >= value_range[0] and image.max() <= value_range[1], \
            f"Image contains values outside of requested range {value_range}"

    if ndim is not None:
        assert image.ndim == ndim, f"Expected image to be of {ndim} dimensions but got shape {image.shape}"

    return True


def base64_encode(np_array, image_format="jpg"):
    """
    Converts a NumPy array to a Base64-encoded Data string using OpenCV.
    """
    is_success, buffer = cv2.imencode(f".{image_format}", np_array)
    if not is_success:
        raise ValueError("OpenCV failed to encode the image array.")

    img_bytes = buffer.tobytes()
    base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')

    return base64_encoded_image


def color_histogram(image: np.ndarray, bins: int = 256) -> \
                        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the color histogram for each channel in the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [bins], [0, bins])
    cv2.normalize(hist, hist)
    return hist