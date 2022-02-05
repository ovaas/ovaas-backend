import cv2
import itertools
import numpy as np
from PIL import Image

CLASSES_COLOR_MAP = [ (255, 255, 255),(255, 255, 255),(255, 255, 255),(255, 255, 255),
                      (255, 255, 255),(255, 255, 255),(255, 255, 255),(255, 255, 255),
                      (255, 255, 255),(255, 255, 255),(255, 255, 255),(0, 0, 0),
                      (255, 255, 255),(255, 255, 255),(255, 255, 255),(255, 255, 255),
                      (255, 255, 255),(255, 255, 255),(255, 255, 255),(255, 255, 255) ]

# make a segmentation mask image
def make_human_mask(res: list, w: int, h: int) -> Image:

    if len(res.shape) == 3: res = np.expand_dims(res, axis=1)

    *_, out_h, out_w, data = *res.shape, res[0]

    # make a mask image cropped in the shape of a person.
    CLASSES_MAP = np.array([CLASSES_COLOR_MAP[min((int(data[:, i, j])\
                            if len(data[:, i, j]) == 1 else np.argmax(data[:, i, j])), 20)] \
                            for i, j in itertools.product(range(out_h), range(out_w))]).reshape(out_h, out_w, 3)

    # convert to gray scale
    mask = CLASSES_MAP.astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = cv2.resize(mask, (w, h))
    
    return Image.fromarray(mask)

# make a depth mask
def make_mono_mask(output, w: int, h: int) -> Image:

    mask = cv2.resize(output, (w, h))

    # scaling
    mask_min, mask_max = mask.min(), mask.max()

    if mask_max - mask_min > 1e-6: mask=(mask-mask_min)/(mask_max-mask_min)
    else: mask.fill(0.5)

    # convert to gray scale
    mask=(255-mask*255).astype(np.uint8)
    
    return Image.fromarray(mask)