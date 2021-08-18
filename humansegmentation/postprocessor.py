import itertools
import numpy as np
from PIL import Image

CLASSES_COLOR_MAP = [ (255, 255, 255),(255, 255, 255),(255, 255, 255),(255, 255, 255),
                      (255, 255, 255),(255, 255, 255),(255, 255, 255),(255, 255, 255),
                      (255, 255, 255),(255, 255, 255),(255, 255, 255),(0, 0, 0),
                      (255, 255, 255),(255, 255, 255),(255, 255, 255),(255, 255, 255),
                      (255, 255, 255),(255, 255, 255),(255, 255, 255),(255, 255, 255) ]

# make segmentation mask image
def segmentation(res: list, w: int, h: int) -> Image:

    if len(res.shape) == 3: res = np.expand_dims(res, axis=1)

    *_, out_h, out_w, data = *res.shape, res[0]

    # make a mask image cropped in the shape of a person.
    CLASSES_MAP = np.array([CLASSES_COLOR_MAP[min((int(data[:, i, j])\
                            if len(data[:, i, j]) == 1 else np.argmax(data[:, i, j])), 20)] \
                            for i, j in itertools.product(range(out_h), range(out_w))]).reshape(out_h, out_w, 3)

    # convert to gray scale
    mask_img = Image.fromarray(CLASSES_MAP.astype(np.uint8)).convert('L')   #白黒写真に変更(2値化)
    
    return mask_img.resize((w, h))