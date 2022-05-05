import cv2
import numpy as np
from PIL import Image

# make segmentation mask image
def make_human_mask(res: list, w: int, h: int) -> Image:

    if len(res.shape) == 3: res = np.expand_dims(res, axis=1)

    #　make a mask image cropped in the shape of a person.
    data = res[0].transpose((1, 2, 0))
    data = np.where(data == 11, (0, 0, 0), (255, 255, 255))     # 11: person

    # convert to gray scale
    mask = Image.fromarray(data.astype(np.uint8)).convert('L')

    return mask.resize((w, h))

# make segmentation mask
def make_mono_mask(output, w: int, h: int) -> Image:

    mask = cv2.resize(output, (w, h))

    # scaling
    mask_min, mask_max = mask.min(), mask.max()

    if mask_max - mask_min > 1e-6: mask = (mask - mask_min) / (mask_max - mask_min)
    else: mask.fill(0.5)

    # convert to gray scale
    mask = (255 - mask * 255).astype(np.uint8)

    return Image.fromarray(mask)