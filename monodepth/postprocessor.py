import cv2
import numpy as np
from PIL import Image

# make segmentation mask
def monodepth(output, w: int, h: int) -> Image:

    mask = cv2.resize(output, (w, h))

    # scaling
    mask_min, mask_max = mask.min(), mask.max()

    if mask_max - mask_min > 1e-6: mask=(mask-mask_min)/(mask_max-mask_min)
    else: mask.fill(0.5)

    # convert to gray scale
    mask=Image.fromarray((255-mask*255).astype(np.uint8)).convert('L')
    
    return mask