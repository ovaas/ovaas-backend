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