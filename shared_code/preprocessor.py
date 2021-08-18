from PIL import Image
import base64
import cv2
import io
import numpy as np


def decode(data):
    img_decode = io.BytesIO(data)
    return img_decode


def to_pil_image(img_bin):
    _decoded = io.BytesIO(img_bin)
    return Image.open(_decoded)


def resize(img: Image, w=456, h=256):
    img = img.resize((w, h), Image.ANTIALIAS)
    # TODO scale+padding is better
    return img


def transformCh(img: Image):
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))
    return img


def encode(img: Image):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    byte_data = buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def transpose(data: np.array):
    # hwc > bchw
    new_shape = (2, 0, 1)
    r = data.transpose(new_shape)
    r = np.expand_dims(r, axis=0)
    return r

# for iOS and iPad OS
def rotate_image(image: Image) -> Image:
    ALLOCATION={3: 180,
                6: -90,
                8: 90}        #{orientation value : rotation angle}
    exif=image._getexif()
    # print(f"exif={exif}")
    if exif != None and 274 in exif:
        if exif[274] in ALLOCATION:
            return image.rotate(ALLOCATION[exif[274]], expand=True)
        else: return image
    else: return image

# handwritten
# 1106
def get_characters(char_list_path):
    with open(char_list_path, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)

# This function change the type of picture into the type of Openvinomodel


def preprocess_input(src, height, width):
    ratio = float(src.shape[1]) / float(src.shape[0])
    tw = int(height * ratio)
    rsz = cv2.resize(src, (tw, height),
                     interpolation=cv2.INTER_AREA).astype(np.float32)
    # [h,w] -> [c,h,w]
    img = rsz[None, :, :]
    _, h, w = img.shape
    # right edge padding
    pad_img = np.pad(img, ((0, 0), (0, height - h),
                           (0, width - w)), mode='edge')
    return pad_img
