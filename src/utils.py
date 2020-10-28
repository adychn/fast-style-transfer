import scipy.misc, numpy as np, os, sys
import imageio
from PIL import Image
import matplotlib.pylab as plt

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    imageio.imwrite(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = imageio.imread(style_path, pilmode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
   img = imageio.imread(src, pilmode='RGB') # returns a numpy array
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size:
       img = Image.fromarray(img)
       img = img.resize((img_size[0], img_size[1]))
       img = np.array(img)
   return img # return a numpy array of the image

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path):
    files = []
    # only walk one file? for a content image?
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files
