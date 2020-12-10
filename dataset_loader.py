import os
import pathlib
import random
import numpy as np
import wget
import zipfile

from PIL import Image 

cache = None

# Model
default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)

# Dataset 
current_location = pathlib.Path(__file__).absolute().parents[0]
root_data_dir = pathlib.Path('data')
dataset_url = "https://drive.google.com/u/0/uc?export=download&confirm=txWv&id=1BmqQiSYJGchWzNoauBDrXHl5p1Sb2MmE"

# HKU-IS
# dataset_dir = root_data_dir.joinpath('HKU-IS')
# image_dir = dataset_dir.joinpath('imgs')
# mask_dir = dataset_dir.joinpath('gt')

# DUTS-TR
dataset_dir = root_data_dir.joinpath('DUTS-TR')
image_dir = dataset_dir.joinpath('DUTS-TR-Image')
mask_dir = dataset_dir.joinpath('DUTS-TR-Mask')

output_dir = pathlib.Path('out')

def get_image_gt_pair(img_name, img_resize=None, mask_resize=None):
    in_img = image_dir.joinpath(img_name)
    # mask_img = mask_dir.joinpath(img_name)
    mask_img = mask_dir.joinpath(img_name.replace('jpg', 'png')) # needed for DUTS-TR Dataset

    if not in_img.exists() or not mask_img.exists():
        return None

    img  = Image.open(in_img)
    mask = Image.open(mask_img)
    
    # resize the image and mask to 320 * 320
    img = img.resize(img_resize[:2], Image.BILINEAR)
    mask = mask.resize(mask_resize[:2], Image.BILINEAR)
    
    # randomly flip the image horizontally
    if bool(random.getrandbits(1)):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return (np.asarray(img, dtype=np.float32), np.expand_dims(np.asarray(mask, dtype=np.float32), -1))

def get_training_img_gt_batch(batch_size=12, in_shape=default_in_shape, out_shape=default_out_shape):
    global cache
    if cache is None:
        cache = os.listdir(image_dir)
    
    imgs = random.choices(cache, k=batch_size)
    image_list = [get_image_gt_pair(img, img_resize=default_in_shape, mask_resize=default_out_shape) for img in imgs]
    
    imgs_batch  = np.stack([i[0]/255. for i in image_list])
    masks_batch = np.stack([i[1]/255. for i in image_list])
   
    return (imgs_batch, masks_batch)
