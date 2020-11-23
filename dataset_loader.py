import os
import pathlib
import random
import numpy as np
import wget
import zipfile
import glob
import patoolib

from PIL import Image 

cache = None

default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)

# Dataset 
current_location = pathlib.Path(__file__).absolute().parents[0]
root_data_dir = pathlib.Path('data')
dataset_url = "https://drive.google.com/u/0/uc?export=download&confirm=txWv&id=1BmqQiSYJGchWzNoauBDrXHl5p1Sb2MmE"
dataset_dir = root_data_dir.joinpath('HKU-IS')
image_dir = dataset_dir.joinpath('imgs')
mask_dir = dataset_dir.joinpath('gt')

# Evaluation
output_dir = pathlib.Path('out')

def format_input(input_image):
    assert(input_image.size == default_in_shape[:2] or input_image.shape == default_in_shape)
    inp = np.array(input_image)
    if inp.shape[-1] == 4:
        input_image = input_image.convert('RGB')
    return np.expand_dims(np.array(input_image)/255., 0)

def download_and_extract_data():
    f = wget.download(dataset_url, out=str(root_data_dir.absolute()))
    
    with zipfile.ZipFile(f, 'r') as zip_file:
        zip_file.extractall(root_data_dir)

def get_image_gt_pair(img_name, img_resize=None, mask_resize=None, augment=True):
    in_img = image_dir.joinpath(img_name)
    mask_img = mask_dir.joinpath(img_name)

    if not in_img.exists() or not mask_img.exists():
        return None

    img  = Image.open(in_img)
    mask = Image.open(mask_img)

    if img_resize:
        img = img.resize(img_resize[:2], Image.BICUBIC)
    
    if mask_resize:
        mask = mask.resize(mask_resize[:2], Image.BICUBIC)

    # the paper specifies the only augmentation done is horizontal flipping.
    if augment and bool(random.getrandbits(1)):
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
    # print(imgs_batch)
    return (imgs_batch, masks_batch)