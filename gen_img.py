import cv2
import numpy as np
import argparse
import os
from PIL import Image

from tensorflow import keras
from dataset_loader import *
from model.u2net import *
import pathlib

def str2bool(str):
    return str is not None and str.lower() in ("yes", "true", "t", "y", "1")

# Args
parser = argparse.ArgumentParser(description='U22 NET Testing')
parser.add_argument('--image', default=None, type=str,
                    help='a single image')
parser.add_argument('--image_dir', default=None, type=str,
                    help='a directory of images')
parser.add_argument('--output_dir', default=None, type=str,
                    help='a directory to output to')
parser.add_argument('--weights', default=None, type=str,
                    help='the weights file of a trained network')
parser.add_argument('--merged', default=False, type=str2bool,
                    help='display input image and output mask side-by-side')
parser.add_argument('--apply_mask', default=False, type=str2bool,
                    help='apply the mask to the input image and show in place of the mask')
parser.add_argument('--include_original', default=False, type=str2bool,
                    help='include the original input image in the final output along with merged')
args = parser.parse_args()

imgs = []

if args.image:
    if not os.path.exists(args.image):
        print("Input image file does not exist: {}".format(args.image))
    else:
        imgs.append(args.image)

if args.image_dir:
    image_dir = pathlib.Path(args.image_dir)
    if not image_dir.exists():
        print("Input image directory does not exist: {}".format(image_dir))
    else:
        images = glob.glob(str(image_dir.joinpath("*.png"))) + glob.glob(str(image_dir.joinpath("*.jpg")))
        if len(images) == 0:
            print("No image found in input image directory: {}".format(image_dir))
        else:
            imgs.extend(images)
    
output_dir = pathlib.Path('out')
if args.output_dir:
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

if args.weights:
    if not os.path.exists(args.weights):
        print("Model weights file does not exist: {}".format(args.weights))
        weights = ""
    else:
        weights = args.weights

merged = None
if args.merged:
    merged = args.merged

apply_mask = None
if args.apply_mask:
    apply_mask = args.apply_mask

include_original = None
if args.include_original:
    include_original = args.include_original

def format_input(image):
    if np.array(image).shape[-1] == 4:
        image = image.convert('RGB')
    return np.expand_dims(np.array(image)/255., 0)

def main():    
    if len(imgs) == 0 or weights == "":
        return
    
    inputs = keras.Input(shape=default_in_shape)
    network = U2NET()
    outputs = network(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name='u2netmodel')
    model.compile(optimizer=optimizer, loss=loss_function, metrics=None)

    model.load_weights(weights)
    
    # evaluate each image
    for img in imgs:
        ori_image = Image.open(img).convert('RGB')
        i = ori_image
        if ori_image.size != default_in_shape:
            i = cv2.resize(np.array(ori_image), default_in_shape[:2]) # default bilinear interpolation
        
        model_input = format_input(i)
        fused_mask = model(model_input, Image.BILINEAR)[0][0]
        output_mask = np.array(fused_mask)
        
        if i.size != default_in_shape:
            output_mask = cv2.resize(output_mask, ori_image.size) # default bilinear interpolation
        
        output_mask = np.tile(np.expand_dims(output_mask, axis=2), [1, 1, 3])
        output_img = np.expand_dims(np.array(ori_image)/255., 0)[0]

        output_img = np.multiply(output_img, output_mask) if apply_mask else output_mask

        if merged and not include_original:
            output_img = np.concatenate((output_mask, output_img), axis=1)

        if merged and include_original:
            original_image = np.expand_dims(np.array(ori_image)/255., 0)[0]
            output_img = np.concatenate((original_image, output_mask, output_img), axis=1)

        output_img = cv2.cvtColor(output_img.astype('float32'), cv2.COLOR_BGR2RGB) * 255.
        out_dir = output_dir.joinpath(pathlib.Path(img).name)
        cv2.imwrite(str(out_dir), output_img)
        
if __name__=='__main__':
    main()
