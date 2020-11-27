import cv2
import numpy as np
import argparse
import os
from PIL import Image

from dataset_loader import default_in_shape
from model.u2net import U2NET
import tensorflow as tf
from tensorflow import keras
from dataset_loader import *
import pathlib

#########################################################
# Optimizer / Loss
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model.checkpoint', save_weights_only=True, verbose=1)
loss_bce = tf.keras.losses.BinaryCrossentropy()


def loss_function(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)
    loss0 = loss_bce(y_true, y_pred[0])
    loss1 = loss_bce(y_true, y_pred[1])
    loss2 = loss_bce(y_true, y_pred[2])
    loss3 = loss_bce(y_true, y_pred[3])
    loss4 = loss_bce(y_true, y_pred[4])
    loss5 = loss_bce(y_true, y_pred[5])
    loss6 = loss_bce(y_true, y_pred[6])

    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return total_loss


output_dir = pathlib.Path('out')
#########################################################

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
    image_dir = pahtlib.Path(args.image_dir)
    if not image_dir.exists():
        print("Input image directory does not exist: {}".format(image_dir))
    else:
        images = glob.glob(str(image_dir.joinpath("*.png"))) + glob.glob(str(image_dir.joinpath("*.jpg")))
        if len(images) == 0:
            print("No image found in input image directory: {}".format(image_dir))
        else:
            imgs.extend(images)
    
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

def apply_mask(img, mask):
    return np.multiply(img, mask)

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
            i = ori_image.resize(default_in_shape[:2], Image.BICUBIC)
        
        model_input = format_input(i)
        fused_mask = model(model_input, Image.BICUBIC)[0][0]
        output_mask = np.asarray(fused_mask)
        
        if i.size != default_in_shape:
            output_mask = cv2.resize(output_mask, dsize=ori_image.size)
        
        output_mask = np.tile(np.expand_dims(output_mask, axis=2), [1, 1, 3])
        output_img = np.expand_dims(np.array(ori_image)/255., 0)[0]

        if apply_mask:
            output_img = apply_mask(output_img, output_mask)
        else:
            output_img = output_mask

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
