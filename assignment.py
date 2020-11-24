import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pathlib
import signal

from dataset_loader import *
from tensorflow import keras
from tensorflow.keras.layers import Input
from model.u2net import *

#Arguments
parser = argparse.ArgumentParser(description='U22 Net')
parser.add_argument('--resume', default=None, type=str)
args = parser.parse_args()
if args.resume:
    resume = args.resume

# Model
resume = None
weight_dir = pathlib.Path('weights').absolute()
weights_file = weight_dir.joinpath('u2net.h5')
default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)

# Training
batch_size = 12
epochs = 10000
learning_rate = 0.001
save_interval = 100

# Optimizer / Loss
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08) 
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model.checkpoint', save_weights_only=True, verbose=1)
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


def train():
    inputs = keras.Input(shape=default_in_shape)
    u2net = U2NET()
    out = u2net(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
    model.compile(optimizer=optimizer, loss=loss_function, metrics=None)
    model.summary()

    if resume:
        print('Loading weights from %s' % resume)
        model.load_weights(resume)

    # helper function to save state of model
    def save_weights():
        print('Saving state of model to %s' % weights_file)
        model.save_weights(str(weights_file))

    # signal handler for early abortion to autosave model state
    def autosave(sig, frame):
        print('Training aborted. Saving weights.')
        save_weights()
        exit(0)

    for sig in [signal.SIGABRT, signal.SIGINT, signal.SIGTSTP]:
        signal.signal(sig, autosave)

    # start training
    print('--- Start Training ---')
    for e in range(epochs):
        try:
            feed, out = get_training_img_gt_batch(batch_size=batch_size)
            loss = model.train_on_batch(feed, out)
        except KeyboardInterrupt:
            save_weights()
            return
        except ValueError:
            continue

        print('Training epoch {}'.format(e))

        if e % 10 == 0:
            print('[%05d] Loss: %.4f' % (e, tf.reduce_sum(loss)))

        if save_interval and e > 0 and e % save_interval == 0:
            save_weights()


if __name__ == "__main__":
    train()
