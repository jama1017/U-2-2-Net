import argparse
import pathlib
import signal

from tensorflow import keras
from dataset_loader import *
from model.u2net import *
from datetime import datetime

# Model
resume = None
weight_dir = pathlib.Path('weights').absolute()
weights_file = weight_dir.joinpath('u2net.h5')
default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)

#Arguments
parser = argparse.ArgumentParser(description='U22 Net')
parser.add_argument('--resume', default=None, type=str)
args = parser.parse_args()
if args.resume:
    resume = args.resume

# Training
batch_size = 9
epochs = 40000
learning_rate = 0.001
save_interval = 100
log_file_path = './my_log.txt'

def train():
    inputs = keras.Input(shape=default_in_shape)
    u2net = U2NET()
    outputs = u2net(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name='u2netmodel')
    model.compile(optimizer=optimizer, loss=loss_function, metrics=None)
    model.summary()

    with open(log_file_path, "a") as f:
        f.write('\nTraining begins at: %s\n' % datetime.now())

    if resume:
        print('Loading weights from %s' % resume)
        model.load_weights(resume)

    # helper function to save state of model
    def save_weights():
        print('Saving state of model to %s' % weights_file)
        with open(log_file_path, "a") as f:
            f.write('\nSaving state of model to %s' % weights_file)
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
            inputs, masks = get_training_img_gt_batch(batch_size=batch_size)
            loss = model.train_on_batch(inputs, masks)
        except KeyboardInterrupt:
            save_weights()
            return
        except ValueError:
            continue

        print('\nTraining epoch {} with loss {}'.format(e, loss))
        with open(log_file_path, "a") as f:
            f.write('\nTraining epoch {} with loss {}'.format(e, loss))

        if e % 10 == 0:
            print('[%05d] Loss: %.4f' % (e, loss))
            with open(log_file_path, "a") as f:
                f.write('\n[%05d] Loss: %.4f' % (e, loss))

        if save_interval and e > 0 and e % save_interval == 0:
            save_weights()
        
        with open(log_file_path, "a") as f:
            f.write('\nCurrent time: %s\n' % datetime.now())

if __name__ == "__main__":
    train()
