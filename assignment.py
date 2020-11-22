import tensorflow as tf
import numpy as np

from model.u2net import U2Net

# Optimizer / Loss
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model.checkpoint', save_weights_only=True, verbose=1)
loss_bce = tf.keras.losses.BinaryCrossentropy()

def loss_function(y_true, y_pred):
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
    u2net = U2NET()
    print('--- Start Training ---')
    pass