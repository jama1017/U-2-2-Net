import tensorflow as tf 
import numpy as np

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D


#########################################################
# Optimizer / Loss

def loss_function(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)
    loss0 = loss_bce(y_true, y_pred[0])
    loss1 = loss_bce(y_true, y_pred[1])
    loss2 = loss_bce(y_true, y_pred[2])
    loss3 = loss_bce(y_true, y_pred[3])
    loss4 = loss_bce(y_true, y_pred[4])
    loss5 = loss_bce(y_true, y_pred[5])
    loss6 = loss_bce(y_true, y_pred[6])

    # try weighting loss differently
    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return total_loss

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model.checkpoint', save_weights_only=True, verbose=1)
loss_bce = tf.keras.losses.BinaryCrossentropy()

#########################################################


# convolution block
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, dilation_rate):
        super(ConvBlock, self).__init__()
        self.convolution = Conv2D(out_channels, (3,3), padding='SAME', dilation_rate=dilation_rate)
        self.batch_norm = BatchNormalization()
        self.relu = ReLU()
        
    def call(self, inputs):
        output = self.convolution(inputs)
        output = self.batch_norm(output)
        output = self.relu(output)

        return output

# RSU-7
class RSU7(tf.keras.layers.Layer):
    def __init__(self, mid_channels, out_channels):
        super(RSU7, self).__init__()
        self.conv_block = ConvBlock(out_channels, 1)

        self.encoder1 = ConvBlock(mid_channels, 1)
        self.pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder2 = ConvBlock(mid_channels, 1)
        self.pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder3 = ConvBlock(mid_channels, 1)
        self.pool3 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder4 = ConvBlock(mid_channels, 1)
        self.pool4 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder5 = ConvBlock(mid_channels, 1)
        self.pool5 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder6 = ConvBlock(mid_channels, 1)
        # no need to pool here?

        self.encoder7 = ConvBlock(mid_channels, 2)

        self.decoder6 = ConvBlock(mid_channels, 1)
        # try bicubic
        self.upsample6 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder5 = ConvBlock(mid_channels, 1)
        self.upsample5 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder4 = ConvBlock(mid_channels, 1)
        self.upsample4 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder3 = ConvBlock(mid_channels, 1)
        self.upsample3 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder2 = ConvBlock(mid_channels, 1)
        self.upsample2 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder1 = ConvBlock(out_channels, 1)

    def call(self, inputs):
        encoder_out = self.conv_block(inputs)

        encoder_out1 = self.encoder1(encoder_out)
        output = self.pool1(encoder_out1)

        encoder_out2 = self.encoder2(output)
        output = self.pool2(encoder_out2)

        encoder_out3 = self.encoder3(output)
        output = self.pool3(encoder_out3)
        
        encoder_out4 = self.encoder4(output)
        output = self.pool4(encoder_out4)

        encoder_out5 = self.encoder5(output)
        output = self.pool5(encoder_out5)

        encoder_out6 = self.encoder6(output)

        encoder_out7 = self.encoder7(encoder_out6)

        decoder_out6 = self.decoder6(tf.concat([encoder_out7, encoder_out6], axis=3))
        decoder_out6 = self.upsample6(decoder_out6)

        decoder_out5 = self.decoder5(tf.concat([decoder_out6, encoder_out5], axis=3))
        decoder_out5 = self.upsample5(decoder_out5)

        decoder_out4 = self.decoder4(tf.concat([decoder_out5, encoder_out4], axis=3))
        decoder_out4 = self.upsample4(decoder_out4)

        decoder_out3 = self.decoder3(tf.concat([decoder_out4, encoder_out3], axis=3))
        decoder_out3 = self.upsample3(decoder_out3)

        decoder_out2 = self.decoder2(tf.concat([decoder_out3, encoder_out2], axis=3))
        decoder_out2 = self.upsample2(decoder_out2)

        decoder_out = self.decoder1(tf.concat([decoder_out2, encoder_out1], axis=3))

        return decoder_out + encoder_out

# RSU-6
class RSU6(tf.keras.layers.Layer):
    def __init__(self, mid_channels, out_channels):
        super(RSU6, self).__init__()
        self.conv_block = ConvBlock(out_channels, 1)

        self.encoder1 = ConvBlock(mid_channels, 1)
        self.pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder2 = ConvBlock(mid_channels, 1)
        self.pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder3 = ConvBlock(mid_channels, 1)
        self.pool3 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder4 = ConvBlock(mid_channels, 1)
        self.pool4 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder5 = ConvBlock(mid_channels, 1)

        self.encoder6 = ConvBlock(mid_channels, 1)

        self.decoder5 = ConvBlock(mid_channels, 1)
        self.upsample5 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder4 = ConvBlock(mid_channels, 1)
        self.upsample4 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder3 = ConvBlock(mid_channels, 1)
        self.upsample3 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder2 = ConvBlock(mid_channels, 1)
        self.upsample2 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder1 = ConvBlock(out_channels, 1)
    
    def call(self, inputs):
        encoder_out = self.conv_block(inputs)

        encoder_out1 = self.encoder1(encoder_out)
        output = self.pool1(encoder_out1)

        encoder_out2 = self.encoder2(output)
        output = self.pool2(encoder_out2)

        encoder_out3 = self.encoder3(output)
        output = self.pool3(encoder_out3)
        
        encoder_out4 = self.encoder4(output)
        output = self.pool4(encoder_out4)

        encoder_out5 = self.encoder5(output)

        encoder_out6 = self.encoder6(encoder_out5)

        decoder_out5 = self.decoder5(tf.concat([encoder_out6, encoder_out5], axis=3))
        decoder_out5 = self.upsample5(decoder_out5)

        decoder_out4 = self.decoder4(tf.concat([decoder_out5, encoder_out4], axis=3))
        decoder_out4 = self.upsample4(decoder_out4)

        decoder_out3 = self.decoder3(tf.concat([decoder_out4, encoder_out3], axis=3))
        decoder_out3 = self.upsample3(decoder_out3)

        decoder_out2 = self.decoder2(tf.concat([decoder_out3, encoder_out2], axis=3))
        decoder_out2 = self.upsample2(decoder_out2)

        decoder_out = self.decoder1(tf.concat([decoder_out2, encoder_out1], axis=3))

        return decoder_out + encoder_out

# RSU-5
class RSU5(tf.keras.layers.Layer):
    def __init__(self, mid_channels, out_channels):
        super(RSU5, self).__init__()
        self.conv_block = ConvBlock(out_channels, 1)

        self.encoder1 = ConvBlock(mid_channels, 1)
        self.pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder2 = ConvBlock(mid_channels, 1)
        self.pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder3 = ConvBlock(mid_channels, 1)
        self.pool3 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder4 = ConvBlock(mid_channels, 1)

        self.encoder5 = ConvBlock(mid_channels, 1)

        self.decoder4 = ConvBlock(mid_channels, 1)
        self.upsample4 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder3 = ConvBlock(mid_channels, 1)
        self.upsample3 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder2 = ConvBlock(mid_channels, 1)
        self.upsample2 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder1 = ConvBlock(out_channels, 1)
    
    def call(self, inputs):
        encoder_out = self.conv_block(inputs)

        encoder_out1 = self.encoder1(encoder_out)
        output = self.pool1(encoder_out1)

        encoder_out2 = self.encoder2(output)
        output = self.pool2(encoder_out2)

        encoder_out3 = self.encoder3(output)
        output = self.pool3(encoder_out3)
        
        encoder_out4 = self.encoder4(output)

        encoder_out5 = self.encoder5(encoder_out4)

        decoder_out4 = self.decoder4(tf.concat([encoder_out5, encoder_out4], axis=3))
        decoder_out4 = self.upsample4(decoder_out4)

        decoder_out3 = self.decoder3(tf.concat([decoder_out4, encoder_out3], axis=3))
        decoder_out3 = self.upsample3(decoder_out3)

        decoder_out2 = self.decoder2(tf.concat([decoder_out3, encoder_out2], axis=3))
        decoder_out2 = self.upsample2(decoder_out2)

        decoder_out = self.decoder1(tf.concat([decoder_out2, encoder_out1], axis=3))

        return decoder_out + encoder_out


# RSU-4
class RSU4(tf.keras.layers.Layer):
    def __init__(self, mid_channels, out_channels):
        super(RSU4, self).__init__()
        self.conv_block = ConvBlock(out_channels, 1)

        self.encoder1 = ConvBlock(mid_channels, 1)
        self.pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder2 = ConvBlock(mid_channels, 1)
        self.pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.encoder3 = ConvBlock(mid_channels, 1)

        self.encoder4 = ConvBlock(mid_channels, 1)

        self.decoder3 = ConvBlock(mid_channels, 1)
        self.upsample3 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder2 = ConvBlock(mid_channels, 1)
        self.upsample2 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.decoder1 = ConvBlock(out_channels, 1)
    
    def call(self, inputs):
        encoder_out = self.conv_block(inputs)

        encoder_out1 = self.encoder1(encoder_out)
        output = self.pool1(encoder_out1)

        encoder_out2 = self.encoder2(output)
        output = self.pool2(encoder_out2)

        encoder_out3 = self.encoder3(output)
        
        encoder_out4 = self.encoder4(encoder_out3)

        decoder_out3 = self.decoder3(tf.concat([encoder_out4, encoder_out3], axis=3))
        decoder_out3 = self.upsample3(decoder_out3)

        decoder_out2 = self.decoder2(tf.concat([decoder_out3, encoder_out2], axis=3))
        decoder_out2 = self.upsample2(decoder_out2)

        decoder_out = self.decoder1(tf.concat([decoder_out2, encoder_out1], axis=3))

        return decoder_out + encoder_out

# RSU-4F
class RSU4F(tf.keras.layers.Layer):
    def __init__(self, mid_channels, out_channels):
        super(RSU4F, self).__init__()
        self.conv_block = ConvBlock(out_channels, 1)
        self.encoder1 = ConvBlock(mid_channels, 1)
        self.encoder2 = ConvBlock(mid_channels, 2)
        self.encoder3 = ConvBlock(mid_channels, 4)
        self.encoder4 = ConvBlock(mid_channels, 8)
        self.decoder3 = ConvBlock(mid_channels, 4)
        self.decoder2 = ConvBlock(mid_channels, 2)
        self.decoder1 = ConvBlock(out_channels, 1)
    
    def call(self, inputs):
        encoder_out = self.conv_block(inputs)

        encoder_out1 = self.encoder1(encoder_out)
        encoder_out2 = self.encoder2(encoder_out1)
        encoder_out3 = self.encoder3(encoder_out2)
        encoder_out4 = self.encoder4(encoder_out3)

        decoder_out3 = self.decoder3(tf.concat([encoder_out4, encoder_out3], axis=3))
        decoder_out2 = self.decoder2(tf.concat([decoder_out3, encoder_out2], axis=3))
        decoder_out = self.decoder1(tf.concat([decoder_out2, encoder_out1], axis=3))

        return decoder_out + encoder_out

# U^2 net
class U2NET(tf.keras.models.Model):
    def __init__(self, out_channels=1):
        super(U2NET, self).__init__()

        self.en1 = RSU7(32, 64)
        self.pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.en2 = RSU6(32, 128)
        self.pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.en3 = RSU5(64, 256)
        self.pool3 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.en4 = RSU4(128, 512)
        self.pool4 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.en5 = RSU4F(256, 512)
        self.pool5 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')

        self.en6 = RSU4F(256, 512)
        self.upsample6 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.de5 = RSU4F(256, 512)
        self.upsample5 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.de4 = RSU4(128, 256)
        self.upsample4 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.de3 = RSU5(64, 128)
        self.upsample3 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.de2 = RSU6(32, 64)
        self.upsample2 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.de1 = RSU7(16, 64)

        
        self.side6 = Conv2D(out_channels, (3,3), padding='SAME')
        self.side_upsample6 = UpSampling2D(size=(32,32), interpolation='bilinear')
    
        self.side5 = Conv2D(out_channels, (3,3), padding='SAME')
        self.side_upsample5 = UpSampling2D(size=(16,16), interpolation='bilinear')
        
        self.side4 = Conv2D(out_channels, (3,3), padding='SAME')
        self.side_upsample4 = UpSampling2D(size=(8,8), interpolation='bilinear')
        
        self.side3 = Conv2D(out_channels, (3,3), padding='SAME')
        self.side_upsample3 = UpSampling2D(size=(4,4), interpolation='bilinear')
        
        self.side2 = Conv2D(out_channels, (3,3), padding='SAME')
        self.side_upsample2 = UpSampling2D(size=(2,2), interpolation='bilinear')

        self.side1 = Conv2D(out_channels, (3,3), padding='SAME')

        self.out_conv = Conv2D(out_channels, (1,1))

        self.sigmoid = tf.keras.activations.sigmoid
    
    def call(self, inputs):
        en_out1 = self.en1(inputs)
        output = self.pool1(en_out1)

        en_out2 = self.en2(output)
        output = self.pool2(en_out2)

        en_out3 = self.en3(output)
        output = self.pool3(en_out3)

        en_out4 = self.en4(output)
        output = self.pool4(en_out4)

        en_out5 = self.en5(output)
        output = self.pool5(en_out5)

        en_out6 = self.en6(output)
        side6 = self.side_upsample6(self.side6(en_out6))
        en_out6 = self.upsample6(en_out6)

        de_out5 = self.de5(tf.concat([en_out6, en_out5], axis=3))
        side5 = self.side_upsample5(self.side5(de_out5))
        de_out5 = self.upsample5(de_out5)

        de_out4 = self.de4(tf.concat([de_out5, en_out4], axis=3))
        side4 = self.side_upsample4(self.side4(de_out4))
        de_out4 = self.upsample4(de_out4)

        de_out3 = self.de3(tf.concat([de_out4, en_out3], axis=3))
        side3 = self.side_upsample3(self.side3(de_out3))
        de_out3 = self.upsample3(de_out3)

        de_out2 = self.de2(tf.concat([de_out3, en_out2], axis=3))
        side2 = self.side_upsample2(self.side2(de_out2))
        de_out2 = self.upsample2(de_out2)

        de_out1 = self.de1(tf.concat([de_out2, en_out1], axis=3))
        side1 = self.side1(de_out1)

        saliency_fused = self.out_conv(tf.concat([side1, side2, side3, side4, side5, side6], axis=3))

        # try different activations functions (derivative of sigmoid - sigmoid*(1-sigmoid))
        output = tf.stack([self.sigmoid(saliency_fused), self.sigmoid(side1), self.sigmoid(side2), self.sigmoid(side3), self.sigmoid(side4), self.sigmoid(side5), self.sigmoid(side6)])

        return output