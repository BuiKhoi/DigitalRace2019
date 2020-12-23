# coding=utf-8
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from ..layers.pooling import MaxUnpooling2D


def bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    # x = Activation('relu')(x)
    x = PReLU(shared_axes=[1, 2])(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    # x = Activation('relu')(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = MaxUnpooling2D()([other, reverse_module])

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        # decoder = Activation('relu')(decoder)
        decoder = PReLU(shared_axes=[1, 2])(decoder)

    return decoder


def build(encoder, nc):
    network, index_stack = encoder
    enet = bottleneck(network, 64, upsample=True, reverse_module=index_stack.pop())  # bottleneck 4.0
    enet = bottleneck(enet, 64)  # bottleneck 4.1
    enet = bottleneck(enet, 64)  # bottleneck 4.2
    enet = bottleneck(enet, 16, upsample=True, reverse_module=index_stack.pop())  # bottleneck 5.0
    enet = bottleneck(enet, 16)  # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    return enet
