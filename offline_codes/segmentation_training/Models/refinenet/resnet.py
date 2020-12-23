"""
Implementation of ResNet with Keras for learning purposes. Influenced by:
- https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
- Keras implementation of ResNet50
Get weights from https://github.com/fchollet/deep-learning-models/releases/
"""
from keras import layers
from keras.models import Model

def residual_identity_block(input_tensor,
                            kernel_size,
                            filters,
                            dilation_rate,
                            name,
                            num):
    """
    Bottleneck v1. Input tensor and processed tensor  are added directly at the shotcut

    # Arguments:
        input_tensor: input tensor
        kernel_size: Size of the conv Kernel of the inner block
        filters: List of 3 integers of amount of filters  each conv block should create
        name: Name of the ResNet block
        num: Number of this (inner) block

    # Returns:
        Output tensor
    """
    base_name = name + "_" + str(num)

    x = layers.Conv2D(filters[0],
                      kernel_size=(1, 1),
                      kernel_initializer='he_normal',
                      dilation_rate=dilation_rate,
                      name=base_name + '_conv_a')(input_tensor)
    x = layers.BatchNormalization(axis=3,
                                  name=base_name + '_batch_a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[1],
                      kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      dilation_rate=dilation_rate,
                      name=base_name + '_conv_b')(x)
    x = layers.BatchNormalization(axis=3,
                                  name=base_name + '_batch_b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[2],
                      kernel_size=(1, 1),
                      kernel_initializer='he_normal',
                      dilation_rate=dilation_rate,
                      name=base_name + '_conv_c')(x)
    x = layers.BatchNormalization(axis=3,
                                  name=base_name + '_batch_c')(x)

    x = layers.add([x, input_tensor],
                    name=base_name + '_add')
    x = layers.Activation('relu')(x)
    return x

def residual_conv_block(input_tensor,
                        kernel_size,
                        filters,
                        dilation_rate,
                        name,
                        num,
                        strides=(2, 2)):
    """
    Bottleneck v1. Reduces output dimensionality. Input tensor and processed tensor
    dimensionality do not match, gets adjusted by conv block at the shortcut.
    # Arguments:
        input_tensor: input tensor,
        kernel_size: Size of the conv Kernel of the inner block and shortcut
        filters: List of 3 integers of amount of filters  each conv block should create
        strides: Stride tuple of the inner conv block
        name: Name of the ResNet block
        num: Number of this (inner) block
    # Returns:
        Output tensor
    """
    base_name = name + "_" + str(num)

    if dilation_rate != 1:
        strides = (1, 1)

    x = layers.Conv2D(filters[0],
                      kernel_size=(1, 1),
                      strides=strides,
                      kernel_initializer='he_normal',
                      dilation_rate=dilation_rate,
                      name=base_name + '_conv_a')(input_tensor)
    x = layers.BatchNormalization(axis=3,
                                  name=base_name + '_batch_a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[1],
                      kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      dilation_rate=dilation_rate,
                      name=base_name + '_conv_b')(x)
    x = layers.BatchNormalization(axis=3,
                                  name=base_name + '_batch_b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[2],
                      kernel_size=(1, 1),
                      kernel_initializer='he_normal',
                      dilation_rate=dilation_rate,
                      name=base_name + '_conv_c')(x)
    x = layers.BatchNormalization(axis=3,
                                  name=base_name + '_batch_c')(x)

    shortcut = layers.Conv2D(filters[2],
                             kernel_size=(1, 1),
                             strides=strides,
                             kernel_initializer='he_normal',
                             dilation_rate=dilation_rate,
                             name=base_name + '_conv_shortcut')(input_tensor)
    shortcut = layers.BatchNormalization(axis=3,
                                         name=base_name + '_batch_shortcut')(shortcut)

    x = layers.add([x, shortcut],
                    name=base_name+'_add')
    x = layers.Activation('relu')(x)
    return x

def residual_block(input_tensor,
                   block_size,
                   filters,
                   dilations_rate,
                   name,
                   strides=(2,2)):
    """
    Creates a ResNet Block. First inner block performs dimensionality reduction.
    # Arguments:
        input_tensor: input tensor
        block_size: amount of blocks that should be created
        filters: list of 3 integers of amount of filters  each conv block should create
        name: TODO
        strides: strides of the second and shortcut conv block of the first layer
    """
    x = residual_conv_block(input_tensor, 3, filters, dilations_rate, name, "0", strides)

    for i in range(1, block_size):
        x = residual_identity_block(x, 3, filters, dilations_rate, name, str(i))

    return x

def ResNet(input_dim=(224, 224, 3),
           blocks_per_block=[3, 4, 6, 3],
           dilation_rates=[1,1,1,1],
           n_classes=1000,
           fcn=False,
           weights=None):
    """
    Creates a ResNet Model. Default model ist ResNet50. That can be changed by adapting
    the blocks_per_block param.
    # Arguments:
        input_tensor: Input tensor
        blocks_per_block: List of numbers of inner bottleneck blocks, see difference between
            ResNet50, ResNet101, ResNet152
        dilation_rates: applies atrous rates to the corresponding block. ATTENTION: if atrous rate !=1
            no downscaling will be applied. OS16 -> [1,1,1,2], OS8 -> [1,1,2,4], OS4 -> [1,2,4,8]
        classes: amount of classes predicted by the network
        fcn: Fully Convolutional Network, if true adds a dense layers at the end
        weights: Path to ResNet weights.
    # Returns:
        Keras model, ResNet blocks and input layer.
    """
    block_filters = [[64, 64, 256],
                     [128, 128, 512],
                     [256, 256, 1024],
                     [512, 512, 2048]]

    def _create():
        """
        Creates a ResNet model
        # Returns:
            Returns the model, the endpoint to each ResNet Block and the input layer as a tupel
            (Keras model, list<keras.layer>, keras.layer.Input)
        """
        resnet_blocks = [None, None, None, None, None]

        # input layer
        img_input = layers.Input(shape=input_dim)

        ##################### ResNet #####################

        # conv1
        resnet_blocks[0] = layers.ZeroPadding2D(padding=(3, 3))(img_input)
        resnet_blocks[0] = layers.Conv2D(64,
                                         kernel_size=(7, 7),
                                         strides=(2, 2),
                                         padding='valid',
                                         name="res_b_0_conv")(resnet_blocks[0])
        resnet_blocks[0] = layers.BatchNormalization(name="res_b_0_batch")(resnet_blocks[0])
        resnet_blocks[0] = layers.Activation('relu')(resnet_blocks[0])

        # conv2 - Block 1
        resnet_blocks[1] = layers.ZeroPadding2D(padding=(1, 1))(resnet_blocks[0])
        resnet_blocks[1] = layers.MaxPool2D(pool_size=(3, 3),
                                            strides=(2, 2))(resnet_blocks[1])
        resnet_blocks[1] = residual_block(resnet_blocks[1],
                                          blocks_per_block[0],
                                          block_filters[0],
                                          dilation_rates[0],
                                          strides=(1,1),
                                          name="res_b_1")

        # conv3 - Block 2
        resnet_blocks[2] = residual_block(resnet_blocks[1],
                                          blocks_per_block[1],
                                          block_filters[1],
                                          dilation_rates[1],
                                          name="res_b_2")

        # conv4 - Block 3
        resnet_blocks[3] = residual_block(resnet_blocks[2],
                                          blocks_per_block[2],
                                          block_filters[2],
                                          dilation_rates[2],
                                          name="res_b_3")

        #conv5  - Block 4
        resnet_blocks[4] = residual_block(resnet_blocks[3],
                                          blocks_per_block[3],
                                          block_filters[3],
                                          dilation_rates[3],
                                          name="res_b_4")

        # For FCNs the prediction layer is dropped
        if not fcn:
            resnet_blocks[4] = layers.GlobalAveragePooling2D()(resnet_blocks[4])
            resnet_blocks[4] = layers.Dense(n_classes, activation="sigmoid")(resnet_blocks[4])

        model = Model(img_input, resnet_blocks[4], name="ResNet")

        # load weights
        if weights is not None:
            model.load_weights(weights, by_name=True)
            print("ResNet weights loaded")

        return (model, resnet_blocks, img_input)

    return _create()

def ResNet50(*args, **kwargs):
    """ Creates a ResNet50 model.
    For param details have a look at ResNet()
    # Returns:
        Keras model
    """
    return ResNet(*args, **kwargs)

def ResNet101(*args, **kwargs):
    """ Creates a ResNet101 model.
    For param details have a look at ResNet()
    # Returns:
        Keras model
    """
    kwargs['blocks_per_block'] = [3, 4, 23, 3]
    return ResNet(*args, **kwargs)

def ResNet152(*args, **kwargs):
    """ Creates a ResNet152 model.
    For param details have a look at ResNet()
    # Returns:
        Keras model
    """
    kwargs['blocks_per_block'] = [3, 8, 36, 3]
    return ResNet(*args, **kwargs)

if __name__ == '__main__':
    model = ResNet50(input_dim=(120, 213,1), dilation_rates=[1,1,1,1], fcn=True, n_classes=1)
    print(model[0].summary())