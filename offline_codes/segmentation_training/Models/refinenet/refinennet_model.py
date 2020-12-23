from keras import layers
from keras.models import Model
from Models.refinenet.resnet import ResNet50, ResNet101, ResNet152
from Models.refinenet.bilinear_upsampling import BilinearUpsampling


def residual_conv_unit(input_tensor,
                       kernel_size=(1, 1),
                       filters=256,
                       name=None,
                       num=None):
    """ RCU light weight style.
    Adaptive convolution set for fine-tuning pretrained ResNet weights.
    Simplified version of ResNet block, where batch norm is removed.
    RefineNet-4-Block filters: 512; remaining ones: 256.
    # Arguments:
        input_tensor: input tensor
        kernel_size: Size of convolutional kernel of both conv layers
        filters: Amount of filter maps produced
        name: Name of RCU
        num: Number of RCU block
    # Returns:
        Output tensor
    """
    base_name = name + "_rcu_" + str(num)

    # No change in dimension (1x1) strides
    x = layers.ReLU()(input_tensor)
    x = layers.Conv2D(filters,
                      kernel_size,
                      padding = "same",
                      name = base_name + "_conv_a")(x)

    x = layers.ReLU()(input_tensor)
    x = layers.Conv2D(filters,
                      kernel_size = (3, 3),
                      padding = "same",
                      name = base_name + "_conv_a")(x)

    x = layers.ReLU()(x)
    x = layers.Conv2D(filters,
                      kernel_size,
                      padding = "same",
                      name = base_name + "_conv_b")(x)

    return layers.add([x, input_tensor])


def multi_resolution_fusion(high_input_tensor=None,
                            low_input_tensor=None,
                            filters=256,
                            name=None):
    """ MRF light weight style.
    Fuses togetherlower and higher resolution input feature maps.
    Lower inputs coming from the previous RefineNet-Block and higher Inputs
    from ResNet.
    # Arguments:
        high_input_tensor: Input tensor with higher resolution (ResNet)
        low_input_tensor: Input tensor with lower resoltion (RefineNet)
        filters: Amount of filters map produces by each convolution
        name: Name of MRF

    # Returns:
        Output tensor
    """
    base_name = name + "_mrf"

    # if only one input path exists the no changes, i.e. RefineBlock 4
    if low_input_tensor is None:
        return high_input_tensor
    else:
        # Convolutions for input adaptaion
        low = layers.Conv2D(filters,
                            (1, 1),
                            padding = "same",
                            name = base_name + "_conv_a")(low_input_tensor)
        high = layers.Conv2D(filters,
                             (1, 1),
                             padding = "same",
                             name = base_name + "_conv_b")(high_input_tensor)

        # don't upscale by two but to original size on that block level
        # low_upsampled = bilinear_up_sampling(2)(low)
        low_upsampled = BilinearUpsampling([high.shape[1].value, high.shape[2].value])(low)

        return layers.add([low_upsampled, high])


def chained_residual_pooling(input_tensor,
                             filters=256,
                             name=None):
    """ CRP light weight style.
    Aims to capture background context from a large image region.
    # Arguments:
        input_tensor: Input tensor
        filters: Amount of filters produced by each convoltion
        name: Name of CRP
    # Returns:
        Output tensor
    """
    base_name = name + "_crp"

    lower_path = layers.ReLU()(input_tensor)

    upper_path = layers.MaxPooling2D(pool_size = (5, 5),
                                     strides = (1, 1),
                                     padding = "same")(lower_path)
    upper_path = layers.Conv2D(filters,
                               kernel_size = (1, 1),
                               padding = "same",
                               name = base_name + "_conv_a")(upper_path)

    sum_1 = layers.add([upper_path, lower_path])

    upper_path = layers.MaxPooling2D(pool_size = (5, 5),
                                     strides = (1, 1),
                                     padding = "same")(upper_path)
    upper_path = layers.Conv2D(filters,
                               kernel_size = (1, 1),
                               padding = "same",
                               name = base_name + "_conv_b")(upper_path)

    sum_2 = layers.add([upper_path, sum_1])

    return sum_2


def refine_block(high_input_tensor=None,
                 low_input_tensor=None,
                 name=None):
    """ Refine block light weight style.
    Each LW RefineNet Block combines a Multi-resolution fusion
    with Chained Residual Pooling.
    # Arguments:
        high_input_tensor: Input tensor with higher resoltion (ResNet)
        low_input_tensor: Input with lower resolution (RefineNet previous)
    # Returns:
        Output tensor for Refine block
    """
    # "RefineNet-4" filter number: 512, others: 256
    if low_input_tensor is None:
        filters = 512
    else:
        filters = 256

    # No fusion happening when no low input path is given
    x = multi_resolution_fusion(high_input_tensor, low_input_tensor, name = name)
    x = chained_residual_pooling(x, filters, name = name)

    return x


def LightWeightRefineNet(input_dim=(240,320, 3),
                         n_classes=2,
                         resnet="resnet_50",
                         resnet_weights=None,
                         refinenet_weights=None):
    """Creates a Light-Weight RefineNet model.
    Attention: Upscaling is not performed by a factor but by the correponding
    resnet block on every step, so all input dimensions can be used.
    # Arguments:
        input_dim: Input dimension.
        n_classes: Amount of segementation classes predicted by the network.
        resnet: ResNet backbone model either "resnet_50", "resnet_101", "resnet_152".
        resnet_weights: Path to ResNet weights.
        refinenet_weights: Path to RefineNet weights.
    # Returns:
        Keras Light-Weight RefineNet model
    """
    if resnet_weights is not None and refinenet_weights is not None:
        raise ValueError('Can not load resnet and refinet weights at the same time.')

    # get ResNet Blocks
    resnet_blocks = [None, None, None, None, None]
    if resnet == "resnet_50":
        resnet_blocks, img_input = ResNet50(input_dim = input_dim,
                                            fcn = True,
                                            weights = resnet_weights)[1:]
    elif resnet == "resnet_101":
        resnet_blocks, img_input = ResNet101(input_dim = input_dim,
                                             fcn = True,
                                             weights = resnet_weights)[1:]
    elif resnet == "resnet_152":
        resnet_blocks, img_input = ResNet152(input_dim = input_dim,
                                             fcn = True,
                                             weights = resnet_weights)[1:]

    def _create():
        """
        Creates a RefineNet model
        # Returns:
            Keras model
        """
        refinenet_blocks = [None, None, None, None, None]

        ##################### RefineNet #####################

        # Adjust feature map dimension. Could not find in paper, but is necessary.
        # Same in Semantic Suite.
        resnet_blocks[1] = layers.Conv2D(256,
                                         kernel_size = (1, 1),
                                         name = "res_to_ref_1")(resnet_blocks[1])
        resnet_blocks[2] = layers.Conv2D(256,
                                         kernel_size = (1, 1),
                                         name = "res_to_ref_2")(resnet_blocks[2])
        resnet_blocks[3] = layers.Conv2D(256,
                                         kernel_size = (1, 1),
                                         name = "res_to_ref_3")(resnet_blocks[2])
        resnet_blocks[4] = layers.Conv2D(512,
                                         kernel_size = (1, 1),
                                         name = "res_to_ref_4")(resnet_blocks[3])

        # build RefineNet blocks
        refinenet_blocks[4] = refine_block(resnet_blocks[4],
                                           None,
                                           name = "ref_b_4")
        refinenet_blocks[3] = refine_block(resnet_blocks[3],
                                           refinenet_blocks[4],
                                           name = "ref_b_3")
        refinenet_blocks[2] = refine_block(resnet_blocks[2],
                                           refinenet_blocks[3],
                                           name = "ref_b_2")
        refinenet_blocks[1] = refine_block(resnet_blocks[1],
                                           refinenet_blocks[2],
                                           name = "ref_b_1")

        # Upscaling not by a factor but to original size of image input
        net = BilinearUpsampling([img_input.shape[1].value, img_input.shape[2].value])(refinenet_blocks[1])

        # We use 1x1 convolutions, instead of 3x3 as mentioned in the paper
        # Output layer: Number of feature maps corresponds to number of n_classes
        if n_classes == 1:
            net = layers.Conv2D(n_classes,
                                (1, 1),
                                name = "output",
                                activation = 'sigmoid')(net)
        else:
            net = layers.Conv2D(n_classes,
                                (1, 1),
                                name = "output",
                                activation = 'sigmoid')(net)

        model = Model(img_input, net, name = "lightweight_refinenet")

        # load weights
        if resnet_weights is not None:
            model.load_weights(resnet_weights)
            print('ResNet weights loaded.')
        if refinenet_weights is not None:
            model.load_weights(refinenet_weights)
            print('RefineNet weights loaded.')

        return model, net, img_input

    return _create()


# In[7]:


def LightWeightRefineNet50(*args, **kwargs):
    """ Creates a LightWeightRefineNet model based on ResNet50.
    For param details have a look at ResNet()
    # Returns:
        Keras model
    """
    return LightWeightRefineNet(*args, **kwargs)


def LightWeightRefineNet101(*args, **kwargs):
    """ Creates a ResNet101 model.
    For param details have a look at ResNet()
    # Returns:
        Keras model
    """
    kwargs['resnet'] = 'resnet_101'
    return LightWeightRefineNet(*args, **kwargs)


def LightWeightRefineNet152(*args, **kwargs):
    """ Creates a ResNet152 model.
    For param details have a look at ResNet()
    # Returns:
        Keras model
    """
    kwargs['resnet'] = 'resnet_152'
    return LightWeightRefineNet(*args, **kwargs)


class Refinenet:
    def __init__(self, size=(320, 240, 3), num_model=0, n_classes = 6):
        self.num_model = num_model
        self.n_classes = n_classes
        self.train_model=None
        self.chose_model(self.num_model)

    def chose_model(self, num_model):
        if num_model == 0:
            self.train_model = LightWeightRefineNet50(n_classes= self.n_classes)
        if num_model == 1:
            self.train_model = LightWeightRefineNet101(n_classes= self.n_classes)
        if num_model == 2:
            self.train_model = LightWeightRefineNet152(n_classes= self.n_classes)
        return self.train_model
if __name__== "__main__":
    model=Refinenet()
    train_model=model.chose_model(0)
    print(train_model[0].summary())
