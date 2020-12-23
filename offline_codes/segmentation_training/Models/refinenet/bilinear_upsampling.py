import tensorflow as tf
from keras.layers import Lambda
from keras.layers import Layer


class BilinearUpsampling(Layer):
    def __init__(self, factor_shape, **kwargs):
        """ Creates an Bilnear Upsamling Layer.
        # Arguments:
            factor_shape: either a scaling factor (int) or list containing
                the new output heigth and width

        # Returns:
            The upsampled tensor.
        """

        self.factor_shape = factor_shape

        super(BilinearUpsampling, self).__init__(**kwargs)

    def build(self, input_shape):
        try:  # heigth, width provided
            self.new_heigth = self.factor_shape[0]
            self.new_width = self.factor_shape[1]

        except Exception:  # scaling factor provided
            self.new_heigth = input_shape[1] * self.factor_shape
            self.new_width = input_shape[2] * self.factor_shape

        self.new_heigth = int(self.new_heigth)
        self.new_width = int(self.new_width)
        super(BilinearUpsampling, self).build(input_shape)

    def call(self, x):
        return tf.image.resize_bilinear(x, (self.new_heigth, self.new_width))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                self.new_heigth,
                self.new_width,
                input_shape[3])

    def get_config(self):
        config = super(BilinearUpsampling, self).get_config()
        config['factor_shape'] = self.factor_shape

        return config