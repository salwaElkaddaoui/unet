import tensorflow as tf
from abc import abstractmethod
from unet_block import UnetBlock
from residualmixin import ResidualMixin

class UnetBottleneck(UnetBlock):
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)        
            
    @abstractmethod
    def __call__(self, x):
        pass

class BasicBottleneck(UnetBottleneck):
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)

    def __call__(self, input, is_training):
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_conv(conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        return conv

        
class ResidualBottleneck(UnetBottleneck, ResidualMixin):
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)
        self.initialize_skip_connection()

    def __call__(self, input, is_training):
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = tf.nn.conv2d(input=conv, filters=self.kernel1, strides=[1, 1, 1, 1], padding=self.padding)
        conv = tf.nn.bias_add(conv, self.bias1)
        conv = self.apply_skip_connection(input, conv)
        if self.use_batchnorm:
            conv = self.batch_norm1(conv, training=is_training)
        conv = tf.nn.relu(conv)
        return conv

