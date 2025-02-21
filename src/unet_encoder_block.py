import tensorflow as tf
from abc import abstractmethod
from unet_block import UnetBlock
from residualmixin import ResidualMixin

class UnetEncoderBlock(UnetBlock):
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True, use_dropout=False):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)
        self.use_dropout = use_dropout

    def apply_dropout(self, conv, is_training):
        if self.use_dropout and is_training:
            return tf.nn.dropout(conv, rate=0.5)
        return conv

    @abstractmethod
    def __call__(self, x):
        pass

class BasicEncoderBlock(UnetEncoderBlock):
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True, use_dropout=False):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm, use_dropout)
        
    def __call__(self, input, is_training):
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_conv(conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        conv = self.apply_dropout(conv, is_training)
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        return conv, pool

        
class ResidualEncoderBlock(UnetEncoderBlock, ResidualMixin):
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True, use_dropout=False):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm, use_dropout)
        self.initialize_skip_connection()

    def __call__(self, input, is_training):
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = tf.nn.conv2d(input=conv, filters=self.kernel1, strides=[1, 1, 1, 1], padding=self.padding)
        conv = tf.nn.bias_add(conv, self.bias1)
        conv = self.apply_skip_connection(input, conv)
        if self.use_batchnorm:
            conv = self.batch_norm1(conv, training=is_training)
        conv = tf.nn.relu(conv)
        if (self.use_dropout and is_training):
            conv = tf.nn.dropout(x=conv, rate=0.5)
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        return conv, pool

