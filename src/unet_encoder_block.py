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

    def apply_pooling(self, conv):
        return tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)

    @abstractmethod
    def __call__(self, x):
        pass

class BasicEncoderBlock(UnetEncoderBlock):
    def __call__(self, input, is_training):
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_conv(conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        conv = self.apply_dropout(conv, is_training)
        pool = self.apply_pooling(conv)
        return conv, pool

        
class ResidualEncoderBlock(UnetEncoderBlock, ResidualMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.initialize_skip_connection()

    def __call__(self, input, is_training):
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_residual_conv(input, conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        conv = self.apply_dropout(conv, is_training)        
        pool = self.apply_pooling(conv)
        return conv, pool

