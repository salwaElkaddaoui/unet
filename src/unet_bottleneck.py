import tensorflow as tf
from abc import abstractmethod
from unet_block import UnetBlock
from residualmixin import ResidualMixin

class UnetBottleneck(UnetBlock):
    @abstractmethod
    def __call__(self, x):
        pass

class BasicBottleneck(UnetBottleneck):
    def __call__(self, input, is_training):
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_conv(conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        return conv

        
class ResidualBottleneck(UnetBottleneck, ResidualMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_skip_connection()

    def __call__(self, input, is_training):
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_residual_conv(input, conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        return conv

