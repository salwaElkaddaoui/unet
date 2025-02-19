import tensorflow as tf
from unet_block import UnetBlock
from abc import abstractmethod

class UnetBottleneck(UnetBlock):
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)        
            
    @abstractmethod
    def __call__(self, x):
        pass

class BasicBottleneck(UnetBottleneck):
    """
    A basic bottleneck for the U-Net architecture, as described in the original U-Net paper.

    conv2d -> batchnorm -> relu -> conv2d -> batchnorm -> relu
    """
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)

    def __call__(self, input, is_training):
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_conv(conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        return conv

        
class ResidualBottleneck(UnetBottleneck):
    """
    This class implements a bottlebneck with a residual connection, inspired by 
    ResNet-style architectures as described in the paper "Deep Residual Learning for 
    Image Recognition" by He et al. 
    """
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)

        self.skip_connection_kernel = tf.Variable(self.initializer([1, 1, self.nb_in_channels, self.nb_out_channels])) # 1x1 conv2d 
        self.skip_connection_bias = tf.Variable(tf.zeros(shape=[self.nb_out_channels]))


    def __call__(self, input, is_training):

        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)

        conv = tf.nn.conv2d(input=conv, filters=self.kernel1, strides=[1, 1, 1, 1], padding=self.padding)
        conv = tf.nn.bias_add(conv, self.bias1)
        
        #the skip connection:
        
        input_depth_changed = tf.nn.conv2d(input=input, filters=self.skip_connection_kernel, strides=[1, 1, 1, 1], padding=self.padding)
        input_depth_changed = tf.nn.bias_add(input_depth_changed, self.skip_connection_bias)
        # making sure that the input and the output of the previous conv operation 
        # have the same height and width before adding them up
        shape_diff = tf.shape(input_depth_changed) - tf.shape(conv)
        padding = [
            [0, 0],  # No padding for batch dimension
            [shape_diff[1]//2, (shape_diff[1]+1)//2],  # Pad height (before and after)
            [shape_diff[2]//2, (shape_diff[2]+1)//2],  # Pad width (before and after)
            [0, 0]  # No padding for channels
        ]
        conv = tf.pad(conv, paddings=padding)
        conv = tf.add(input_depth_changed, conv)
        if self.use_batchnorm:
            conv = self.batch_norm1(conv, training=is_training)
        conv = tf.nn.relu(conv)

        return conv

