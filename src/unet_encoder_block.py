from abc import abstractmethod
import tensorflow as tf
from unet_block import UnetBlock

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
    """
    A basic encoder block for the U-Net architecture, as described in the original U-Net paper.

    conv2d -> batchnorm -> relu -> conv2d -> batchnorm -> relu ->  pool2d
    """
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True, use_dropout=False):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm, use_dropout)
        

    def __call__(self, input, is_training):
        
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_conv(conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        conv = self.apply_dropout(conv, is_training)
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        return conv, pool

        
class ResidualEncoderBlock(UnetEncoderBlock):
    """
    This class implements an encoder block with a residual connection, inspired by 
    ResNet-style architectures as described in the paper "Deep Residual Learning for 
    Image Recognition" by He et al. 
    
    The block consists of two convolutional layers with optional batch normalization and ReLU activation. 
    The skip connection adds the input of the block to the output of the first convolution before passing 
    the result into the second convolution operation. This helps mitigate the vanishing
    gradient problem and allows the network to learn identity mappings more effectively.
    """
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True, use_dropout=False):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm, use_dropout)

        self.skip_connection_kernel = tf.Variable(self.initializer([1, 1, self.nb_in_channels, self.nb_out_channels])) # 1x1 conv2d 
        self.skip_connection_bias = tf.Variable(tf.zeros(shape=[self.nb_out_channels]))

    def __call__(self, input, is_training):
        
        conv = self.apply_conv(input, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = tf.nn.conv2d(input=conv, filters=self.kernel1, strides=[1, 1, 1, 1], padding=self.padding)
        conv = tf.nn.bias_add(conv, self.bias1)
        
        #the skip connection:
        # making sure that the input and the output of the previous conv operation 
        # have the same height and width before adding them up
        input_depth_changed = tf.nn.conv2d(input=input, filters=self.skip_connection_kernel, strides=[1, 1, 1, 1], padding=self.padding)
        input_depth_changed = tf.nn.bias_add(input_depth_changed, self.skip_connection_bias)
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

        if (self.use_dropout and is_training):
            conv = tf.nn.dropout(x=conv, rate=0.5)

        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        return conv, pool

