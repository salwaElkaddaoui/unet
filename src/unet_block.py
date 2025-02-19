from abc import ABC, abstractmethod
import tensorflow as tf
from batch_normalization import BatchNormalization

class UnetBlock(ABC, tf.Module):
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.nb_in_channels = nb_in_channels
        self.nb_out_channels = nb_out_channels
        self.padding = padding
        self.use_batchnorm = use_batchnorm
        
        if initializer == "he_normal":
            self.initializer = tf.compat.v1.initializers.he_normal()

        self.kernel0 = tf.Variable(self.initializer([self.conv_kernel_size, self.conv_kernel_size, self.nb_in_channels, self.nb_out_channels]))
        self.kernel1 = tf.Variable(self.initializer([self.conv_kernel_size, self.conv_kernel_size, self.nb_out_channels, self.nb_out_channels]))
        
        self.bias0 = tf.Variable(tf.zeros(shape=[self.nb_out_channels]))
        self.bias1 = tf.Variable(tf.zeros(shape=[self.nb_out_channels]))

        if use_batchnorm:
            self.batch_norm0 = BatchNormalization(nb_channels=self.nb_out_channels)
            self.batch_norm1 = BatchNormalization(nb_channels=self.nb_out_channels)
    
    def apply_conv(self, input, kernel, bias, batch_norm, is_training):
        conv = tf.nn.conv2d(input=input, filters=kernel, strides=[1, 1, 1, 1], padding=self.padding)
        conv = tf.nn.bias_add(conv, bias)
        if batch_norm:
            conv = batch_norm(conv, training=is_training)
        return tf.nn.relu(conv)

    @abstractmethod
    def __call__(self, input, is_training):
        pass