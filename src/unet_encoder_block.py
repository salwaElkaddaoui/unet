import tensorflow as tf
from batch_normalization import BatchNormalization

class UnetEncoderBlock:
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        self.conv_kernel_size = conv_kernel_size
        self.nb_in_channels = nb_in_channels
        self.nb_out_channels = nb_out_channels
        self.padding = padding
        self.use_batchnorm = use_batchnorm

        if initializer=="he_normal":
            self.initializer = tf.compat.v1.initializers.he_normal()
        if use_batchnorm:
            self.batch_norm = BatchNormalization()

    def __call__(self, x):
        raise NotImplementedError("Subclasses must implement the `__call__` method.")


class BasicEncoderBlock(UnetEncoderBlock):
    """
    A basic encoder block for the U-Net architecture, as described in the original U-Net paper.

    conv2d -> batchnorm -> relu -> conv2d -> batchnorm -> relu ->  pool2d
    """
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)
        
        #kernels definition
        self.conv0 = tf.Variable(self.initializer(shape=[self.conv_kernel_size, self.conv_kernel_size, self.nb_in_channels, self.nb_out_channels])) #[Conv_kernel, nb_input_channels, nb_output_channels]
        self.conv1 = tf.Variable(self.initializer(shape=[self.conv_kernel_size, self.conv_kernel_size, self.nb_out_channels, self.nb_out_channels]))
        #the pool operation does not any kernel definition

    def __call__(self, input, is_training):
        conv = tf.nn.conv2d(input=input, filters=self.conv0, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = self.batch_norm(conv, training=is_training)
        conv = tf.nn.relu(conv)

        conv = tf.nn.conv2d(input=conv, filters=self.conv1, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = self.batch_norm(conv, training=is_training)
        conv = tf.nn.relu(conv)

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
    def __init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        super().__init__(self, conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)
        self.conv0 = tf.Variable(self.initializer(shape=[self.kernel_size, self.kernel_size, self.nb_in_channels, self.nb_out_channels])) #[Conv_kernel, nb_input_channels, nb_output_channels]
        self.conv1 = tf.Variable(self.initializer(shape=[self.kernel_size, self.kernel_size, self.nb_out_channels, self.nb_out_channels]))
        
    def __call__(self, input, is_training):
        conv = tf.nn.conv2d(input=input, filters=self.conv0, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = self.batch_norm(conv, training=is_training)
        conv = tf.nn.relu(conv)

        conv = tf.nn.conv2d(input=conv, filters=self.conv1, strides=[1, 1, 1, 1], padding=self.padding)

        #the skip connection:
        
        # making sure that the input and the output of the previous conv operation 
        # have the same height and width before adding them up
        shape_diff = tf.shape(input) - tf.shape(conv)
        padding = [
            [0, 0],  # No padding for batch dimension
            [shape_diff[1]//2, (shape_diff[1]+1)//2],  # Pad height (before and after)
            [shape_diff[2]//2, (shape_diff[2]+1)//2],  # Pad width (before and after)
            [0, 0]  # No padding for channels
        ]
        conv = tf.pad(conv, paddings=padding)
        conv = tf.add(input, conv)
        if self.use_batchnorm:
            conv = self.batch_norm(conv, training=is_training)
        conv = tf.nn.relu(conv)

        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        return conv, pool

