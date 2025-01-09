import tensorflow as tf

class UnetEncoderBlock:
    def __init__(self, kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        self.nb_in_channels = nb_in_channels
        self.nb_out_channels = nb_out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_batchnorm = use_batchnorm

        if initializer=="he_normal":
            self.initializer = tf.compat.v1.initializers.he_normal

    def __call__(self, x):
        raise NotImplementedError("Subclasses must implement the `__call__` method.")


class BasicEncoderBlock(UnetEncoderBlock):
    """
    A basic encoder block for the U-Net architecture, as described in the original U-Net paper.

    conv2d -> batchnorm -> relu -> conv2d -> batchnorm -> relu ->  pool2d
    """
    def __init__(self):
        super().__init__()
        
        #kernels definition
        self.conv0 = tf.Variable(self.initializer(shape=[self.kernel_size, self.kernel_size, self.nb_in_channels, self.nb_out_channels])) #[Conv_kernel, nb_input_channels, nb_output_channels]
        self.conv1 = tf.Variable(self.initializer(shape=[self.kernel_size, self.kernel_size, self.nb_out_channels, self.nb_out_channels]))
        #the pool operation does not any kernel definition

    def __call__(self, input):
        conv = tf.nn.conv2d(input=input, filters=self.conv0, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu(conv)

        conv = tf.nn.conv2d(input=conv, filters=self.conv1, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = tf.nn.batch_normalization(conv)
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
    def __init__(self):
        super().__init__()
        self.conv0 = tf.Variable(self.initializer(shape=[self.kernel_size, self.kernel_size, self.nb_in_channels, self.nb_out_channels])) #[Conv_kernel, nb_input_channels, nb_output_channels]
        self.conv1 = tf.Variable(self.initializer(shape=[self.kernel_size, self.kernel_size, self.nb_out_channels, self.nb_out_channels]))
        
    def __call__(self, input):
        conv = tf.nn.conv2d(input=input, filters=self.conv0, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu(conv)

        #the skip connection:
        
        # making sure that the input and the output of the previous conv operation 
        # have the same height and width before adding them up
        shape_diff = tf.shape(input) - tf.shape(conv)
        if shape_diff[1] != 0 or shape_diff[2] != 0:  
            padding = [
                [0, 0],  # No padding for batch dimension
                [shape_diff[1]//2, (shape_diff[1]+1)//2],  # Pad height (before and after)
                [shape_diff[2]//2, (shape_diff[2]+1)//2],  # Pad width (before and after)
                [0, 0]  # No padding for channels
            ]
            conv = tf.pad(conv, paddings=padding)

        conv = tf.nn.conv2d(input=tf.add(input, conv), filters=self.conv1, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu(conv)

        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        return conv, pool


class MobileNetv1EncoderBlock(UnetEncoderBlock):
    """

   This class implements the encoder block of a U-Net architecture using depthwise separable 
    convolutions, as described in the paper "MobileNets: Efficient Convolutional Neural Networks 
    for Mobile Vision Applications" by Howard et al. 
    
    The block includes two consecutive depthwise separable convolution layers followed by a max-pooling layer.

    Depthwise separable convolution is composed of:
    - A depthwise convolution that applies a single filter per input channel.
    - A pointwise convolution that applies 1x1 convolutions to combine the outputs 
      of the depthwise convolution.

    """
    def __init__(self):
        super().__init__()
        self.conv0_depthwise = tf.Variable(self.initializer(shape=[self.kernel_size, self.kernel_size, self.nb_in_channels, 1])) 
        self.conv0_pointwise = tf.Variable(self.initializer(shape=[1, 1, self.nb_in_channels, self.nb_out_channels]))

        self.conv1_depthwise = tf.Variable(self.initializer(shape=[self.kernel_size, self.kernel_size, self.nb_out_channels, 1])) 
        self.conv1_pointwise = tf.Variable(self.initializer(shape=[1, 1, self.nb_out_channels, self.nb_out_channels]))

    def __call__(self, input):

        # depthwise convolution
        conv = tf.nn.conv2d(input=input, filters=self.conv0_depthwise, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu6(conv)

        #pointwise convolution
        conv = tf.nn.conv2d(input=conv, filters=self.conv0_pointwise, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu6(conv)

        #depthwise convolution
        conv = tf.nn.conv2d(input=conv, filters=self.conv1_depthwise, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu6(conv)

        #pointwise convolution
        conv = tf.nn.conv2d(input=conv, filters=self.conv1_pointwise, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu6(conv)

        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        return conv, pool

