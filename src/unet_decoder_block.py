import tensorflow as tf

class UnetDecoderBlock:
    def __init__(self, conv_kernel_size, up_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True):
        self.nb_in_channels = nb_in_channels
        self.nb_out_channels = nb_out_channels
        self.up_kernel_size = up_kernel_size
        self.conv_kernel_size = conv_kernel_size
        self.padding = padding
        self.use_batchnorm = use_batchnorm

        if initializer=="he_normal":
            self.initializer = tf.compat.v1.initializers.he_normal

    def __call__(self, x):
        raise NotImplementedError("Subclasses must implement the `__call__` method.")


class BasicDecoderBlock(UnetDecoderBlock):
    """
    A basic decoder block for the U-Net architecture, as described in the original U-Net paper.

    conv2d_transpose -> concat -> conv2d -> batchnorm -> relu ->  conv2d -> batchnorm -> relu
    """
    def __init__(self):
        super().__init__()
        
        #kernels definition
        self.up = tf.Variable(self.initializer(shape=[self.up_kernel_size, self.up_kernel_size, self.nb_out_channels, self.nb_in_channels])), # conv2d_transpose (upsampling) [height, width, nb_out_channels, nb_in_channels]
        self.conv0 = tf.Variable(self.initializer([self.conv_kernel_size, self.conv_kernel_size, self.nb_in_channels, self.nb_out_channels])),  # conv2d [Conv_kernel, nb_input_channels, nb_output_channels]
        self.conv1 = tf.Variable(self.initializer([self.conv_kernel_size, self.conv_kernel_size, self.nb_out_channels, self.nb_out_channels]))  # conv2d [Conv_kernel, nb_input_channels, nb_output_channels]

    def __call__(self, previous_decoder_output, opposite_encoder_output):
        up = tf.nn.conv2d_transpose(    input=previous_decoder_output, \
                                        filters=self.up, \
                                        output_shape=[tf.shape(previous_decoder_output)[0], previous_decoder_output.shape[1]*2, previous_decoder_output.shape[2]*2, previous_decoder_output.shape[-1]//2], \
                                        strides=[1, 2, 2, 1], \
                                        padding=self.padding)
        
        concat = tf.concat([opposite_encoder_output[:, :up.shape[1], :up.shape[2], :], up], axis=-1) 
        
        conv = tf.nn.conv2d(concat, self.conv0, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv  = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu(conv)


        conv = tf.nn.conv2d(conv, self.conv1, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv  = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu(conv)

        return conv


        

class ResidualDecoderBlock(UnetDecoderBlock):
    """
    This class implements a decoder block with a residual connection, inspired by 
    ResNet-style architectures as described in the paper "Deep Residual Learning for 
    Image Recognition" by He et al.
    """
    def __init__(self):
        super().__init__()
        
        #kernels definition
        self.up = tf.Variable(self.initializer(shape=[self.up_kernel_size, self.up_kernel_size, self.nb_out_channels, self.nb_in_channels])), # conv2d_transpose (upsampling) [height, width, nb_out_channels, nb_in_channels]
        self.conv0 = tf.Variable(self.initializer([self.conv_kernel_size, self.conv_kernel_size, self.nb_in_channels, self.nb_out_channels])),  # conv2d [Conv_kernel, nb_input_channels, nb_output_channels]
        self.conv1 = tf.Variable(self.initializer([self.conv_kernel_size, self.conv_kernel_size, self.nb_out_channels, self.nb_out_channels]))  # conv2d [Conv_kernel, nb_input_channels, nb_output_channels]

    def __call__(self, previous_decoder_output, opposite_encoder_output):
        up = tf.nn.conv2d_transpose(    input=previous_decoder_output, \
                                        filters=self.up, \
                                        output_shape=[
                                            tf.shape(previous_decoder_output)[0], \
                                                      previous_decoder_output.shape[1]*2, \
                                                        previous_decoder_output.shape[2]*2, \
                                                            previous_decoder_output.shape[-1]//2 \
                                                            ], \
                                        strides=[1, 2, 2, 1], \
                                        padding=self.padding)
        
        concat_shape_diff = tf.shape(opposite_encoder_output) - tf.shape(up) #we suppose that opposite_encoder_output.shape > up.shape
                    
        concat = tf.concat( values= [opposite_encoder_output[:, \
                                                    concat_shape_diff[1]//2:up.shape[1]+((concat_shape_diff[1]+1)//2), \
                                                    concat_shape_diff[2]//2:up.shape[2]+((concat_shape_diff[2]+1)//2), \
                                                    :],\
                                    up],\
                            axis=-1) 
        
        conv = tf.nn.conv2d(concat, self.conv0, strides=[1, 1, 1, 1], padding=self.padding)
        if self.use_batchnorm:
            conv  = tf.nn.batch_normalization(conv)
        conv = tf.nn.relu(conv)

        conv = tf.nn.conv2d(conv, self.conv1, strides=[1, 1, 1, 1], padding=self.padding)
        
        # the skip connection
        shape_diff = tf.shape(concat) - tf.shape(conv)
        padding = [
            [0, 0],  # No padding for batch dimension
            [shape_diff[1]//2, (shape_diff[1]+1)//2],  # Pad height (before and after)
            [shape_diff[2]//2, (shape_diff[2]+1)//2],  # Pad width (before and after)
            [0, 0]  # No padding for channels
        ]
        conv = tf.pad(conv, paddings=padding)
        conv = tf.add(concat, conv)
        if self.use_batchnorm:
            conv = tf.nn.batch_normalization(conv)
        
        conv = tf.nn.relu(conv)

        return conv
