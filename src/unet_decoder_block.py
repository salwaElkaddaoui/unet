import tensorflow as tf
from abc import abstractmethod
from unet_block import UnetBlock

class UnetDecoderBlock(UnetBlock):
    def __init__(self, nb_classes, conv_kernel_size, deconv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True, is_last=False):

        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)
        self.nb_classes = nb_classes
        self.deconv_kernel_size = deconv_kernel_size
        self.is_last = is_last
        self.deconv_kernel = tf.Variable(self.initializer(shape=[self.deconv_kernel_size, self.deconv_kernel_size, self.nb_out_channels, self.nb_in_channels])) # conv2d_transpose (upsampling) [height, width, nb_out_channels, nb_in_channels]
        self.deconv_bias = tf.Variable(tf.zeros(shape=[self.nb_out_channels]))
        
        if self.is_last:
            self.last_kernel = tf.Variable(self.initializer([1, 1, self.nb_out_channels, self.nb_classes]))
            self.last_bias = tf.Variable(tf.zeros(shape=[self.nb_classes]))

    @abstractmethod
    def __call__(self, x):
        pass


class BasicDecoderBlock(UnetDecoderBlock):
    """
    A basic decoder block for the U-Net architecture, as described in the original U-Net paper.

    conv2d_transpose -> concat -> conv2d -> batchnorm -> relu ->  conv2d -> batchnorm -> relu
    """
    def __init__(self, nb_classes, conv_kernel_size, deconv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True, is_last=False):
        super().__init__(nb_classes, conv_kernel_size, deconv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm, is_last)
               
    def __call__(self, previous_decoder_output, opposite_encoder_output, is_training):
        
        deconv = tf.nn.conv2d_transpose(input=previous_decoder_output, \
                                        filters=self.deconv_kernel, \
                                        output_shape=[tf.shape(previous_decoder_output)[0], tf.shape(previous_decoder_output)[1]*2, tf.shape(previous_decoder_output)[2]*2, tf.shape(previous_decoder_output)[-1]//2], \
                                        strides=[1, 2, 2, 1], \
                                        padding=self.padding)
        
        deconv = tf.nn.bias_add(deconv, self.deconv_bias) 

        if tf.shape(opposite_encoder_output)[1] < tf.shape(deconv)[1]:
            shape_diff = tf.shape(deconv) - tf.shape(opposite_encoder_output) 
            padding = [
                [0, 0],  # No padding for batch dimension
                [shape_diff[1]//2, (shape_diff[1]+1)//2],  # Pad height (before and after)
                [shape_diff[2]//2, (shape_diff[2]+1)//2],  # Pad width (before and after)
                [0, 0]  # No padding for channels
            ]
            concat = tf.concat( [
                                    tf.pad(opposite_encoder_output, padding),
                                    deconv
                                    # deconv[:, start:end, start:end, :]
                                ],
                                axis=-1)
        else:
            shape_diff = tf.shape(opposite_encoder_output)[1] - tf.shape(deconv)[1]# I assume that H==W, which means tf.shape(up)[1] == tf.shape(up)[2]
            start = shape_diff//2 
            end = start + tf.shape(deconv)[1]
            concat = tf.concat( [
                                    opposite_encoder_output[:, start:end, start:end, :],
                                    deconv
                                ],
                                axis=-1)
            

        conv = self.apply_conv(concat, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_conv(conv, self.kernel1, self.bias1, self.batch_norm1, is_training)

        if self.is_last:
            conv = tf.nn.conv2d(conv, self.last_kernel, strides=[1, 1, 1, 1], padding=self.padding)
            logits = tf.nn.bias_add(conv, self.last_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            probabilities = tf.clip_by_value(probabilities, 1e-7, 1.0 - 1e-7) #to avoid log(0)
    
            return probabilities
        else:
            return conv

class ResidualDecoderBlock(UnetDecoderBlock):
    """
    This class implements a decoder block with a residual connection, inspired by 
    ResNet-style architectures as described in the paper "Deep Residual Learning for 
    Image Recognition" by He et al.
    """
    def __init__(self, nb_classes, conv_kernel_size, deconv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True, is_last=False):
        super().__init__(nb_classes, conv_kernel_size, deconv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm, is_last)
        
        self.skip_connection_kernel = tf.Variable(self.initializer(shape=[1, 1, self.nb_in_channels, self.nb_out_channels])) # 1x1 conv2d 
        self.skip_connection_bias = tf.Variable(tf.zeros(shape=[self.nb_out_channels]))

    def __call__(self, previous_decoder_output, opposite_encoder_output, is_training):
        deconv = tf.nn.conv2d_transpose(    input=previous_decoder_output, \
                                        filters=self.deconv_kernel, \
                                        output_shape=[
                                            tf.shape(previous_decoder_output)[0], \
                                                      tf.shape(previous_decoder_output)[1]*2, \
                                                        tf.shape(previous_decoder_output)[2]*2, \
                                                            tf.shape(previous_decoder_output)[-1]//2 \
                                                            ], \
                                        strides=[1, 2, 2, 1], \
                                        padding=self.padding)
        
        deconv = tf.nn.bias_add(deconv, self.deconv_bias) 

        if tf.shape(opposite_encoder_output)[1] < tf.shape(deconv)[1]:
            shape_diff = tf.shape(deconv) - tf.shape(opposite_encoder_output) 
            padding = [
                [0, 0],  # No padding for batch dimension
                [shape_diff[1]//2, (shape_diff[1]+1)//2],  # Pad height (before and after)
                [shape_diff[2]//2, (shape_diff[2]+1)//2],  # Pad width (before and after)
                [0, 0]  # No padding for channels
            ]
            concat = tf.concat( [
                                    tf.pad(opposite_encoder_output, padding),
                                    deconv
                                    # deconv[:, start:end, start:end, :]
                                ],
                                axis=-1)
        else:
            shape_diff = tf.shape(opposite_encoder_output)[1] - tf.shape(deconv)[1]# I assume that H==W, which means tf.shape(up)[1] == tf.shape(up)[2]
            start = shape_diff//2 
            end = start + tf.shape(deconv)[1]
            concat = tf.concat( [
                                    opposite_encoder_output[:, start:end, start:end, :],
                                    deconv
                                ],
                                axis=-1)

        conv = self.apply_conv(concat, self.kernel0, self.bias0, self.batch_norm0, is_training)
        
        
        conv = tf.nn.conv2d(conv, self.kernel1, strides=[1, 1, 1, 1], padding=self.padding)
        conv = tf.nn.bias_add(conv, self.bias1)
        
        # the skip connection
        concat_depth_changed = tf.nn.conv2d(input=concat, filters=self.skip_connection_kernel, strides=[1, 1, 1, 1], padding=self.padding)
        concat_depth_changed = tf.nn.bias_add(concat_depth_changed, self.skip_connection_bias)
        
        shape_diff = tf.shape(concat_depth_changed) - tf.shape(conv)
        padding = [
            [0, 0],  # No padding for batch dimension
            [shape_diff[1]//2, (shape_diff[1]+1)//2],  # Pad height (before and after)
            [shape_diff[2]//2, (shape_diff[2]+1)//2],  # Pad width (before and after)
            [0, 0]  # No padding for channels
        ]
        conv = tf.pad(conv, paddings=padding)
        conv = tf.add(concat_depth_changed, conv)
        if self.use_batchnorm:
            conv = self.batch_norm1(conv, training=is_training)
        
        conv = tf.nn.relu(conv)
        if self.is_last:
            logits = tf.nn.conv2d(conv, self.last_kernel, strides=[1, 1, 1, 1], padding=self.padding)
            probabilities = tf.nn.softmax(logits, axis=-1)
            probabilities = tf.clip_by_value(probabilities, 1e-7, 1.0 - 1e-7) #to avoid log(0)
    
            return probabilities
        else:
            return conv
