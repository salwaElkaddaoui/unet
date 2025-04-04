import tensorflow as tf
from abc import abstractmethod
from unet_block import UnetBlock
from residualmixin import ResidualMixin

class UnetDecoderBlock(UnetBlock):
    def __init__(self, num_classes, conv_kernel_size, deconv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer="he_normal", use_batchnorm=True, is_last=False):
        super().__init__(conv_kernel_size, nb_in_channels, nb_out_channels, padding, initializer, use_batchnorm)
        self.num_classes = num_classes
        self.deconv_kernel_size = deconv_kernel_size
        self.is_last = is_last
        self.deconv_kernel = tf.Variable(self.initializer(shape=[self.deconv_kernel_size, self.deconv_kernel_size, self.nb_out_channels, self.nb_in_channels]), name="deconv_kernel") # conv2d_transpose (upsampling) [height, width, nb_out_channels, nb_in_channels]
        self.deconv_bias = tf.Variable(tf.zeros(shape=[self.nb_out_channels]), name="deconv_bias")
        if self.is_last:
            self.last_kernel = tf.Variable(self.initializer([1, 1, self.nb_out_channels, self.num_classes]), name="last_conv_kernel")
            self.last_bias = tf.Variable(tf.zeros(shape=[self.num_classes]), name="last_conv_bias")

    def deconv_and_concat(self, previous_decoder_output, opposite_encoder_output):
        deconv = tf.nn.conv2d_transpose(
            input=previous_decoder_output,
            filters=self.deconv_kernel,
            output_shape=[
                tf.shape(previous_decoder_output)[0], 
                tf.shape(opposite_encoder_output)[1],
                tf.shape(opposite_encoder_output)[2],
                # tf.shape(previous_decoder_output)[1] * 2, 
                # tf.shape(previous_decoder_output)[2] * 2, 
                tf.shape(previous_decoder_output)[-1] // 2
            ],
            strides=[1, 2, 2, 1],
            padding=self.padding
        )
        deconv = tf.nn.bias_add(deconv, self.deconv_bias)
        diff = tf.shape(opposite_encoder_output)[1] - tf.shape(deconv)[1]
        if diff < 0:  # padding the encoder output
            diff = -diff
            padding = [[0, 0], [diff//2, (diff+1)//2], [diff//2, (diff+1)//2], [0, 0]]
            opposite_encoder_output = tf.pad(opposite_encoder_output, padding)
        else:  # cropping the encoder output
            start, end = diff // 2, diff // 2 + tf.shape(deconv)[1]
            opposite_encoder_output = opposite_encoder_output[:, start:end, start:end, :]
        return tf.concat([opposite_encoder_output, deconv], axis=-1)
    
    def apply_head(self, conv):
        conv = tf.nn.conv2d(conv, self.last_kernel, strides=[1, 1, 1, 1], padding=self.padding)
        logits = tf.nn.bias_add(conv, self.last_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        probabilities = tf.clip_by_value(probabilities, 1e-7, 1.0 - 1e-7) #to avoid log(0)

        return probabilities
    
    @abstractmethod
    def __call__(self, x):
        pass


class BasicDecoderBlock(UnetDecoderBlock):
    def __call__(self, previous_decoder_output, opposite_encoder_output, is_training):
        concat = self.deconv_and_concat(previous_decoder_output, opposite_encoder_output)
        conv = self.apply_conv(concat, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_conv(conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        if self.is_last:
            return self.apply_head(conv)
        else:
            return conv

class ResidualDecoderBlock(UnetDecoderBlock, ResidualMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_skip_connection()

    def __call__(self, previous_decoder_output, opposite_encoder_output, is_training):
        concat = self.deconv_and_concat(previous_decoder_output, opposite_encoder_output)
        conv = self.apply_conv(concat, self.kernel0, self.bias0, self.batch_norm0, is_training)
        conv = self.apply_residual_conv(concat, conv, self.kernel1, self.bias1, self.batch_norm1, is_training)
        if self.is_last:
            return self.apply_head(conv)
        else:
            return conv
