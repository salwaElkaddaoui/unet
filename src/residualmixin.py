import tensorflow as tf
class ResidualMixin:
   
    def initialize_skip_connection(self):
        self.skip_connection_kernel = tf.Variable(self.initializer([1, 1, self.nb_in_channels, self.nb_out_channels]))
        self.skip_connection_bias = tf.Variable(tf.zeros(shape=[self.nb_out_channels]))

    def apply_skip_connection(self, input, conv):
        input_depth_changed = tf.nn.conv2d(input=input, filters=self.skip_connection_kernel, strides=[1, 1, 1, 1], padding=self.padding)
        input_depth_changed = tf.nn.bias_add(input_depth_changed, self.skip_connection_bias)
        
        # Ensure input and output shapes match
        shape_diff = tf.shape(input_depth_changed) - tf.shape(conv)
        padding = [
            [0, 0],  # Batch dimension
            [shape_diff[1]//2, (shape_diff[1]+1)//2],  # Height padding
            [shape_diff[2]//2, (shape_diff[2]+1)//2],  # Width padding
            [0, 0]   # Channels
        ]
        conv = tf.pad(conv, paddings=padding)
        return tf.add(input_depth_changed, conv)
