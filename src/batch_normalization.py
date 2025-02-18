import tensorflow as tf

class BatchNormalization(tf.Module):
    def __init__(self, nb_channels, epsilon=1e-5, momentum=0.9):
        """
        Batch Normalization class.
        
        Args:
            nb_channels (int): Number of features (channels) in the input.
            epsilon (float): Small constant to avoid division by zero.
            momentum (float): Momentum for updating moving averages.
        """
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = tf.Variable(tf.ones([nb_channels]), trainable=True)
        self.beta = tf.Variable(tf.zeros([nb_channels]), trainable=True)

        # Moving averages (for inference)
        self.moving_mean = tf.Variable(tf.zeros([nb_channels]), trainable=False)
        self.moving_variance = tf.Variable(tf.ones([nb_channels]), trainable=False)

    def __call__(self, x, training=True):
        """
        Apply batch normalization to the input tensor.
        
        Args:
            x (tf.Tensor): Input tensor with shape [..., nb_channels].
            training (bool): Indicates if the model is in training mode.
        
        Returns:
            tf.Tensor: Batch-normalized output.
        """
        if training:
            batch_mean, batch_variance = tf.nn.moments(x, axes=list(range(len(x.shape) - 1)), keepdims=False)
            
            self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean)
            self.moving_variance.assign(self.momentum * self.moving_variance + (1 - self.momentum) * batch_variance)
            
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        
        mean = tf.reshape(mean, [1, 1, 1, -1])
        variance = tf.reshape(variance, [1, 1, 1, -1])
        gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
        beta = tf.reshape(self.beta, [1, 1, 1, -1])

        return tf.nn.batch_normalization(x, mean, variance, offset=beta, scale=gamma, variance_epsilon=self.epsilon)
