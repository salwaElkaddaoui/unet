import tensorflow as tf

class BatchNormalization:
    def __init__(self, epsilon=1e-5, momentum=0.9, trainable=True):
        """
        Custom implementation of batch normalization.
        
        Args:
            epsilon (float): Small constant to avoid division by zero.
            momentum (float): Momentum for updating moving averages.
        """
        self.epsilon = epsilon
        self.momentum = momentum
        self.trainable = trainable
        self.moving_mean = None
        self.moving_variance = None
    
    def __call__(self, x, training=True):
        """
        Applies batch normalization to the input.
        
        Args:
            x (tf.Tensor): Input tensor (shape: [batch_size, ..., num_features]).
            training (bool): Whether the model is in training mode.
        
        Returns:
            tf.Tensor: Batch-normalized output.
        """
        input_shape = x.shape
        num_features = input_shape[-1]
        
        # Initialize moving mean and variance
        if self.moving_mean is None:
            self.moving_mean = tf.Variable(tf.zeros([num_features]), trainable=False)
        if self.moving_variance is None:
            self.moving_variance = tf.Variable(tf.ones([num_features]), trainable=False)
        
        # Trainable parameters: scale (gamma) and offset (beta)
        gamma = tf.Variable(tf.ones([num_features]), trainable=True)
        beta = tf.Variable(tf.zeros([num_features]), trainable=True)
        
        if training:
            # Compute batch statistics
            batch_mean, batch_variance = tf.nn.moments(x, axes=list(range(len(input_shape) - 1)))
            
            # Update moving statistics
            if self.trainable:
                self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean)
                self.moving_variance.assign(self.momentum * self.moving_variance + (1 - self.momentum) * batch_variance)
            
            mean, variance = batch_mean, batch_variance
        else:
            # Use moving statistics for inference
            mean, variance = self.moving_mean, self.moving_variance
        
        # Normalize input
        x_normalized = (x - mean) / tf.sqrt(variance + self.epsilon)
        
        # Apply scale (gamma) and offset (beta)
        output = gamma * x_normalized + beta
        
        return output
