import tensorflow as tf

def loss(y_pred, y_true):
    # cross-entropy loss
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return tf.reduce_mean(loss)