import tensorflow as tf

def loss(predictions, labels):
    # pixel-wise softmax
    # softmax = tf.nn.softmax(logits, axis=-1)
    # softmax = tf.clip_by_value(softmax, 1e-7, 1.0 - 1e-7) #to avoid log(0)
    
    # cross-entropy loss
    loss = -tf.reduce_sum(labels * tf.math.log(predictions), axis=-1)
    return tf.reduce_mean(loss)