import tensorflow as tf

def compute_iou(y_true, y_pred):
    num_classes = tf.shape(y_true)[-1]
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=num_classes, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2]) # sum over [batch, H, W], returns [Ch]
    union = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2]) - intersection
    iou = intersection / tf.maximum(union, 1e-10)
    miou = tf.reduce_mean(iou)
    return iou, miou

def compute_pixel_accuracy(y_pred, y_true):
    y_pred = tf.argmax(y_pred, axis=-1)  # get the estimated class index
    correct = tf.equal(y_pred, tf.argmax(y_true, axis=-1))  
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

def compute_pixel_error(y_pred, y_true):
    return 1-compute_pixel_accuracy(y_pred, y_true)