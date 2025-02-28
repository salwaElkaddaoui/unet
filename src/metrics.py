import tensorflow as tf

def compute_iou(y_pred: tf.Tensor, y_true: tf.Tensor)->tf.Tensor:

    num_classes = tf.shape(y_pred)[-1]
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=num_classes)
    y_pred = tf.cast(y_pred, tf.bool)
    y_true = tf.cast(y_true, tf.bool)

    intersection = tf.reduce_sum(tf.cast(tf.logical_and(y_pred, y_true), tf.float32), axis=[0, 1, 2])
    union = tf.reduce_sum(tf.cast(tf.logical_or(y_pred, y_true), tf.float32), axis=[0, 1, 2])

    class_absent = tf.equal(tf.reduce_sum(tf.cast(y_true, tf.float32), axis=[0, 1, 2]), 0)
    false_positive = tf.greater(tf.reduce_sum(tf.cast(y_pred, tf.float32), axis=[0, 1, 2]), 0)

    # If class is absent in y_true:
    # - If false positives exist (i.e., model wrongly predicts class) → IoU = 0
    # - Otherwise (both y_true and y_pred are absent) → IoU = 1
    iou = tf.where(class_absent, tf.where(false_positive, 0.0, 1.0), intersection / (union + 1e-10))

    # Mean IoU over all classes
    miou = tf.reduce_mean(iou) 

    return iou, miou


def compute_pixel_accuracy(y_pred:tf.Tensor, y_true:tf.Tensor)->tf.Tensor:
    y_pred = tf.argmax(y_pred, axis=-1)  # get the estimated class index
    correct = tf.equal(y_pred, tf.argmax(y_true, axis=-1))  
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def compute_pixel_error(y_pred:tf.Tensor, y_true:tf.Tensor)->tf.Tensor:
    return 1-compute_pixel_accuracy(y_pred, y_true)