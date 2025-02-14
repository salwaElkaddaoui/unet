import tensorflow as tf

def iou(y_pred, y_true, num_classes):
    """
    num_classes = number of foreground classes + 1 (we add 1 for the background)
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    iou_per_class = []
    
    for class_id in range(num_classes):
        true_class = tf.equal(y_true, class_id)  
        pred_class = tf.equal(y_pred, class_id)  
        
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_class, pred_class), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(true_class, pred_class), tf.float32))
        
        iou = tf.math.divide_no_nan(intersection, union)
        iou_per_class.append(iou)
    
    return tf.reduce_mean(iou_per_class)


def pixel_accuracy(y_pred, y_true):
    y_pred = tf.argmax(y_pred, axis=-1)  # get the class index
    correct = tf.equal(y_pred, y_true)  
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

def pixel_error(y_pred, y_true):
    return 1-pixel_accuracy(y_pred, y_true)