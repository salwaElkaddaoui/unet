import tensorflow as tf

def cross_entropy(y_pred, y_true):
    # cross-entropy loss
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return tf.reduce_mean(loss)


def compute_class_weights_for_batch(batch_mask):
    batch_class_mask = tf.argmax(batch_mask, axis=-1)  # shape [batch_size, height, width]
    batch_size, height, width = tf.shape(batch_class_mask)
    total_pixels = tf.cast(height * width, tf.float32)
    batch_weight_maps = []
    for b in range(batch_size): #class weights are computed image-wise
        class_mask = batch_class_mask[b]
        flattened_mask = tf.reshape(class_mask, [-1])
        unique_classes, _, counts = tf.unique_with_counts(flattened_mask)
        class_frequencies = {unique_classes[i].numpy(): counts[i].numpy() for i in range(len(unique_classes))}
        class_weights = {class_id: total_pixels / (freq + 1e-10) for class_id, freq in class_frequencies.items()}
        weight_map = tf.zeros_like(class_mask, dtype=tf.float32)
        for class_id, weight in class_weights.items():
            weight_map = tf.where(class_mask == class_id, weight, weight_map)
        batch_weight_maps.append(weight_map)
    batch_weight_maps = tf.stack(batch_weight_maps, axis=0)  # shape [batch_size, height, width]
    return batch_weight_maps


def weighted_cross_entropy(y_pred, y_true):
    class_weights = compute_class_weights_for_batch(y_true)
    # class_weights = tf.expand_dims(class_weights, axis=-1)
    cross_entropy_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    weighted_loss = cross_entropy_loss * class_weights 
    return tf.reduce_mean(weighted_loss)

import tensorflow as tf

def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])  # Sum over height and width => [Batch_Size, Num_Classes]
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice_coefficient = (2.0 * intersection + 1e-6) / (union + 1e-6)
    dice_loss_per_class = 1.0 - dice_coefficient
    mean_dice_loss = tf.reduce_mean(dice_loss_per_class)    
    return mean_dice_loss
