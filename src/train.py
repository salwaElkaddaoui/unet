import tensorflow as tf
# tf.debugging.set_log_device_placement(True)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from metrics import compute_iou, compute_pixel_accuracy

import datetime
log_dir = "logs/train_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class Trainer:
    def __init__(self, loss_fn, optimizer):
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @tf.function
    def train_iteration(self, model, image_batch, mask_batch):
        with tf.device('/GPU:0'):
            with tf.GradientTape() as tape:
                mask_pred = model(image_batch)
                loss = self.loss_fn(mask_pred, mask_batch)
                iou, miou = compute_iou(mask_batch, mask_pred)
                pixel_accuracy = compute_pixel_accuracy(mask_batch, mask_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, miou, pixel_accuracy


    def __call__(self, model, dataset, epochs):
        writer = tf.summary.create_file_writer(log_dir)
        for epoch in range(epochs):
            total_loss = 0.0
            total_iou = 0.0
            total_pixel_accuracy = 0.0
            num_iterations = 0.0
            for iteration, (image_batch, mask_batch) in enumerate(dataset):
                with tf.device('/GPU:0'):
                    loss, miou, pixel_accuracy = self.train_iteration(model, image_batch, mask_batch)
                print(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss.numpy():.4f}, Pixel Accuracy: {pixel_accuracy}")
            
                total_loss += loss
                total_iou += miou
                total_pixel_accuracy += pixel_accuracy
                num_iterations += 1.
            
            avg_loss = total_loss / num_iterations
            avg_iou = total_iou / num_iterations
            avg_pixel_accuracy = total_pixel_accuracy / num_iterations

            # write to tensorboard
            with writer.as_default():
                tf.summary.scalar(name="Loss", data=avg_loss, step=epoch)
                tf.summary.scalar(name="Mean IoU", data=avg_iou, step=epoch)
                tf.summary.scalar(name="Pixel Accuracy", data=avg_pixel_accuracy, step=epoch)