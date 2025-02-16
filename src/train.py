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
        return loss, miou, iou, pixel_accuracy


    def __call__(self, model, dataset, epochs):
        writer = tf.summary.create_file_writer(log_dir)
        num_iterations = 0
        for epoch in range(epochs):
            for iteration, (image_batch, mask_batch) in enumerate(dataset):
                with tf.device('/GPU:0'):
                    loss, miou, iou, pixel_accuracy = self.train_iteration(model, image_batch, mask_batch)
                print(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss.numpy():.4f}, Pixel Accuracy: {pixel_accuracy}")
            
                num_iterations += 1
            
                # write to tensorboard
                with writer.as_default():
                    tf.summary.scalar(name="Loss", data=loss, step=num_iterations)
                    tf.summary.scalar(name="Pixel Accuracy", data=pixel_accuracy, step=num_iterations)
                    tf.summary.scalar(name="Mean IoU", data=miou, step=num_iterations)
                    for idx, element in enumerate(iou):
                        tf.summary.scalar(name="IoU Class i"+str(idx)+": ", data=element, step=num_iterations)