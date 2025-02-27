import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
import hydra
from config.config import Config
from model import Unet
from data import DataProcessor
from metrics import compute_iou, compute_pixel_accuracy
from loss import loss

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

    def __call__(self, model, dataset, epochs, checkpoint_dir, log_dir):
        """
        Train model and save checkpoints
        """
        writer = tf.summary.create_file_writer(log_dir)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

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
            checkpoint_manager.save()
        # tf.saved_model.save(model, "./saved_model")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: Config):
    import json
    with open(cfg.dataset.labelmap_path, 'r') as f:
        labelmap = json.load(f)
    num_classes = len(labelmap)

    data_processor = DataProcessor(
        img_size=cfg.model.image_size, 
        batch_size=cfg.training.batch_size, 
        num_classes=num_classes
    )
    train_dataset = data_processor.create_dataset(
        image_paths=cfg.dataset.train_image_path, 
        mask_paths=cfg.dataset.train_mask_path, 
        training=True
    )
    
    model = Unet(
        in_image_depth=cfg.model.in_image_depth,
        num_classes=num_classes,
        nb_blocks=cfg.model.nb_blocks,
        block_type=cfg.model.block_type,
        padding=cfg.model.padding,
        nb_initial_filters=cfg.model.nb_initial_filters,
        initializer=cfg.model.initializer,
        use_batchnorm=cfg.model.use_batchnorm,
        use_dropout=cfg.model.use_dropout,
    )

    optimizer = tf.optimizers.Adam(learning_rate=cfg.training.learning_rate)

    trainer = Trainer(loss_fn=loss, optimizer=optimizer)
    trainer(
        model=model,
        dataset=train_dataset,
        epochs=cfg.training.num_epochs,
        checkpoint_dir=cfg.training.checkpoint_dir,
        log_dir=cfg.training.log_dir
    )

if __name__ == "__main__":
    main()