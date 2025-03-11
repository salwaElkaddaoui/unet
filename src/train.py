import gc
gc.collect()
import os, sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
import tensorflow as tf
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

    # @tf.function
    def train_iteration(self, model, image_batch, mask_batch):
        with tf.device('/GPU:0'):
            with tf.GradientTape() as tape:
                mask_pred = model(image_batch, is_training=True)
                loss = self.loss_fn(y_pred=mask_pred, y_true=mask_batch)
                # iou, miou = compute_iou(y_pred=mask_pred, y_true=mask_batch)
                # pixel_accuracy = compute_pixel_accuracy(y_pred=mask_pred, y_true=mask_batch)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # return loss, miou, iou, pixel_accuracy

    def evaluate(self, model, dataset):
        total_loss = 0
        total_iou = 0
        total_miou = 0
        total_pixel_accuracy = 0
        num_iterations = 0

        for (image_batch, mask_batch) in dataset:
            mask_pred = model(image_batch, is_training=False)
            loss = self.loss_fn(y_pred=mask_pred, y_true=mask_batch)
            iou, miou = compute_iou(y_pred=mask_pred, y_true=mask_batch)
            pixel_accuracy = compute_pixel_accuracy(y_pred=mask_pred, y_true=mask_batch)

            total_loss += loss
            total_iou += iou
            total_miou += miou
            total_pixel_accuracy += pixel_accuracy
            num_iterations += 1

        avg_loss = total_loss / num_iterations
        avg_iou = total_iou / num_iterations
        avg_miou = total_miou / num_iterations
        avg_pixel_accuracy = total_pixel_accuracy / num_iterations

        return avg_loss, avg_miou, avg_iou, avg_pixel_accuracy

    def __call__(self, model, train_dataset, val_dataset, epochs, checkpoint_dir, log_dir):
        """
        Train model and save checkpoints
        """
        writer = tf.summary.create_file_writer(log_dir)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

        num_iterations = 0
        for epoch in range(epochs):
            #train on training set
            for iteration, (image_batch, mask_batch) in enumerate(train_dataset):
                with tf.device('/GPU:0'):
                    # loss, miou, iou, pixel_accuracy = self.train_iteration(model, image_batch, mask_batch)
                    self.train_iteration(model, image_batch, mask_batch)

                print(f"\rTraining: Epoch {epoch}/{epochs}, Iteration {iteration}", end="", flush=True)
            
                num_iterations += 1
            
                # write to tensorboard
                # with writer.as_default():
                #     tf.summary.scalar(name="Training/Loss", data=loss, step=num_iterations)
                #     tf.summary.scalar(name="Training/Pixel Accuracy", data=pixel_accuracy, step=num_iterations)
                #     tf.summary.scalar(name="Training/Mean IoU", data=miou, step=num_iterations)
                #     for idx, element in enumerate(iou):
                #         tf.summary.scalar(name="Training/IoU Class i"+str(idx)+": ", data=element, step=num_iterations)
            
            #evaluate on training set and validation set
            train_loss, train_miou, train_iou, train_pixel_accuracy = self.evaluate(model, train_dataset)
            print(f"Training: Epoch {epoch}, Loss: {train_loss.numpy():.4f}, Pixel Accuracy: {train_pixel_accuracy.numpy():.4f}")

            val_loss, val_miou, val_iou, val_pixel_accuracy = self.evaluate(model, val_dataset)
            print(f"Validation: Epoch {epoch}, Loss: {val_loss.numpy():.4f}, Pixel Accuracy: {val_pixel_accuracy.numpy():.4f}")
            
            with writer.as_default():
                tf.summary.scalar(name="Training/Loss", data=train_loss, step=epoch)
                tf.summary.scalar(name="Training/Pixel Accuracy", data=train_pixel_accuracy, step=epoch)
                tf.summary.scalar(name="Training/mIoU", data=train_miou, step=epoch)
                for idx, element in enumerate(train_iou):
                    tf.summary.scalar(name="Training/IoU Class "+str(idx)+": ", data=element, step=epoch)

                tf.summary.scalar(name="Validation/Loss", data=val_loss, step=epoch)
                tf.summary.scalar(name="Validation/Pixel Accuracy", data=val_pixel_accuracy, step=epoch)
                tf.summary.scalar("Validation/mIoU", val_miou, step=epoch)
                for idx, element in enumerate(val_iou):
                    tf.summary.scalar(name="Validation/IoU Class "+str(idx)+": ", data=element, step=epoch)

            checkpoint_manager.save()
            
        
        # tf.saved_model.save(model, "./saved_model")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: Config):
    import json
    with open(cfg.dataset.labelmap_path, 'r') as f:
        labelmap = json.load(f)
    num_classes = len(labelmap)

    with open(cfg.dataset.colormap_path, 'r') as f:
        colormap = json.load(f)

    data_processor = DataProcessor(
        img_size=cfg.model.image_size, 
        batch_size=cfg.training.batch_size, 
        num_classes=num_classes,
        colormap=colormap
    )
    train_dataset = data_processor.create_dataset(
        image_paths=cfg.dataset.train_image_path, 
        mask_paths=cfg.dataset.train_mask_path, 
        training=True #for image augmentation
    )
    val_dataset = data_processor.create_dataset(
        image_paths=cfg.dataset.test_image_path, 
        mask_paths=cfg.dataset.test_mask_path, 
        training=False
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

    print(model.count_parameters())

    optimizer = tf.optimizers.Adam(learning_rate=cfg.training.learning_rate)

    trainer = Trainer(loss_fn=loss, optimizer=optimizer)
    trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=cfg.training.num_epochs,
        checkpoint_dir=cfg.training.checkpoint_dir,
        log_dir=cfg.training.log_dir
    )

if __name__ == "__main__":
    main()