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
from loss import weighted_cross_entropy
import datetime

class Trainer:
    def __init__(self, loss_fn, initial_lr, labelmap):
        self.loss_fn = loss_fn
        self.learning_rate = initial_lr
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.labelmap = labelmap
        
    def lr_schedule(self, epoch):
        factor = tf.math.pow(10, tf.cast((epoch+1) % 2, tf.float32)) #divide lr by 10 every 2 epochs
        return tf.math.divide_no_nan(self.learning_rate, factor) if epoch > 1 else self.learning_rate

    # @tf.function
    def train_iteration(self, model, image_batch, mask_batch):
        with tf.device('/GPU:0'):
            with tf.GradientTape() as tape:
                mask_pred = model(image_batch, is_training=True)
                loss = self.loss_fn(y_pred=mask_pred, y_true=mask_batch)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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
        Train model and save checkpoints, log hyperparameters, metrics and loss to tensorboard.
        """
        log_dir = log_dir+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

        class_writers = []
        for idx in range(len(self.labelmap)):
            class_folder = os.path.join(log_dir, self.labelmap[str(idx)])
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            class_writers.append(tf.summary.create_file_writer(class_folder))
        train_folder = os.path.join(log_dir, "train")
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        train_writer=tf.summary.create_file_writer(train_folder)
        val_folder = os.path.join(log_dir, "val")
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)
        val_writer=tf.summary.create_file_writer(val_folder)
        hyperparameters_folder = os.path.join(log_dir, "hyperparameters")
        if not os.path.exists(hyperparameters_folder):
            os.makedirs(hyperparameters_folder)
        hyperparameters_writer=tf.summary.create_file_writer(hyperparameters_folder)

        for epoch in range(epochs):
            self.learning_rate = self.lr_schedule(epoch=epoch)
            self.optimizer.learning_rate.assign(self.learning_rate)
            
            for iteration, (image_batch, mask_batch) in enumerate(train_dataset):
                with tf.device('/GPU:0'):
                    batch_size, image_size, image_size, channels = tf.shape(image_batch)
                    self.train_iteration(model, image_batch, mask_batch)
                print(f"\rTraining: Epoch {epoch+1}/{epochs}, Iteration {iteration}, LR {self.optimizer.learning_rate.numpy():.1e} ", end="", flush=True)
            
            train_loss, train_miou, train_iou, train_pixel_accuracy = self.evaluate(model, train_dataset)
            print(f"\nEvaluation Trainset: Epoch {epoch+1}, Loss: {train_loss.numpy():.4f}, Pixel Accuracy: {train_pixel_accuracy.numpy():.4f}")

            val_loss, val_miou, val_iou, val_pixel_accuracy = self.evaluate(model, val_dataset)
            print(f"Evaluation Valset: Epoch {epoch+1}, Loss: {val_loss.numpy():.4f}, Pixel Accuracy: {val_pixel_accuracy.numpy():.4f}\n")

            for idx in range(len(self.labelmap)):
                with class_writers[idx].as_default():
                    tf.summary.scalar(name="IoU/Training Set", data=train_iou[idx], step=epoch)
                    tf.summary.scalar(name="IoU/Validation Set", data=val_iou[idx], step=epoch)
            with train_writer.as_default():
                tf.summary.scalar(name="Loss", data=train_loss , step=epoch)
                tf.summary.scalar(name="Pixel Accuracy", data=train_pixel_accuracy , step=epoch)
                tf.summary.scalar(name="Mean IoU", data=train_miou , step=epoch)
            with val_writer.as_default():
                tf.summary.scalar(name="Loss", data=val_loss , step=epoch)
                tf.summary.scalar(name="Pixel Accuracy", data=val_pixel_accuracy , step=epoch)
                tf.summary.scalar(name="Mean IoU", data=val_miou , step=epoch)
            with hyperparameters_writer.as_default():
                tf.summary.scalar(name="Learning Rate", data=self.optimizer.learning_rate , step=epoch)
                tf.summary.text(name="Batch Size", data=tf.strings.as_string(batch_size), step=0)
                tf.summary.text(name="Input Image Size", data=tf.strings.as_string(image_size), step=0)
                tf.summary.text(name="Loss Function", data="weighted cross entropy", step=0)
                
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

    
    trainer = Trainer(loss_fn=weighted_cross_entropy, initial_lr=cfg.training.learning_rate, labelmap=labelmap)
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