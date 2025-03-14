import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import hydra
from config.config import Config
from model import Unet
from data import DataProcessor
from metrics import compute_iou, compute_pixel_error
import cv2
from matplotlib import pyplot as plt
import seaborn as sns

class Predictor:
    """Class for making predictions on single images and image batches, with metric computation."""
    def __init__(self, model: tf.Module, image_size: int, labelmap: dict, colormap: dict)->None:
        self.model = model
        self.image_size = image_size
        self.labelmap = labelmap
        self.colormap = colormap
    
    # @tf.function
    def predict_batch(self, input_tensor: tf.Tensor)->tf.Tensor:
        """Perform inference on a batch input tensors."""
        probabilities = self.model(input_tensor, is_training=False)
        predictions = tf.argmax(probabilities, axis=-1)
        return predictions

    def predict_single(self, image_path: str)->tf.Tensor:
        """Perform inference on a single image path."""
        image = self._load_and_preprocess_image(image_path)
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        prediction = self.predict_batch(image)
        return tf.squeeze(prediction, axis=0)  # Remove batch dimension
    
    def visualize_predictions(self, predictions: tf.Tensor)->tf.Tensor:
        """Visualize the predictions as a colored mask."""
        colored_mask = tf.gather(self.colormap, predictions)
        return colored_mask
    
    def _load_and_preprocess_image(self, image_path: str)->tf.Tensor:
        """Load and preprocess a single image."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size]) / 255.0
        return image

    def evaluate(self, image_batch: tf.Tensor, mask_batch: tf.Tensor)->tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute IoU, mean IoU, and pixel error for a batch of predictions."""
        model_output = self.model(image_batch, is_training=False)
        iou, miou = compute_iou(y_pred=model_output, y_true=mask_batch)
        pixel_error = compute_pixel_error(y_pred=model_output, y_true=mask_batch)
        return iou, miou, pixel_error

    
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: Config):
    # Read Label Map to get num_classes
    import json
    with open(cfg.dataset.labelmap_path, 'r') as f:
        labelmap = json.load(f)
    num_classes = len(labelmap)
    with open(cfg.dataset.colormap_path, 'r') as f:
        colormap = json.load(f)

    # Define Model to Load Weights from Checkpoint
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
    checkpoint_path = os.path.join(cfg.training.checkpoint_dir, cfg.training.checkpoint_name)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()  # Restore Weights from Checkpoint

    # Load data
    data_processor = DataProcessor(
        img_size=cfg.model.image_size, 
        batch_size=cfg.training.batch_size, 
        num_classes=num_classes,
        colormap=colormap
    )    
    test_dataset = data_processor.create_dataset(
        image_paths=cfg.dataset.test_image_path, 
        mask_paths=cfg.dataset.test_mask_path, 
        training=False # Set 'training' to False For Inference 
    )

    # Predict
    predictor = Predictor(model=model, image_size=cfg.model.image_size, labelmap=labelmap, colormap=colormap)
    y_pred = []
    y_true = []
    for (image_batch, mask_batch) in test_dataset:
        y_pred.append(tf.reshape(predictor.predict_batch(image_batch), [-1]))
        y_true.append(tf.reshape(tf.argmax(mask_batch, axis=-1), [-1]))

    y_pred = tf.concat(y_pred, axis=0)  # Shape: [total_pixels]
    y_true = tf.concat(y_true, axis=0) 
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=list(labelmap.values()), yticklabels=list(labelmap.values()))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()