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

class Predictor:
    """Class for making predictions on single images and image batches, with metric computation."""
    def __init__(self, model: tf.Module, num_classes: int, image_size: int)->None:
        self.model = model
        self.image_size = image_size
        self.color_map = self._generate_color_map(num_classes=num_classes)

    def _generate_color_map(self, num_classes: int):
        """Generate a distinct color map with as many colors as needed."""
        colormap = cm.get_cmap("tab10", num_classes)  # Use a colormap with `num_classes` distinct colors
        colors = (colormap(np.linspace(0, 1, num_classes))[:, :3] * 255).astype(np.uint8)
        return tf.constant(colors, dtype=tf.uint8)

    @tf.function
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
        colored_mask = tf.gather(self.color_map, predictions)
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
        num_classes=num_classes
    )
    test_dataset = data_processor.create_dataset(
        image_paths=cfg.dataset.test_image_path, 
        mask_paths=cfg.dataset.test_mask_path, 
        training=False # Set 'training' to False For Inference 
    )

    # Predict
    predictor = Predictor(model=model, num_classes=num_classes, image_size=cfg.model.image_size)
    for batch in test_dataset:
        test_images, test_masks = batch
        predictions = predictor.predict_batch(test_images)

        # Visualize predictions for the first image only
        colored_mask = predictor.visualize_predictions(predictions)

        print(colored_mask.numpy().shape)
        cv2.imshow("image", test_images[0, ...].numpy()[:, :, ::-1])
        cv2.imshow("mask", colored_mask[0, ...].numpy()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()