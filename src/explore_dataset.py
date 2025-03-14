import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from data import DataProcessor
import gc
gc.collect()
import os, sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
import tensorflow as tf
import hydra
from config.config import Config

def count_pixels_per_class(dataset, labelmap):
    """Counts the number of pixels for each class over the entire dataset."""
    num_classes = len(labelmap)
    pixel_counts = np.zeros(num_classes, dtype=int)

    for _, masks in dataset:
        masks = tf.argmax(masks, axis=-1) # Undo The One-Hot Encoding
        masks = tf.reshape(masks, [-1])
        unique, counts = np.unique(masks.numpy(), return_counts=True)

        for u, c in zip(unique, counts):
            pixel_counts[u] += c

    return {labelmap[str(i)]: pixel_counts[i] for i in range(num_classes)}

def visualize_class_distribution(pixel_counts, label_map=None):
    """Plots a bar chart of class distribution."""
    labels = list(pixel_counts.keys())
    counts = list(pixel_counts.values())

    plt.bar(labels, counts, color='blue')
    plt.xlabel("Class")
    plt.ylabel("Pixel Count")
    plt.title("Class Distribution in the Dataset")
    plt.xticks(rotation=45)
    plt.show()

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: Config):
    with open(cfg.dataset.labelmap_path, 'r') as f:
        labelmap = json.load(f)
    num_classes = len(labelmap)

    with open(cfg.dataset.colormap_path, 'r') as f:
        colormap = json.load(f)

    # Load dataset (replace with your dataset)
    data_processor = DataProcessor(
        img_size=cfg.model.image_size, 
        batch_size=cfg.training.batch_size, 
        num_classes=num_classes,
        colormap=colormap
    )  
    train_dataset = data_processor.create_dataset(
        image_paths=cfg.dataset.train_image_path, 
        mask_paths=cfg.dataset.train_mask_path, 
        training=False #for image augmentation
    )

    pixel_counts = count_pixels_per_class(train_dataset, labelmap)
    print(pixel_counts)

    # Visualize the class distribution
    visualize_class_distribution(pixel_counts, labelmap)


if __name__ == "__main__":
    main()