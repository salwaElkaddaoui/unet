import tensorflow as tf
import os
from matplotlib import pyplot as plt


class DataProcessor:
    def __init__(self, img_size, batch_size, num_classes):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def load_data(self, img_path, mask_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3) #dataset images are RGB jpg images 
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1) #masks are encoded as grayscale png images
        return image, mask

    def preprocess(self, image, mask):
        image = tf.image.resize(image, self.img_size, method='bilinear') / 255.0  # Normalize between 0 and 1
        mask = tf.image.resize(mask, self.img_size, method='nearest')  # Normalize between 0 and 1
        mask = tf.one_hot(mask, depth=self.num_classes)        # One-hot encode (the depth dim. is one hot encoded)
        mask = tf.squeeze(mask, axis=-2)
        return image, mask

    def augment(self, image, mask):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.rot90(image)
            mask = tf.image.rot90(mask)
        return image, mask

    def create_dataset(self, image_paths, mask_paths, training=True):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            dataset = dataset.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.shuffle(10).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
