import tensorflow as tf
from loss import loss
from model import Unet
from data import DataProcessor
import os, sys
# from model_seq import UNet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Trainer:
    def __init__(self, loss_fn, optimizer):
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @tf.function
    def train_step(self, model, image, mask):
        with tf.GradientTape() as tape:
            mask_pred = model(image)
            loss = self.loss_fn(mask_pred, mask)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def __call__(self, model, dataset, epochs):
        for epoch in range(epochs):
            for step, (image_batch, mask_batch) in enumerate(dataset):
                loss = self.train_step(model, image_batch, mask_batch)
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy():.4f}")

if __name__=='__main__':

    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    trainer = Trainer(loss_fn=loss, optimizer=optimizer)
    model = Unet(in_image_depth=3, nb_classes=5, use_batchnorm=False)
    print("Trainable Variables:", model.trainable_variables)

    # sys.exit()
    # model = UNet(num_classes=5)

    img_size, batch_size, num_classes = [256, 256], 16, 5
    dataprocessor = DataProcessor(img_size, batch_size, num_classes)

    image_folder = "/home/salwa/Documents/code/smartcar/dataset3"
    annotation_folder = "/home/salwa/Documents/code/smartcar/grayscale_masks_3"

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Folder not found: {image_folder}")

    if not os.path.exists(annotation_folder):
        raise FileNotFoundError(f"Folder not found: {annotation_folder}")
    
    image_filenames = os.listdir(image_folder)
    annotation_filenames = os.listdir(annotation_folder)
    
    print(len(image_filenames), len(annotation_filenames))

    # keep only the images whose annotations exist
    image_filenames = [ f for f in image_filenames if f[:-3]+"png" in annotation_filenames]
    print(len(image_filenames))

    image_paths = [os.path.join(image_folder, f) for f in image_filenames]
    mask_paths = [os.path.join(annotation_folder, f[:-3]+"png") for f in image_filenames]

    train_dataset = dataprocessor.create_dataset(image_paths, mask_paths, training=True)

    trainer(model, train_dataset, epochs=10)