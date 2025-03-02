# U-Net Implementation
This project provides a configurable U-Net for image segmentation, implemented using the TensorFlow 2.x low-level API. It is based on:

- U-Net: "U-Net: Convolutional Networks for Biomedical Image Segmentation" By Olaf Ronneberger, Philipp Fischer, Thomas Brox, 2015
- Residual connections (optional): He et al., 2016 (ResNet)

### Features

- **Connection type**: Choose between basic or residual connections.
- **Network depth**: Configurable by setting the number of blocks.
- **Initial filters**: Start from 64 (default) or any other value.

All parameters are set in **config/config.yaml**, making it easy to experiment with different architectures.

### Data Format

- Training and test images: RGB JPEG images.
- Masks: Grayscale PNG images, where each class is assigned a specific gray level.
- Dataset listing:
    - The list of training images must be stored in a text file, with each line containing the absolute path to an image.
    - A separate text file should contain the absolute paths to the corresponding masks, one per line.
    - The paths to the training and test sets must be specified in the **config/config.yaml** file.
- Label Map:
    - A JSON file must be provided to define the class mappings.
    - The class with index 0 is the background.
    - The path to this label map should be set in **config/config.yaml**.
### Requirements installation

A GPU is required. Install dependencies with:
```
pip install -r requirements.txt
```
### Usage

- **Training:** Run 
```
python src/train.py
```
- **Prediction:** To generate segmentations **from a checkpoint on a set of images**, run 
```
python src/predict.py
``` 
The checkpoint's name and path should be set in **config/config.yaml**.

### Monitoring Training with TensorBoard

- Evaluation metrics and errors can be visualized during training using TensorBoard.
- The logs directory path should be defined in **config/config.yaml**.
- To launch TensorBoard, run:
```
tensorboard --logdir=<yourlogdir>
```