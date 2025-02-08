import tensorflow as tf
from model import Unet
import requests
import cv2
import numpy as np

if __name__=='__main__':
    model = Unet(in_image_depth=3, nb_classes=2)
    
    image_url= "https://shorturl.at/2v3B3"
    
    response = requests.get(image_url)
    image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    resized_image = cv2.resize(image, (572, 572))
    batch_image = np.expand_dims(resized_image, axis=0)

    output = model(batch_image)
    print(output.shape)