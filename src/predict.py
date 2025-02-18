import tensorflow as tf

class Predictor:
    def __init__(self, model):
        
        self.model = model

        # Define color map (Colors are defined in the RGB Color System)
        self.color_map = tf.constant([
            [0, 0, 0],        # Background (black)
            [255, 0, 0],      # red light
            [255, 255, 0],    # yellow light
            [0, 255, 0],      # green light
            [100, 100, 100],  # off light
            [0, 0, 255]       # go left
        ], dtype=tf.uint8)

    @tf.function
    def predict(self, input_tensor):
        """Perform inference on a given input tensor."""
        probabilities = self.model(input_tensor, is_training=False)
        predictions = tf.argmax(probabilities, axis=-1)
        return predictions

    def visualize_predictions(self, predictions):
        """Visualize the predictions as a colored mask."""
        colored_mask = tf.gather(self.color_map, predictions)
        return colored_mask