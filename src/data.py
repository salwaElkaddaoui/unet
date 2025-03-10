import tensorflow as tf

class DataProcessor:
    def __init__(self, img_size: int, batch_size: int, num_classes: int, colormap: dict):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.colormap = colormap

    def load_data(self, img_path: str, mask_path: str)->tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3) #dataset images are RGB jpg images 
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3) #masks are encoded as grayscale png images
        return image, mask

    def preprocess(self, image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.resize(image, [self.img_size, self.img_size], method='bilinear') / 255.0  
        mask = tf.image.resize(mask, [self.img_size, self.img_size], method='nearest')
        mask_rgb = tf.cast(mask, tf.uint8)
        mask_indices = tf.zeros_like(mask[..., 0], dtype=tf.int32)

        for idx, rgb_list in self.colormap.items():  # Each class index maps to a list of colors (RGB values)
            idx = tf.constant(int(idx), dtype=tf.int32)
            for rgb in rgb_list:
                class_mask = tf.reduce_all(tf.equal(mask_rgb, rgb), axis=-1)
                mask_indices = tf.where(class_mask, idx, mask_indices)

        mask = tf.one_hot(mask_indices, depth=self.num_classes)

        return image, mask


    def augment(self, image: tf.Tensor, mask: tf.Tensor)-> tuple[tf.Tensor, tf.Tensor]:
        choice = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
        
        def rotate():
            return tf.image.rot90(image), tf.image.rot90(mask)
        
        def shift():
            """
            shift by padding and cropping
            """
            pad_left = tf.random.uniform((), minval=0, maxval=50, dtype=tf.int32)
            pad_top = tf.random.uniform((), minval=0, maxval=50, dtype=tf.int32)
            
            if tf.random.uniform(()) > 0.5: #shift to the right 
                # 1. pad
                padded_image = tf.pad(image, [[0, 0], [pad_left, 0], [0, 0]], constant_values=0)
                padded_mask = tf.pad(mask, [[0, 0], [pad_left, 0], [0, 0]], constant_values=0)
                # 2. crop
                return padded_image[:image.shape[0], :image.shape[1], :], padded_mask[:mask.shape[0], :mask.shape[1], :]
                
            else: #shift to the bottom
                # 1. pad
                padded_image = tf.pad(image, [[pad_top, 0], [0, 0], [0, 0]], constant_values=0)
                padded_mask = tf.pad(mask, [[pad_top, 0], [0, 0], [0, 0]], constant_values=0)
                # 2. crop
                return padded_image[:image.shape[0], :image.shape[1], :], padded_mask[:mask.shape[0], :mask.shape[1], :]

        image, mask = tf.switch_case(choice, branch_fns={0: rotate, 1: shift})
        return image, mask


    def create_dataset(self, image_paths:str, mask_paths:str, training:bool=True) -> tf.data.Dataset:
        """ 
        Creates a TensorFlow Dataset from image and mask file paths for training and evaluation.
        Args:
            image_paths: path of a .txt file containing the absolute path of images, one per line.
            mask_paths: path of .txt file containing the absolute path of masks, one per line.
            training: Flag to apply image augmentation. Set to True for training, False for evaluation.
        Returns:
            tf.data.Dataset: a batched and preprocessed dataset, ready for training and evaluation.
        """    
        image_paths = tf.data.TextLineDataset(image_paths)
        mask_paths = tf.data.TextLineDataset(mask_paths)
        paths_dataset = tf.data.Dataset.zip((image_paths, mask_paths))
        dataset = paths_dataset.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            dataset = dataset.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.shuffle(100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)