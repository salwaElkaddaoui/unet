import tensorflow as tf

class UnetDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size=3, use_batchnorm=True):
        super().__init__()
        self.upsample = tf.keras.layers.UpSampling2D(size=(2,2))
        self.concat = tf.keras.layers.Concatenate()
        self.conv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization() if use_batchnorm else None
        self.conv2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization() if use_batchnorm else None
        
    def call(self, previous_decoder_output, opposite_encoder_output, is_training=False):
        x = self.upsample(previous_decoder_output)
        x = self.concat([x, opposite_encoder_output])
        x = self.conv1(x)
        if self.bn1: x = self.bn1(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        if self.bn2: x = self.bn2(x, training=is_training)
        return tf.nn.relu(x)

class Unet(tf.keras.Model):
    def __init__(self, input_shape, num_classes, use_batchnorm=True):
        super().__init__()
        
        # Load ResNet101 as encoder backbone
        base_model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet', input_shape=input_shape)
        self.encoder_layers = [base_model.get_layer(name).output for name in ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out']]
        self.encoder = tf.keras.models.Model(inputs=base_model.input, outputs=self.encoder_layers)
        
        # Bottleneck
        self.bottleneck = tf.keras.models.Model(inputs=base_model.get_layer('conv5_block1_1_conv').input, outputs=base_model.get_layer('conv5_block3_out').output)

        # Decoder
        self.decoder_blocks = [UnetDecoderBlock(1024, use_batchnorm=use_batchnorm),
                               UnetDecoderBlock(512, use_batchnorm=use_batchnorm),
                               UnetDecoderBlock(256, use_batchnorm=use_batchnorm),
                               UnetDecoderBlock(64, use_batchnorm=use_batchnorm)]

        # Final upsamling layer
        self.final_upsample = tf.keras.layers.UpSampling2D(size=(2,2))

        # Final output layer
        self.final_conv = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, activation='softmax')
    
    def __call__(self, inputs, is_training=False):
        encoder_outputs = self.encoder(inputs, training=False)  # Freeze encoder during training
        x = self.bottleneck(encoder_outputs[-1], training=is_training)
        for decoder_block, encoder_output in zip(self.decoder_blocks, reversed(encoder_outputs)):
            x = decoder_block(previous_decoder_output=x, opposite_encoder_output=encoder_output, training=is_training)
        return self.final_conv(self.final_upsample(x))
