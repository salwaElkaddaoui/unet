import tensorflow as tf
from unet_encoder_block import BasicEncoderBlock, ResidualEncoderBlock
from unet_bottleneck import BasicBottleneck, ResidualBottleneck
from unet_decoder_block import BasicDecoderBlock, ResidualDecoderBlock

BLOCK_FACTORY = {
    "basic": [BasicEncoderBlock, BasicBottleneck, BasicDecoderBlock],
    "resnet": [ResidualEncoderBlock, ResidualBottleneck, ResidualDecoderBlock],
}

class Unet(tf.Module):
    def __init__(self, 
                 in_image_depth: int,
                 nb_classes: int, 
                 nb_blocks: int=4, 
                 block_type='basic', 
                 padding: str='SAME', 
                 nb_initial_filters: int=64, 
                 initializer: str="he_normal", 
                 use_batchnorm: bool=True, 
                 use_dropout:bool=False) -> tf.Tensor:
        """
        Unet model for image segmentation with configurable depth and block type.
        
        Indices of the blocks:
        Encoder         Decoder
        0               3
        1               2
        2               1
        3               0
            Bottleneck        
        """
        super().__init__()
        
        filter_sizes = [nb_initial_filters*2**i for i in range(nb_blocks+1)]

        self.encoder_class, self.bottleneck_class, self.decoder_class = BLOCK_FACTORY[block_type]
        self.encoder_blocks = [ self.encoder_class( conv_kernel_size=3, 
                                                    nb_in_channels=in_image_depth if i==0 else filter_sizes[i-1], 
                                                    nb_out_channels=filter_sizes[i], 
                                                    padding=padding,
                                                    initializer=initializer, 
                                                    use_batchnorm=use_batchnorm,
                                                    use_dropout=use_dropout if i >= nb_blocks-2 else False) #use dropout in the last 2 blocks of the encoder
                                for i in range(nb_blocks)
                            ]

        self.bottleneck = self.bottleneck_class(conv_kernel_size=3, 
                                                nb_in_channels=filter_sizes[-2], #nb_initial_filters*2**(nb_blocks-1) , 
                                                nb_out_channels=filter_sizes[-1], #nb_initial_filters*2**nb_blocks, 
                                                padding=padding, 
                                                initializer=initializer, 
                                                use_batchnorm=use_batchnorm)

        self.decoder_blocks=[   self.decoder_class( nb_classes = nb_classes,
                                                    conv_kernel_size=3, 
                                                    deconv_kernel_size=2, 
                                                    nb_in_channels=filter_sizes[i+1], #nb_initial_filters*2**(i+1),
                                                    nb_out_channels=filter_sizes[i], #nb_initial_filters*2**i, 
                                                    padding=padding,
                                                    initializer=initializer, use_batchnorm=use_batchnorm,
                                                    is_last=(i==0))
                                for i in range(nb_blocks-1, -1, -1)
                            ]

    def __call__(self, input, is_training=True):
        encoder_outputs = []
        pool = input
        # encoder
        for encoder_block in self.encoder_blocks:        
            conv, pool = encoder_block(input=pool, is_training=is_training)
            encoder_outputs.append(conv)
        # bottleneck
        output = self.bottleneck(pool, is_training)
        # decoder
        for i, decoder_block in self.decoder_blocks:
            output = decoder_block( previous_decoder_output=output, 
                                    opposite_encoder_output=encoder_outputs[-(i+1)],
                                    is_training=is_training)        
        return output