import tensorflow as tf
from unet_encoder_block import BasicEncoderBlock, ResidualEncoderBlock
from unet_bottleneck import BasicBottleneck, ResidualBottleneck
from unet_decoder_block import BasicDecoderBlock, ResidualDecoderBlock

BLOCK_FACTORY = {
    "basic": [BasicEncoderBlock, BasicBottleneck, BasicDecoderBlock],
    "resnet": [ResidualEncoderBlock, ResidualBottleneck, ResidualDecoderBlock],
}

class Unet(tf.Module):

    def __init__(self, nb_classes: int, nb_blocks: int=4, block_type='basic', padding: str='SAME', nb_initial_filters: int=64):
        """
        nb_classes: The number of output classes for the segmentation task.
        nb_blocks: The number of convolutional blocks in the encoder  (same number for the decoder). 
            Default is 4.
        padding: The padding type ('same' or 'valid') for the convolution layers.
            Default is 'SAME'
        nb_initial_filters : The number of convolutional filters in the first block of the encoder.
            Subsequent blocks will have a number of filters that are multiplies of this value.
            Default is 64.
        """
        super().__init__()

        if  nb_classes <= 0: 
            raise ValueError("The number of classes should be greater or equal to 1")
        self.nb_classes = nb_classes
        
        if nb_blocks <= 0:
            raise ValueError("The number of convolution blocks should be greater or equal to 1")
        self.nb_blocks = nb_blocks


        if not (padding.upper() in ["SAME", "VALID"]):
            raise ValueError("Padding should be either \'SAME\' or \'VALID\'")
        self.padding = padding.upper()

        if nb_initial_filters <= 0:
            raise ValueError("Number of initial filter should be greater or equal to 1")
        self.nb_initial_fitlers = nb_initial_filters

        self.initializer = tf.compat.v1.initializers.he_normal()

        if not (block_type.lower() in ['basic', 'resnet']):
            raise ValueError(f"Block type {block_type.lower()}. Valid values: \'basic\', \'resnet\'")
        self.encoder_class, self.bottleneck_class, self.decoder_class = BLOCK_FACTORY[block_type]
        
        self.encoder_blocks = []
        self.decoder_blocks = []
        for i in range(self.nb_blocks):
            self.encoder_blocks.append(self.encoder_class(  conv_kernel_size=3, 
                                                            nb_in_channels=64*2**(i-1), 
                                                            nb_out_channels=64*2**i, 
                                                            padding=self.padding))
            self.decoder_blocks.append()
        
        self.bottleneck = self.bottleneck_class(conv_kernel_size=3, 
                                                nb_in_channels=64*2**(self.nb_blocks-1) , 
                                                nb_out_channels=64*2**nb_blocks, 
                                                padding=self.padding)

        for i in range(nb_blocks, -1, -1):
            self.decoder_blocks.append(self.decoder_class(conv_kernel_size=3, 
                                                          up_kernel_size=2, 
                                                          nb_in_channels=64*2**(i+1), 
                                                          nb_out_channels=64*2**i, 
                                                          padding=self.padding
                                    ))
