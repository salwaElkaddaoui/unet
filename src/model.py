import tensorflow as tf
from unet_encoder_block import BasicEncoderBlock, ResidualEncoderBlock
from unet_bottleneck import BasicBottleneck, ResidualBottleneck
from unet_decoder_block import BasicDecoderBlock, ResidualDecoderBlock


BLOCK_FACTORY = {
    "basic": [BasicEncoderBlock, BasicBottleneck, BasicDecoderBlock],
    "resnet": [ResidualEncoderBlock, ResidualBottleneck, ResidualDecoderBlock],
}

class Unet(tf.Module):

    def __init__(self, in_image_depth: int, nb_classes: int, nb_blocks: int=4, block_type='basic', 
                 padding: str='SAME', nb_initial_filters: int=64, initializer="he_normal", use_batchnorm=True, use_dropout=False):
        """
        in_image_depth: number of chennels (depth) of theinput image of the network
        nb_classes: The number of output classes for the segmentation task.
        nb_blocks: The number of convolutional blocks in the encoder  (same number for the decoder). 
            Default is 4.
        padding: The padding type ('same' or 'valid') for the convolution layers.
            Default is 'SAME'
        nb_initial_filters : The number of convolutional filters in the first block of the encoder.
            Subsequent blocks will have a number of filters that are multiplies of this value.
            Default is 64.
        
        Indices of the blocks:
        0               3
        1               2
        2               1
        3               0
            Bottleneck
        
        """
        super().__init__()

        if  in_image_depth <= 0: 
            raise ValueError("The number of classes should be greater or equal to 1")
        self.in_image_depth = in_image_depth

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
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        
        self.initializer = initializer

        if not (block_type.lower() in ['basic', 'resnet']):
            raise ValueError(f"Block type {block_type.lower()}. Valid values: \'basic\', \'resnet\'")
        self.encoder_class, self.bottleneck_class, self.decoder_class = BLOCK_FACTORY[block_type]
        
        self.encoder_blocks = []
        self.decoder_blocks = []
        for i in range(self.nb_blocks):
            self.encoder_blocks.append(self.encoder_class(  conv_kernel_size=3, 
                                                            nb_in_channels=self.nb_initial_fitlers*2**(i-1) if i>0 else self.in_image_depth, 
                                                            nb_out_channels=self.nb_initial_fitlers*2**i, 
                                                            padding=self.padding,
                                                            initializer=self.initializer, 
                                                            use_batchnorm=self.use_batchnorm,
                                                            use_dropout=self.use_dropout if i >= self.nb_blocks-2 else False)) #use dropout in the last 2 blocks of the contractive path
        
        self.bottleneck = self.bottleneck_class(conv_kernel_size=3, 
                                                nb_in_channels=self.nb_initial_fitlers*2**(self.nb_blocks-1) , 
                                                nb_out_channels=self.nb_initial_fitlers*2**nb_blocks, 
                                                padding=self.padding, 
                                                initializer=self.initializer, 
                                                use_batchnorm=self.use_batchnorm)

        for i in range(self.nb_blocks-1, -1, -1):
            self.decoder_blocks.append(self.decoder_class(  nb_classes = self.nb_classes,
                                                            conv_kernel_size=3, 
                                                            deconv_kernel_size=2, 
                                                            nb_in_channels=self.nb_initial_fitlers*2**(i+1), #ceci sous-entend que i commence de nb_blocks-1
                                                            nb_out_channels=self.nb_initial_fitlers*2**i, 
                                                            padding=self.padding,
                                                            initializer=self.initializer, use_batchnorm=self.use_batchnorm,
                                                            is_last=True if i==0 else False
                                    ))

    def __call__(self, input, is_training=True):

        encoder_blocks_outputs = []
        for i in range(self.nb_blocks):        
            if i==0:
                pool=input
            conv, pool = self.encoder_blocks[i](input=pool, is_training=is_training)
            encoder_blocks_outputs.append(conv)

        bottleneck_output = self.bottleneck(pool, is_training)

        for i in range(self.nb_blocks):
            if i==0:
                output = bottleneck_output
            output = self.decoder_blocks[i](previous_decoder_output=output, 
                                            opposite_encoder_output=encoder_blocks_outputs[self.nb_blocks-(i+1)],
                                            is_training=is_training)
        
        return output