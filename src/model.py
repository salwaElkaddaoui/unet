import tensorflow as tf

class Unet(tf.Module):

    def __init__(self, nb_classes: int, nb_blocks: int=4, padding: str='SAME', nb_initial_filters: int=64):
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

        assert nb_classes>0, "The number of classes should be greater or equal to 1" 
        self.nb_classes = nb_classes
        
        assert nb_blocks>0, "The number of convolution blocks should be greater or equal to 1" 
        self.nb_blocks = nb_blocks

        assert padding.upper() in ["SAME", "VALID"], "Padding should be either \'SAME\' or \'VALID\'"
        self.padding = padding.upper()

        assert nb_initial_filters > 0, "Number of initial filter should be greater or equal to 1"
        self.nb_initial_fitlers = nb_initial_filters

        self.initializer = tf.compat.v1.initializers.he_normal()

        # Definition of convolution kernels
        self.encoder_kernels = []
        self.decoder_kernels = []
        self.bottleneck_kernels = []

        for i in range(nb_blocks):
            # Definition of the kernels of the encoder (contractive path) 
            nb_input_channels_down = self.nb_initial_filters*2**(i-1) if i>0 else 3  # 3 for i==0, 64, 128, 256, 512 ...
            nb_output_channels = self.nb_initial_filters*2**i # 64 for i==0, 128, 256, 512, 1024 ..., the nb of output channels is the same for the down block and its opposite up block
            self.encoder_kernels.append([
                tf.Variable(self.initializer(shape=[3, 3, nb_input_channels_down, nb_output_channels])),  # [Conv_kernel, nb_input_channels, nb_output_channels]
                tf.Variable(self.initializer(shape=[3, 3, nb_output_channels, nb_output_channels])),  # [Conv_kernel, nb_input_channels, nb_output_channels]
                #the pooling operation does not need learnable parameters
            ])
            
            # Definition of the convolution and conv2d_transpose kernels of the decoder (expansive path)
            nb_input_channels_up = self.nb_initial_filters*2**(i+1)
            # nb_output_channels = 64*2**i
            
            self.decoder_kernels.append([
                tf.Variable(self.initializer(shape=[2, 2, nb_output_channels, nb_input_channels_up])), # 2x2 conv2d_transpose (upsampling) [height, width, nb_out_channels, nb_in_channels]
                tf.Variable(self.initializer(shape=[3, 3, nb_input_channels_up, nb_output_channels])),  # 3x3 conv2d [Conv_kernel, nb_input_channels, nb_output_channels]
                tf.Variable(self.initializer(shape=[3, 3, nb_output_channels, nb_output_channels]))  # 3x3 conv2d [Conv_kernel, nb_input_channels, nb_output_channels]   
            ])
            if i==0:
                self.decoder_kernels[-1].append( tf.Variable( self.initializer(shape=[1, 1, nb_output_channels, self.nb_classes]) ) )

        nb_input_channels_down = self.nb_initial_filters*2**(nb_blocks-1) if i>0 else 3  # 3 for i==0, 64, 128, 256, 512 ...
        nb_output_channels = self.nb_initial_filters*2**nb_blocks # 64 for i==0, 128, 256, 512, 1024 ..., the nb of output channels is the same for the down block and its opposite up block
        self.bottleneck_kernels.append([
            tf.Variable(self.initializer(shape=[3, 3, nb_input_channels_down, nb_output_channels])),  # [Conv_kernel, nb_input_channels, nb_output_channels]
            tf.Variable(self.initializer(shape=[3, 3, nb_output_channels, nb_output_channels])),  # [Conv_kernel, nb_input_channels, nb_output_channels]
            #the pooling operation does not need learnable parameters
        ])
    
        