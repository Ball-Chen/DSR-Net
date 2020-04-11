import keras
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras import backend
from keras.engine import Layer
from keras import regularizers
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from layers_build import Combine,Split,Split_channels
import numpy as np

def conv_bn_block(x,kernel,num_output,pre_name,dilate_rate=1):

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.Conv2D(filters=num_output, kernel_size=kernel, strides=1,padding='same',dilation_rate=dilate_rate, kernel_regularizer=regularizers.l2(1e-5), use_bias=False,name=pre_name + 'conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=pre_name + 'BN')(x)
    x = layers.Activation('relu')(x)
    return x

def parallel_ASPP(input_tensor,num_output,pre_name,dilate_rates):

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x0 = input_tensor

    ax1 = conv_bn_block(input_tensor, 3 ,num_output/2, pre_name+'a1_', dilate_rates[0])
    bx1 = conv_bn_block(input_tensor, 3 ,num_output/2, pre_name+'b1_', dilate_rates[0]*2)
    x1 = layers.Concatenate(axis=bn_axis)([ax1,bx1])
    x1 = conv_bn_block(x1, 3 ,num_output/2, pre_name+'ab1_')

    x1 = layers.Concatenate(axis=bn_axis)([x0,x1])

    ax2 = conv_bn_block(x1, 3 ,num_output/2, pre_name+'a2_', dilate_rates[1])
    bx2 = conv_bn_block(x1, 3 ,num_output/2, pre_name+'b2_', dilate_rates[1]*2)
    x2 = layers.Concatenate(axis=bn_axis)([ax2,bx2])
    x2 = conv_bn_block(x2, 3 ,num_output/2, pre_name+'ab2_')

    x2 = layers.Concatenate(axis=bn_axis)([x1,x2])

    ax3 = conv_bn_block(x2, 3 ,num_output/2, pre_name+'a3_', dilate_rates[2])
    bx3 = conv_bn_block(x2, 3 ,num_output/2, pre_name+'b3_', dilate_rates[2]*2)
    x3 = layers.Concatenate(axis=bn_axis)([ax3,bx3])
    x3 = conv_bn_block(x3, 3 ,num_output/2, pre_name+'ab3_')

    x3 = layers.Concatenate(axis=bn_axis)([x2,x3])

    refined_x = conv_bn_block(x3,1,num_output,pre_name+'fusion_')

    return refined_x

def SRDB(input_tensor,num_output,k,featuremap_size,pre_name,dilate_rates):

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if k == 1:
        xs = [input_tensor]
    else:
        xs = Split(k, featuremap_size)(input_tensor)

    x = layers.MaxPooling2D(k, strides=k, name=pre_name+'_pool')(input_tensor)
    x = conv_bn_block(x,3,num_output,pre_name+'_attention1')
    x = conv_bn_block(x,3,num_output,pre_name+'_attention2')
    x = layers.Conv2D(k*k, (1, 1), padding='same', activation='sigmoid', name=pre_name+'_attention')(x)
    if k ==1:
        attentions = [x]
    else:
        attentions = Split_channels(k)(x)

    XS = []
    for i, x in enumerate(xs):

        refined_x = parallel_ASPP(x,num_output,pre_name+'_'+str(i)+'_',dilate_rates)
        refined_x = layers.Multiply()([refined_x, attentions[i]])
        XS.append(refined_x)

    if k == 1:
        output_tensor = XS[0]
    else:
        output_tensor = Combine(k)(XS)

    output_tensor = layers.Concatenate(axis=bn_axis)([input_tensor,output_tensor])
    output_tensor = conv_bn_block(output_tensor, 3 ,num_output, pre_name+'fusion_')

    return output_tensor


def transition_block(x, reduction, name, reduce_shape):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    skip = x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    if reduce_shape:
        x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x, skip



def conv_block(x, growth_rate, name):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = backend.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return backend.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return backend.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def DSR_Net(input_shape = (384,384,3)):

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    w, h, c = input_shape
    #encoder: densenet121
    img_input = Input(shape=input_shape, name='input')

    blocks = [6, 12, 24, 16]

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x, x1 = transition_block(x, 0.5, name='pool2', reduce_shape=True)
    x = dense_block(x, blocks[1], name='conv3')
    x, x2 = transition_block(x, 0.5, name='pool3', reduce_shape=True)
    x = dense_block(x, blocks[2], name='conv4')
    x, x3 = transition_block(x, 0.5, name='pool4', reduce_shape=False)
    
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='BN')(x)
    x = layers.Activation('relu', name='relu')(x)


    #decoder
    sr3 = SRDB(x,512,1,w/16,'sr3_',[1,2,5])

    x = layers.Add()([x3,sr3])
    x = conv_bn_block(x,3,512,'decoder3_1_')
    x3 = x = conv_bn_block(x,3,512,'decoder3_2_')

    x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 8)),
                                         int(np.ceil(input_shape[1] / 8))))(x)

    sr2 = SRDB(x,256,2,w/8,'sr2_',[1,2,4])
    x = layers.Add()([x2,sr2])
    x = conv_bn_block(x,3,256,'decoder2_1_')
    x2 = x = conv_bn_block(x,3,256,'decoder2_2_')

    x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                         int(np.ceil(input_shape[1] / 4))))(x)

    sr1 = SRDB(x,128,3,w/4,'sr1_',[1,2,3])
    x = layers.Add()([x1,sr1])
    x = conv_bn_block(x,3,128,'decoder1_1_')
    x1 = x = conv_bn_block(x,3,128,'decoder1_2_')

    x3 = layers.Conv2DTranspose(128, 3, strides=(4, 4), padding='same')(x3)
    x2 = layers.Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(x2)

    x = layers.Concatenate(axis=bn_axis)([x1,x2,x3])

    predict = layers.Conv2D(1, (1, 1), padding='same', name='predict')(x)
    predict = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(predict)
    predict = layers.Activation('sigmoid',name='output')(predict)

    model = Model(img_input, predict)
    
    return model

