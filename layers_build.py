from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
from keras.backend import tf as ktf
from keras import backend as K

class Split(layers.Layer):

    def __init__(self, k, input_size, **kwargs):
        self.k = k
        self.input_size = input_size
        super(Split, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Split, self).build(input_shape)

    def call(self, inputs, **kwargs):
        k_height, k_width = self.k, self.k
        height, width = self.input_size, self.input_size
        stride_height = height/k_height
        stride_width = width/k_width

        featuremaps = []
        for i in range(self.k):
            for j in range(self.k):
                featuremaps.append(inputs[:, i*stride_height:(i+1)*stride_height, j*stride_width:(j+1)*stride_width, :])
        return featuremaps

    def compute_output_shape(self, input_shape):
        return [tuple([None, self.input_size/self.k, self.input_size/self.k, input_shape[3]])]*self.k*self.k

class Split_channels(layers.Layer):

    def __init__(self, k, **kwargs):
        self.k = k
        super(Split_channels, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Split_channels, self).build(input_shape)

    def call(self, inputs, **kwargs):

        featuremaps = []
        for i in range(self.k*self.k):
                featuremaps.append(inputs[:, :, :, i:i+1])
        return featuremaps

    def compute_output_shape(self, input_shape):
        return [tuple([None, input_shape[1], input_shape[2], 1])]*self.k*self.k



class Combine(layers.Layer):

    def __init__(self, k, **kwargs):
        self.k = k
        super(Combine, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Combine, self).build(input_shape)

    def call(self, inputs, **kwargs):


        origin_featuremaps = []
        for i in range(self.k):
            origin_featuremaps.append(K.concatenate(inputs[i*self.k:(i+1)*self.k],axis=2))
        origin_featuremap = K.concatenate(origin_featuremaps,axis=1)
            
        return origin_featuremap

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[0][1]*self.k, input_shape[0][2]*self.k, input_shape[0][3]])

