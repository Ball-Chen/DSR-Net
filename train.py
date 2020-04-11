import os
from network import DSR_Net
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint,TensorBoard
from data_generator import multi_threads_generator
import numpy as np
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras import losses,metrics
from keras.callbacks import LearningRateScheduler

def lr_scheduler(epoch):

    if epoch < 6:
        lr = 1e-3
    else:
        lr = 1e-4

    print '--- lr: ',lr

    return lr

def Mae(y_true, y_pred):

    mae = K.abs(y_true-y_pred)
    mae = K.mean(mae,axis=[1,2,3])
    return K.mean(mae)

def toy_Fmeasure(y_true, y_pred):
    _epsilon =  1e-4
    beta_square = 0.3

    y_pred = tf.where(tf.greater(y_pred, 0.5), tf.ones_like(y_pred), tf.zeros_like(y_pred))

    tp = K.sum(y_true*y_pred,axis=[1,2,3])
    p = K.sum(y_pred,axis=[1,2,3])
    t = K.sum(y_true,axis=[1,2,3])

    P = (tp + _epsilon) / (p + _epsilon)
    R = (tp + _epsilon) / (t + _epsilon)

    fmeasure = ((1 + beta_square)*P*R) / (beta_square*P + R)
    return fmeasure


def train(model, train_data_path, batch_size=8, num_gpus=2):
      
    parallel_model = multi_gpu_model(model,gpus=num_gpus)
    train_generator = multi_threads_generator(train_data_path, batch_size)

    lrate = LearningRateScheduler(lr_scheduler)

    parallel_model.compile(optimizer=SGD(lr=1e-3, momentum=0.9), loss=losses.binary_crossentropy, metrics={'output':[Mae, toy_Fmeasure]})

    parallel_model.fit_generator(train_generator, steps_per_epoch=2500, epochs=16, verbose=1, callbacks=[lrate], workers=2)

    model.save_weights('model/model-final.hdf5')



if __name__ == '__main__':
    model = DSR_Net((384, 384, 3))
    model.load_weights('model/densenet121_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)# load densenet pretrained weights 
    train_data_path = '/data1/Saliency_Dataset/DUTS/DUTS-TR/'
    train(model, train_data_path)

