# -- coding: utf-8 --
# -- coding: utf-8 --
import os
from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from audio_model import analyser
import warnings
warnings.filterwarnings('ignore')


def Train():
    #载入数据
    x_train=pickle.load(open('../cache/X_train.p','rb'))
    y_train=pickle.load(open('../cache/y_train.p','rb'))
    x_test=pickle.load(open('../cache/X_test.p','rb'))
    y_test=pickle.load(open('../cache/y_test.p','rb'))
    # Reshape for convolution
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)


    #K.clear_session()
    SER=analyser()
    net=SER.build_model()
    net.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.01,momentum=0.8),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=tf.keras.metrics.sparse_categorical_accuracy
    )
    best_model_save = ModelCheckpoint('../cache/best_model1.ckpt',
                                      save_best_only=True, monitor='val_sparse_categorical_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=30, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=10,
                                  verbose=1)
    history=net.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_test,y_test),
                    callbacks=[early_stopping,reduce_lr,best_model_save])

    net.save('../cache/audio.h5')
    #Loss Curves
    plt.figure(figsize=(25, 10))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], '-g', linewidth=1.0)
    plt.plot(history.history['val_loss'], 'r', linewidth=1.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=14)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=22)

    # Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history['sparse_categorical_accuracy'], '-g', linewidth=1.0)
    plt.plot(history.history['val_sparse_categorical_accuracy'], 'r', linewidth=1.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=22)
    plt.show()

Train()
