import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras import backend as backend

class Neural_Network(object):
    """Class for neural network"""
    def __init__(self, eta = 0.01, epochs = 50, verbose=0, batch_size =64, lear_rate = 1e-4,num_classes = 63):
        self.eta = eta
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.lear_rate = lear_rate
        self.num_classes = num_classes

    def model_neural_network(input_shape, activation_pattern, activation_last_dense_layer, padding_pattern, metrics_pattern):
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        #Define the object
        model = Sequential()
        #Define the neural network
        model.add(Conv2D(filters = 16, kernel_size = 2, padding = padding_pattern, activation = activation_pattern, input_shape = input_shape))
        model.add(MaxPooling2D(pool_size = 2))
        model.add(Conv2D(filters = 32, kernel_size = 2, padding = padding_pattern, activation = activation_pattern))
        model.add(MaxPooling2D(pool_size = 2))
        model.add(Conv2D(filters = 64, kernel_size = 2, padding = padding_pattern, activation = activation_pattern))
        model.add(MaxPooling2D(pool_size = 2))
        model.add(Flatten())
        model.add(Dense(500, activation = activation_pattern))
        model.add(Dense(63, activation = activation_last_dense_layer))
        #Define Type compiler
        model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = [metrics_pattern], lr = self.lear_rate)

        return model


