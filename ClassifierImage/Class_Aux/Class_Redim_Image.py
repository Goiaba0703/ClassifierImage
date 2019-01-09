import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class Class_Redim_Image(object):
    """Class for resize image"""
    def __init__(self, eta = 0.01, epochs = 50, verbose=2, batch_size =64, lear_rate = 1e-4, num_classes = 63):
        self.eta = eta
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.lear_rate = lear_rate
        self.num_classes = num_classes

    def redim_image(self,images):
        images32 = [skimage.transform.resize(image, (32, 32)) for image in images]
        return images32

    def redim_pixels(self,X_train, X_test, y_train, y_test):
        #Coloca elas no tamanho 32x32
        img_rows, img_cols = 32, 32

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        #Print dos shapes no jupyter

        #print("Shape x_train: {}".format(X_train.shape))
        #print("Shape y_train: {}".format(y_train.shape))
        #print()
        #print("Shape x_test: {}".format(X_test.shape))
        #print("Shape y_test: {}".format(y_test.shape))

        # Reshape dos dados de treino e de teste e input_shape

        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,3)
        input_shape = (img_rows, img_cols, 3)

        # Conversão para float32 
        #X_train = X_train.astype('float32')
        #X_test = X_test.astype('float32')

        # Altera escala dos pixels das imagens para ficar dentro do range de 0 a 1
        X_train /= 255
        X_test /= 255

        # Print in jupyter

        #print('x_train shape:', X_train.shape)
        #print("Exemplos de Treino: {}".format(X_train.shape[0]))
        #print("Exemplos de Teste: {}".format(X_test.shape[0]))
        #print("Input Shape: {}".format(input_shape))
        return X_train, X_test, y_train, y_test, input_shape

