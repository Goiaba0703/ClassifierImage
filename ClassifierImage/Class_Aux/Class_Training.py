import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import backend as backend
from sklearn.model_selection import train_test_split


class Class_Training(object):
    """Class training"""
    def __init__(self, eta = 0.01, epochs = 50, verbose=0, batch_size =64, lear_rate = 1e-4, num_classes = 63):
        self.eta = eta
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.lear_rate = lear_rate
        self.num_classes = num_classes

    def div_training_test(images32, labels):
        #Divide as imagens de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(images32, labels, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test
      

    def training(X_test, y_test):
        #Training model
        model.fit(X_train, y_train, batch_size = self.batch_size, epochs = self.epochs, verbose = self.verbose, validation_data = (X_test, y_test))
        score = model.evaluate(X_test, y_test, verbose=0)
        # Print no jupyter
        print('Perda em Teste: {}'.format(score[0]))
        print('Acurácia em Teste: {}'.format(score[1]))

        #Return Visual Studio
        msg_perda_teste = ('Perda em Teste: {}'.format(score[0]))
        msg_acuraria_teste = ('Acurácia em Teste: {}'.format(score[1]))

        return msg_perda_teste, msg_acuraria_teste