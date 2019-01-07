import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Load_Data import *
from Class_Redim_Image import *
from Class_Training import *


if __name__ == '__main__':
    #Local das imagens
    ROOT_PATH = "Data/"
    train_data_dir = os.path.join(ROOT_PATH, "BelgiumTSC_Training/Training")
    Val_data_dir = os.path.join(ROOT_PATH, "BelgiumTSC_Testing/Testing")

    #Carregando as imagens
    LoadData = Load_Data()
    images, labels = LoadData.load_data()

    #Redimensionando as imagens para 32x32
    ClassRedimImage = Class_Redim_Image()
    images32 = ClassRedimImage.redim_image(images)

    #Separando as imagens em treino e teste
    ClassTraining = Class_Training()
    #ClassTraining.div_training_test(images32, labels)
    X_train, X_test, y_train, y_test = ClassTraining.div_training_test(images32, labels)


    #Dividindo os pixels das imagens por 255, para que todos os pixels tenham um valor de 0 a 1 (Facilita o treinamento e melhora a acuracia do modelo)
    X_train, X_test, y_train, y_test, input_shape = ClassRedimImage.redim_pixels(X_train, X_test, y_train, y_test)

    #Treinando modelo
    msg_perda_teste, msg_acuraria_teste = ClassTraining.training(X_test, y_test)

    #Print resultado
    print(msg_perda_teste)
    print(msg_acuraria_teste)