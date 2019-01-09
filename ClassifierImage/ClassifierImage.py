import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
#Rever como fazer para maper as classes em produção
#No ambiente de teste foi usado a linha abaixo para corrigir o problema ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
sys.path.append('C:/Users/gfsilva/source/repos/ClassifierImage/ClassifierImage/Class_Aux')
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
from Load_Data import *
from Class_Redim_Image import *
from Class_Training import *
from Neural_Network import *
from Class_Save import *
from Class_Load import *

if __name__ == '__main__':
    
    #Necessário baixar as imagens de teste.
    #Necessário mapear para o local da sua maquina onde esta as imagens
    #Baixar as imagens em http://btsd.ethz.ch/shareddata/ 
    #BelgiumTSC_Training e BelgiumTSC_Testing

    #Local das imagens
    ROOT_PATH = "C:/Users/gfsilva/Documents/DataSetClassifierImage/"
    train_data_dir = os.path.join(ROOT_PATH, "BelgiumTSC_Training/Training")
    Val_data_dir = os.path.join(ROOT_PATH, "BelgiumTSC_Testing/Testing")

    #Load modelo treinado
    #print("Iniciando a leitura do modelo treinado")
    #ClassLoad = Class_Load()
    #ClassLoad.load_model(model,"modeloTreinado.h5")

    #Carregando as imagens
    LoadData = Load_Data()
    images, labels = LoadData.load_data(train_data_dir)

    #Redimensionando as imagens para 32x32
    ClassRedimImage = Class_Redim_Image()
    images32 = ClassRedimImage.redim_image(images)

    #Separando as imagens em treino e teste
    ClassTraining = Class_Training(epochs = 500)
    #ClassTraining.div_training_test(images32, labels)
    X_train, X_test, y_train, y_test = ClassTraining.div_training_test(images32, labels)

    #Dividindo os pixels das imagens por 255, para que todos os pixels tenham um valor de 0 a 1 (Facilita o treinamento e melhora a acuracia do modelo)
    X_train, X_test, y_train, y_test, input_shape = ClassRedimImage.redim_pixels(X_train, X_test, y_train, y_test)

    #Criando Modelo
    NeuralNetwork = Neural_Network(lear_rate=0.01)
    model, y_train, y_test, Monitor = NeuralNetwork.model_neural_network(input_shape,'relu','softmax', 'same','accuracy',y_train,y_test)
    #model, y_train, y_test = NeuralNetwork.model_neural_network(input_shape,'relu','softmax', 'same','accuracy',y_train,y_test)
    
    #Treinando modelo
    msg_perda_teste, msg_acuraria_teste = ClassTraining.training(X_train, y_train, X_test, y_test, model, Monitor)
    #msg_perda_teste, msg_acuraria_teste = ClassTraining.training(X_train, y_train, X_test, y_test, model)

    #Print resultado
    print(msg_perda_teste)
    print(msg_acuraria_teste)

    print("Iniciando a gravação do modelo")
    ClassSave = Class_Save()
    print("Finalizando a gravação do modelo")

    ClassSave.save_model(model,"modeloTreinado.h5")

    print('Fim do treinamento')