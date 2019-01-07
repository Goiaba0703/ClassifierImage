import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Load_Data(object):
    """Class for load data"""
    def __init__(self, eta = 0.01, epochs = 50, verbose=0, batch_size =64, lear_rate = 1e-4, num_classes = 63):
        self.eta = eta
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.lear_rate = lear_rate
        self.num_classes = num_classes

    def load_data(data_dir):
        #Carrega um conjunto de dados e retorna duas listas:
        #Images: Uma lista de arrays Numpy, cada uma representando uma imagem.
        #Labels: Uma lista de números que representam as etiquetas das imagens.
        #Obter todos os subdiretórios do data_dir. Cada um representa um rótulo.
        directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        # Percorre os diretórios de labels e coleta os dados nas duas listas, labels e imagens
        labels = []
        images = []
        for d in directories:
            label_dir = os.path.join(data_dir, d)
            file_names = [os.path.join(label_dir, f) 
                            for f in os.listdir(label_dir) if f.endswith(".ppm")]
        
            # Para cada label, carrega as imagens e adiciona-as à lista de imagens.
            # E adiciona o número do rótulo (ou seja, o nome do diretório) à lista de labels.
            for f in file_names:
                images.append(skimage.data.imread(f))
                labels.append(int(d))
        return images, labels

