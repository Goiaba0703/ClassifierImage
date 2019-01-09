import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Class_Save(object):
    """Class the save"""
    def __init__(self, eta = 0.01, epochs = 50, verbose=2, batch_size =64, lear_rate = 1e-4, num_classes = 63):
        self.eta = eta
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.lear_rate = lear_rate
        self.num_classes = num_classes

    def save_model(self, model,nomeArquivo):
        model.save(nomeArquivo)

