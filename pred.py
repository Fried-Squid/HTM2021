from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
from main import *
import tensorflow as tf
import os
import numpy as np
import sys

model = tf.keras.models.load_model('output/siamese_model')
(images, labels) = make_pairs()
plot_predict(model, images, "lol")