import main
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import os
import numpy as np
import sys
import colorama
colorama.init()

#constants
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], False)

SHAPE = (640, 352, 1)
BATCH_SIZE = 1
EPOCHS = 100
BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

#data prep
print(colorama.Fore.GREEN + "[INFO] prepping TRAIN and TEST data" + colorama.Fore.RESET)
(pTrain, lTrain) =main.make_pairs() #fine to call twice as it has rrandom elements
(pTest, lTest) =  main.make_pairs()

#create network
print(colorama.Fore.GREEN + "[INFO] building network"+ colorama.Fore.RESET)
imgA = Input(shape=SHAPE)
imgB = Input(shape=SHAPE)
featureExtractor = main.build_siamese_model(SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
outputs = Dense(1, activation="sigmoid")(Lambda(main.euclidean_distance)([featsA, featsB]))
model = Model(inputs=[imgA, imgB], outputs=outputs)

#compile it
print(colorama.Fore.GREEN + "[INFO] compiling model"+ colorama.Fore.RESET)
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

#train the model
print(colorama.Fore.GREEN + "[INFO] training model"+ colorama.Fore.RESET)
try: #if this returns "signal: killed" then the data isnt of shape expected
  history = model.fit([pTrain[:, 0], pTrain[:, 1]], lTrain[:],validation_data=([pTest[:, 0], pTest[:, 1]], lTest[:]),batch_size=BATCH_SIZE, epochs=EPOCHS)
except Exception as e:
  print(e)
  sys.exit()

#save it
print(colorama.Fore.GREEN + "[INFO] saving model"+ colorama.Fore.RESET)
model.save(MODEL_PATH)

#plot the data
print(colorama.Fore.GREEN + "[INFO] plotting training data"+ colorama.Fore.RESET)
main.plot_training(history, PLOT_PATH)
