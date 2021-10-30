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
#from psutil import virtual_memory
#ram_gb = virtual_memory().total / 1e9
#print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

#constants
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
SHAPE = (640, 352, 1)
BATCH_SIZE = 15
EPOCHS = 200
BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

#data prep
print("[INFO] prepping TRAIN and TEST data")
(pTrain, lTrain) =make_pairs() #fine to call twice as it has rrandom elements
(pTest, lTest) =  make_pairs()

#create network
print("[INFO] building network")
imgA = Input(shape=SHAPE)
imgB = Input(shape=SHAPE)
featureExtractor = build_siamese_model(SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
outputs = Dense(1, activation="sigmoid")(Lambda(euclidean_distance)([featsA, featsB]))
model = Model(inputs=[imgA, imgB], outputs=outputs)

#compile it
print("[INFO] compiling model")
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

#train the model
print("[INFO] training model")
try: #if this returns "signal: killed" then the data isnt of shape expected
  history = model.fit([pTrain[:, 0], pTrain[:, 1]], lTrain[:],validation_data=([pTest[:, 0], pTest[:, 1]], lTest[:]),batch_size=BATCH_SIZE, epochs=EPOCHS)
except Exception as e:
  print(e)
  sys.exit()

#save it
print("[INFO] saving model")
model.save(MODEL_PATH)

#plot the data
print("[INFO] plotting training data")
plot_training(history, PLOT_PATH)
(images, labels) = make_pairs()
plot_predict(model, images, "lol")
