## TODO LIST | STATUS
# see trello

## TEAM
# Ace    - backend for AI
# Maxime - web frontend
# Lucy   - data aquisition

##CODE
print("running")
import cv2
import pytesseract
from pytesseract import Output
import os
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
global currentlyProcessing
stack = [] #but its an array

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


#greyscale
def get_grayscale(image):
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#get frames from an mp4 and turn it into a folder of jpegs
def mp4toframes(video):
  global currentlyProcessing
  count=""
  while True:
    try:
      os.mkdir(video.split(".")[0])
      break
    except Exception as e:
      return None
  folder = video.split(".")[0] + str(count)
  vidcap = cv2.VideoCapture(video)
  success,image = vidcap.read()
  count = 0
  while success:
    image=canny(get_grayscale(remove_noise(image)))
    cv2.imwrite("%s/frame%s.jpg" % (folder,str(count)), image)     # save frame as JPEG file
    if count == 64:
      break
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
  #os.remove(video)
  stack.append(folder)

def drawBox(outputFile, img):
  try:
      h, w, c = img.shape
  except:
      h, w = img.shape
  boxes = pytesseract.image_to_boxes(img)
  for b in boxes.splitlines():
      b = b.split(' ')
      img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 255, 255), 2)

  cv2.imwrite(outputFile, img)
  cv2.waitKey(0)

#process next folder in the stack
def train():
  IMG_SHAPE = (640, 352, 3)
  BATCH_SIZE = 64
  EPOCHS = 2
  BASE_OUTPUT = "output"
  MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
  PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

  data = make_pairs()

def euclidean_distance(v):
	(a,b)=v
	return K.sqrt(K.maximum(K.sum(K.square(a-b), axis=1,keepdims=True), K.epsilon()))

def genFrames():
  for filename in os.listdir("test vids"):
    if filename.endswith(".mp4"):
      try:
        mp4toframes("test vids/"+filename)
      except:
        pass

def build_siamese_model(inputShape, embeddingDim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    # build the model
    x = Dropout(0.3)(x)
    pooled = GlobalAveragePooling2D()(x)
    model = Model(inputs, Dense(embeddingDim)(pooled))
    # return the model to the calling function
    return model

def make_pairs():
  pairLabels = []
  pairImages = []
  classes = []
  for filename in os.listdir("test vids"):
    if filename.endswith(".mp4"):
      classes.append(filename.split(".mp4")[0])
  numClasses = len(classes)
  print(classes)
  # For each unique label, we compute idxs, which is a list of all indexes that belong to the current class labe
  idx=[]
  idLabels=classes
  counts=[1]
  for i in range(0, numClasses):
      count=0
      for filename in os.listdir("test vids/"+classes[numClasses-1]):
        if filename.endswith(".jpg"):
          count+=1
      print(count)
      idx.append(list(range(sum(counts)-1, sum(counts)+count-1)))
      counts.append(count)

  images=[]
  for filename1 in os.listdir("test vids"): 
    if filename1.endswith(".mp4"):
      for filename in os.listdir("test vids/"+filename1.split(".")[0]):
        if filename.endswith(".jpg"):
          label = filename1.split(".mp4")[0]
          images.append("test vids/"+label+"/"+filename)
    
  
  for filename1 in os.listdir("test vids"): 
    if filename1.endswith(".mp4"):
      for filename in os.listdir("test vids/"+filename1.split(".")[0]):
        if filename.endswith(".jpg"):
          label = filename1.split(".mp4")[0]
          image = "test vids/"+label+"/"+filename

          idxIndex = idLabels.index(label)
          randomSame = images[np.random.choice(idx[idxIndex])]

          pairImages.append([image, randomSame])
          pairLabels.append([1])

          labelChosen = label 
          while labelChosen == label:
            labelChosen = np.random.choice(classes) #lol a really bad way to do this
          label = labelChosen
          idxIndex = idLabels.index(label)
          randomDiff = images[np.random.choice(idx[idxIndex])]

          pairImages.append([image, randomDiff])
          pairLabels.append([0])


  return (np.array(pairImages), np.array(pairLabels))

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
#genFrames()
#train()
