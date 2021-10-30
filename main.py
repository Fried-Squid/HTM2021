## TODO LIST | STATUS
# see trello

## TEAM
# Ace    - backend for AI
# Maxime - web frontend
# Lucy   - data aquisition

##CODE
#print("running")
import cv2
import os
import numpy as np
global currentlyProcessing
stack = [] #but its an array

from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout
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
def mp4toframes(video, copyNum):
  global currentlyProcessing
  count=""
  while True:
    try:
      os.mkdir(video.split(".")[0])
      break
    except Exception as e:
      return None
  folder = video.split(".")[0] + str(count)
  print(folder)
  vidcap = cv2.VideoCapture(video)
  success,image = vidcap.read()
  count = 0
  while success:
    image=canny(get_grayscale(remove_noise(image)))
    if copyNum == 0:
      cv2.imwrite("%s/frame%s.jpg" % (folder,str(count)), image)     # save frame as JPEG file
    else:
      cv2.imwrite("%s/CopyNo%s-frame%s.jpg" % (folder,str(copyNum),str(count)), image) 
    if count == 200: #no more than 200 frames
      break
    success,image = vidcap.read()
    #print('Read a new frame: ', success)
    count += 1
  #os.remove(video)
  stack.append(folder)

def drawBox(outputFile, img):
    pass

def euclidean_distance(v):
	(a,b)=v
	return K.sqrt(K.maximum(K.sum(K.square(a-b), axis=1,keepdims=True), K.epsilon()))

def genFrames():
  classes2=[]
  for filename in os.listdir("test vids"):
    if filename.endswith(".mp4"):
      try:
        
        if "COPY" in str(filename): #fix this one day
          pass
        else:
          mp4toframes("test vids/"+filename,classes2.count(filename))
          classes2.append(filename)
        #print(classes2)        
      except Exception as e:
        print(e)

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

def make_pairs(): #TODO: ADD TRANSFORMS AND CONTROLS, BUT FIRST I WANNA TEST
  pairLabels = []
  pairImages = []
  classes = []
  for filename in os.listdir("test vids"):
    if filename.endswith(".mp4"):
      classes.append(filename.split(".mp4")[0])
  numClasses = len(classes)
  idLabels=classes
  keys,values=[],[]
  for eachclass in classes:
    keys.append(eachclass)
    sussyVariable=[]
    for filename in os.listdir("test vids/"+eachclass):
      if filename.endswith(".jpg"):
        sussyVariable.append("test vids/"+eachclass+"/"+filename)
      else:pass
    values.append(sussyVariable)

  foo = dict(zip(keys, values))
  #dict foo = {"blue charge":["bluechrage1","bluecharrge2"]}

  for filename1 in os.listdir("test vids"):
    if filename1.endswith(".mp4"):
      for filename in os.listdir("test vids/"+filename1.split(".")[0]):
        if filename.endswith(".jpg"):
          label = filename1.split(".mp4")[0]
          image = "test vids/"+label+"/"+filename

          #idx generates wrong?????????????? ?? ???? ??? ?? !! ke4?! nf0!! sussy 
          #replaced with foo

          randomSame = np.random.choice(foo[label])

          pairImages.append([np.asarray(Image.open(image)), np.asarray(Image.open(randomSame))])
          pairLabels.append([1])

          labelChosen = label
          while labelChosen == label:
            labelChosen = np.random.choice(classes) #lol a really bad way to do this
          label = labelChosen
          randomDiff = np.random.choice(foo[label])

 
          pairImages.append([np.asarray(Image.open(image)), np.asarray(Image.open(randomDiff))])
          pairLabels.append([0])

  #print(np.array(pairImages)) error is not here ( i hope)
  return (np.array(pairImages), np.array(pairLabels))


import matplotlib
def plot_training(H, plotPath): #'edit this'
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


def plot_predict(model, test_data, BASE_OUTPUT):
  
   
    fig2 = plt.figure(figsize=(20,20))
    for b in range(100,116):
        np.random.shuffle(test_data)
        row0 = test_data[:, 0]
        row1 = test_data[:, 1]
        plt.subplot(4,4,(b-100)+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        #print(predictions)
        im1 = Image.fromarray(np.uint8(matplotlib.cm.gist_earth(row0[b-100])*255))
        im2 = Image.fromarray(np.uint8(matplotlib.cm.gist_earth(row1[b-100])*255))
        concat_im = Image.new('RGB', (im1.width + im2.width, im1.height))
        concat_im.paste(im1, (0, 0))
        concat_im.paste(im2, (im1.width, 0))
        
        imageA = row0[b-100]
        imageB = row1[b-100]
        imageA = np.expand_dims(imageA, axis=-1)
        imageB = np.expand_dims(imageB, axis=-1)
        imageA = np.expand_dims(imageA, axis=0)
        imageB = np.expand_dims(imageB, axis=0)
        imageA = imageA / 255.0
        imageB = imageB / 255.0
        try:
          predictions = model.predict([imageA,imageB])
          plt.xlabel(str(predictions))
        except Exception as e:
          print(e)
        #outputfile = "output/" + str(predictions)
        #cv2.imwrite(outputfile, concat_im)
        plt.imshow(concat_im, cmap=plt.cm.binary)
        
        fig2.savefig('output/prediction.png')
        
genFrames()