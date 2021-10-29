## TODO LIST | STATUS
# see trello

## TEAM
# Ace    - backend for AI
# Maxime - web frontend
# Lucy   - data aquisition

##CODE
#print("running")
import cv2
import pytesseract
from pytesseract import Output
import os
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
global currentlyProcessing
stack = [] #but its an array

from PIL import Image
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
    if count == 200: #no more than 200 frames
      break
    success,image = vidcap.read()
    #print('Read a new frame: ', success)
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

def make_pairs(): #TODO: ADD TRANSFORMS AND CONTROLS, BUT FIRST I WANNA TEST
  pairLabels = []
  pairImages = []
  classes = []
  for filename in os.listdir("test vids"):
    if filename.endswith(".mp4"):
      classes.append(filename.split(".mp4")[0])
  numClasses = len(classes)
  #print(classes)
  # For each unique label, we compute idxs, which is a list of all indexes that belong to the current class labe
  idx=[]
  idLabels=classes
  counts=[1]
  for i in range(0, numClasses):
      count=0
      for filename in os.listdir("test vids/"+classes[numClasses-1]):
        if filename.endswith(".jpg"):
          count+=1
      #print(count)
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

          #print(np.asarray(Image.open(image)).shape)

          pairImages.append([np.asarray(Image.open(image)), np.asarray(Image.open(randomSame))])
          pairLabels.append([1])

          labelChosen = label
          while labelChosen == label:
            labelChosen = np.random.choice(classes) #lol a really bad way to do this
          label = labelChosen
          idxIndex = idLabels.index(label)
          randomDiff = images[np.random.choice(idx[idxIndex])]

          pairImages.append([np.asarray(Image.open(image)), np.asarray(Image.open(randomDiff))])
          pairLabels.append([0])


  return (np.array(pairImages), np.array(pairLabels))

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
    predictions = model.predict(test_data)
    fig2 = plt.figure(figsize=(10,10))
    for b in range(100,164):
        plt.subplot(8,8,(b-100)+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        im1 = Image.fromarray(np.uint8(cm.gist_earth(test_data[b][0])*255))
        im2 = Image.fromarray(np.uint8(cm.gist_earth(test_data[b][1])*255))
        concat_im = Image.new('RGB', (im1.width + im2.width, im1.height))
        concat_im.paste(im1, (0, 0))
        concat_im.paste(im2, (im1.width, 0))

        plt.imshow(concat_im, cmap=plt.cm.binary)
        plt.xlabel(class_names[np.argmax(predictions[b])])

        fig2.savefig('output/prediction.png')
        
genFrames()
#train()
