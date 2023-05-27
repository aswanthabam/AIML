import os
from skimage import data
import numpy as np
from pandas import DataFrame
from scipy import misc
from PIL import Image
import imageio
import keras # for neural network creation
from matplotlib import pyplot as plt
from skimage import io

X=[]
Y=[]
base_path='/content/drive/MyDrive/Workshop/new' # data is available in drive
source_path=base_path
for child in os.listdir(source_path):
  print(child)
  sub_path = os.path.join(source_path, child)
  bsub_path = os.path.join(base_path, child)
  if os.path.isdir(sub_path):
    for data_file in os.listdir(sub_path):
      Qry = Image.open(os.path.join(sub_path, data_file))
      Qry = Qry.convert("RGB")
      Qry = np.array(Qry.resize((224,224))) #resize the image
      Qry = Qry.reshape([224,224,3]) #3 dimentions
      Qry = Qry[:,:,2]
      flist=np.array(Qry)
      X.append(flist)
      Y.append(child)
print(len(X)) #print how many images are available

# convert to floating point numpy object
images_arr = np.asarray(X)  
images_arr = images_arr.astype('float32')
images_arr = images_arr / np.max(images_arr)

# print the image f(irst 5)
for i in range(5):
  plt.figure(figsize = [5,5])
  curr_img = np.reshape(images_arr[i], (224,224))
  plt.imshow(curr_img, cmap='gray')
  plt.show()
  
# Convert the two dimentional image array to one dimentional, for giving in the DNN Input
from skimage.transform import rescale, resize
X=[]
print(images_arr.shape)
for i in range(images_arr.shape[0]):
  img= resize(images_arr[i],(224,224,1),anti_aliasing=True) # change to 3 dimentional, with preserving corner density
  flist=np.array(img.flatten()) # flattenf= to one dimentional values
  X.append(flist) 

  # convert to numpy object
X=np.asarray(X)
X = X.astype('float32')
X.shape

# binarize the label instead of using the entire label name
from sklearn.preprocessing import LabelBinarizer
labelBinarizer = LabelBinarizer()
y=labelBinarizer.fit_transform(Y)
print(y)

# split the test and train network
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# just print the length ppf test and training data
print(len(X_train))
print(len(y_test))

# mport packages for neural network creation
from tensorflow.keras.models import Sequential # Sequential or connection, sequential is automatic connection bw hidden layer and other
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout
import random
import tensorflow as tf
from tensorflow.keras.optimizers import SGD #optimizers

# Set the random seeds for numpy, random and tensor flow
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# initialize the dnn model as a sequential one
dnnModel=Sequential()
# add hidden layers , connections are made automatticaly bcz its sequential
dnnModel.add(Dense(256,activation="relu",input_shape=(50176,))) # activation is the activation function and input_shape is the size of the single image array (flatten)
dnnModel.add(Dense(256,activation="relu"))
dnnModel.add(Dense(128,activation="relu"))
dnnModel.add(Dense(64,activation="relu"))
dnnModel.add(Dropout(0.5)) # remove the least significant 50% neurons
dnnModel.add(Dense(3,activation="softmax")) # softmax function is the activation function
dnnModel.summary()
# deffinition of neural ntwork is complteted

sgd = SGD(learning_rate=0.01) # staucastic gradient decent, 0.1 is the learning rate, highter = faster learning

dnnModel.compile(optimizer='sgd',loss="categorical_crossentropy",metrics=["accuracy"]) 

history = dnnModel.fit(X_train,y_train,epochs=150,batch_size=32,verbose=1,validation_split=0.1) # learn, 150 epoc, 64 batch execution(GPU)

testLoss, testAccuracy = dnnModel.evaluate(X_test,y_test)
print("Test Accuracy =", testAccuracy)

# for saving nnModel.save('location.hz')
# for extracting
  # tensor flow.keras.models import load_model
  # model = load_model('location.hz')
  # model.predict(....)

  
.# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Predict a new unseen case
image = images_arr[10]
image = image.reshape(224,224)
image = image/np.max(image)
flist = np.array(image.flatten())
img = np.asarray(flist)
img = img.astype('float32')
img = img.reshape(-1,50176)

pred = dnnModel.predict(img)
pred = labelBinarizer.inverse_transform(pred)
print(pred,Y[10])

from tensorflow.keras.models import load_model
model = load_model('dnnmodel.h5')
pred = model.predict(img)
pred = labelBinarizer.inverse_transform(pred)
print(pred)

