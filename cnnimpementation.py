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
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization

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
images_arr = images_arr.reshape(-1, 224,224, 1)
images_arr = images_arr / np.max(images_arr)

# print the image f(irst 5)
# for i in range(5):
#   plt.figure(figsize = [5,5])
#   curr_img = np.reshape(images_arr[i], (224,224))
#   plt.imshow(curr_img, cmap='gray')
#   plt.show()

from sklearn.preprocessing import LabelBinarizer
labelBinarizer = LabelBinarizer()
y=labelBinarizer.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(images_arr,y,test_size=0.2,random_state =42,stratify=y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten

cnnModel= Sequential() 
# in this the 64 is filter count, 5,5 is filter size, padding = same will automaticaly pad tp match the image size and activation map size
# activation unction is relu,input shape is a gray scale so the processing 224(width)height and only the intensity
cnnModel.add(Conv2D(64, (5,5) , padding = 'same', activation="relu", input_shape=(224,224,1)))
cnnModel.add(Conv2D(64, (5,5) , padding = 'same', activation="relu"))
cnnModel.add(Conv2D(64, (5,5) , padding = 'same', activation="relu"))
cnnModel.add(MaxPooling2D((2,2)))
cnnModel.add(Conv2D(128, (5,5) , padding = 'same', activation="relu"))
cnnModel.add(Conv2D(128, (5,5) , padding = 'same', activation="relu"))
cnnModel.add(MaxPooling2D((2,2)))
cnnModel.add(Conv2D(128, (5,5) , padding = 'same', activation="relu"))
cnnModel.add(Conv2D(128, (5,5) , padding = 'same', activation="relu"))
cnnModel.add(Conv2D(128, (5,5) , padding = 'same', activation="relu"))
cnnModel.add(MaxPooling2D((2,2)))
cnnModel.add(Flatten())
cnnModel.add(Dense(100,activation="relu"))
cnnModel.add(Dense(200,activation="relu"))
cnnModel.add(Dense(3,activation="softmax"))
cnnModel.summary()

cnnModel.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"])

testLoss, testAccuracy = cnnModel.evaluate(X_test,y_test)
print("Test Accuracy =", testAccuracy)

image = X[10]
image = image.reshape(-1, 224,224, 1)
image = image / np.max(image)

pred = cnnModel.predict(image)
pred = labelBinarizer.inverse_transform(pred)
print(pred)

from tensorflow.keras.models import load_model
model = load_model('cnnmodel.h5')
pred = model.predict(image)
pred = labelBinarizer.inverse_transform(pred)
print(pred,Y[10])
