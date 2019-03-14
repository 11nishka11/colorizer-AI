import keras
import tensorflow as tf
import glob
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
from skimage.transform import resize
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, InputLayer,MaxPooling2D, Dropout
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import History 
from keras.preprocessing.image import ImageDataGenerator
import pylab
import os

def get_data_resize_4d(file):

    #image = img_to_array(load_img(file))
    
    image = imread(file)
    image_grey = rgb2gray(image)
    image_resize = resize(image, (240,320,3),mode='constant',anti_aliasing=True)
    #image_array = np.reshape(image_resize,(688,1168,3))
    
    img_shape = image_resize.shape
    #image = np.array(image, dtype=float)
    
    x = rgb2lab(image_resize)[:, :, 0]
    y = rgb2lab(image_resize)[:, :, 1:]
    y /= 128
    x = x.reshape(1,img_shape[0],img_shape[1],1)
    y = y.reshape(1,img_shape[0],img_shape[1],2)
    return x,y,img_shape

def get_data_resize_3d(file):

    #image = img_to_array(load_img(file))
    
    image = imread(file)
    image_grey = rgb2gray(image)
    image_resize = resize(image, (240,320,3),mode='constant',anti_aliasing=True)
    #image_array = np.reshape(image_resize,(688,1168,3))
    
    img_shape = image_resize.shape
    #image = np.array(image, dtype=float)
    
    x = rgb2lab(image_resize)[:, :, 0]
    y = rgb2lab(image_resize)[:, :, 1:]
    y /= 128
    x = x.reshape(img_shape[0],img_shape[1],1)
    y = y.reshape(img_shape[0],img_shape[1],2)
    return x,y,img_shape
def get_data(file):

    #image = img_to_array(load_img(file))
    
    image = imread(file)
    image_grey = rgb2gray(image)
    img_shape = image.shape
    #image = np.array(image, dtype=float)
    
    x = rgb2lab(image)[:, :, 0]
    y = rgb2lab(image)[:, :, 1:]
    y /= 128
    x = x.reshape(1,img_shape[0],img_shape[1],1)
    y = y.reshape(1,img_shape[0],img_shape[1],2)
    return x,y,img_shape

path = r'C:\Users\mini\Desktop\Rutgers\2018Fall\520 Introduction to Artificial Intelligence\HW4\sunflower'                
all_files = glob.glob(os.path.join(path, "*.jpg"))  
image_X=np.stack((get_data_resize_3d(i)[0] for i in all_files),axis=0)
image_Y=np.stack((get_data_resize_3d(i)[1] for i in all_files),axis=0)



def build_model():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same',input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=4))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae', 'acc'])  
    #model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['mae', 'acc'])  
    return model
  
def build_model_simple():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same',input_shape=(240, 320, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae', 'acc'])
    return model
  
train_model = build_model()

model_file = 'model.h5'
i=150
his=train_model.fit(image_X, image_Y, batch_size=16, epochs=i, validation_split = 0.2)
train_model.save(model_file)
train_model.summary()
train_loss= his.history['loss']
val_loss   = his.history['val_loss']
train_acc = his.history['acc']
val_acc    = his.history['val_acc']
xepochs  = range(i)

pylab.plot(xepochs, train_loss, '-b', label='train_loss')
pylab.plot(xepochs, val_loss, '-r', label='val_loss')
pylab.xlabel('epochs')
pylab.ylabel('loss')
pylab.legend(loc='upper right')
pylab.show()

pylab.plot(xepochs, train_acc, '-b', label='train_acc')
pylab.plot(xepochs, val_acc, '-r', label='val_acc')
pylab.xlabel('epochs')
pylab.ylabel('acc')
pylab.legend(loc='lower right')
pylab.show() 

  
#change color to grey
imsave("grey.jpg", rgb2gray(imread('color.jpg')))   
x, y, shape = get_data_resize_4d('grey.jpg')   #gray
test_model = build_model()
test_model.load_weights('model.h5')
    
output = test_model.predict(x)
output *= 128
out = np.zeros((shape[0], shape[1], 3))
out[:, :, 0] = x[0,:, :, 0]     #LAB not RGB
out[:, :, (1,2)] = output[0]
imsave("result.jpg", lab2rgb(out))
imsave("result_gray.jpg", rgb2gray(lab2rgb(out)))
  