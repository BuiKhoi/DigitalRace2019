import os
import gc
import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
import keras
import keras.backend as K
from keras.layers import Conv2D, MaxPool2D, Dropout, UpSampling2D, Input, Lambda, Conv2DTranspose
from keras.models import *
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import *



def rv_iou(y_true, y_pred):

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  intersection = y_true * y_pred

  not_true = 1 - y_true
  union = y_true + (not_true * y_pred)

  #return intersection.sum() / union.sum()
  return 1 - K.sum(intersection) / K.sum(union)

def rv_fbeta(y_true, y_pred):
  pred0 = Lambda(lambda x : x[:,:,:,0])(y_pred)
  pred1 = Lambda(lambda x : x[:,:,:,1])(y_pred)
  pred2 = Lambda(lambda x : x[:,:,:,2])(y_pred)

  true0 = Lambda(lambda x : x[:,:,:,0])(y_true)
  true1 = Lambda(lambda x : x[:,:,:,1])(y_true)
  true2 = Lambda(lambda x : x[:,:,:,2])(y_true)
    
  y_pred_0 = K.flatten(pred0)
  y_true_0 = K.flatten(true0)
    
  y_pred_1 = K.flatten(pred1)
  y_true_1 = K.flatten(true1)

  y_pred_2 = K.flatten(pred2)
  y_true_2 = K.flatten(true2)
    
  intersection0 = K.sum(y_true_0 * y_pred_0)
  intersection1 = K.sum(y_true_1 * y_pred_1)
  intersection2 = K.sum(y_true_2 * y_pred_2)

  precision0 = intersection0/(K.sum(y_pred_0)+K.epsilon())
  recall0 = intersection0/(K.sum(y_true_0)+K.epsilon())
  
  precision1 = intersection1/(K.sum(y_pred_1)+K.epsilon())
  recall1 = intersection1/(K.sum(y_true_1)+K.epsilon())

  precision2 = intersection2/(K.sum(y_pred_2)+K.epsilon())
  recall2 = intersection2/(K.sum(y_true_2)+K.epsilon())
    
  fbeta0 = (1.0+1.0)*(precision0*recall0)/(1.0*precision0+recall0+K.epsilon())
  fbeta1 = (1.0+5.0)*(precision1*recall1)/(5.0*precision1+recall1+K.epsilon())
  fbeta2 = (1.0+5.0)*(precision2*recall2)/(5.0*precision2+recall2+K.epsilon())
    
  return ((fbeta0 + fbeta1 + fbeta2)/3.0)

def rv_dice(y_true, y_pred):
  y_true_flat = K.flatten(y_true)
  y_pred_flat = K.flatten(y_pred)
  intersection = K.sum(y_true_flat * y_pred_flat)

  return 1 - (2. * intersection + 1) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + 1)

def weighted_categorical_cross_entropy(y_true, y_pred):
  weights = K.variable([1.0, 5.0, 5.0, 1.0])
        
  # scale predictions so that the class probas of each sample sum to 1
  y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
  # clip to prevent NaN's and Inf's
  y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
  # calc
  loss = y_true * K.log(y_pred) * weights
  loss = -K.sum(loss, -1)
    
  return loss

def triple_loss(y_true, y_pred):
  return rv_dice(y_true, y_pred) + weighted_categorical_cross_entropy(y_true, y_pred)

# train_generator = UnetDataGenerator('./Training Data/train/', 4, preview=True)
# val_generator = UnetDataGenerator('./Training Data/val/', 4, shuffle=False)

images = np.load('./Training Data/X_data/images.npy')
y_true = np.load('./Training Data/y_true/labels.npy')

y_true[:, :, :, 2] = y_true[:, :, :, 2] + y_true[:, :, :, 3]
y_true[:, :, :, 3] = y_true[:, :, :, 5]
y_true = y_true[:, :, :, 0:4]

X_train, X_test, y_train, y_test = train_test_split(images, y_true)
del(images)
del(y_true)

train_model = prepare_fcn_model((240, 320, 3))

with open('./Saved Models/Model/Tuyen_seg_model.json', 'w') as save_file:
  save_file.write(train_model.to_json())

# train_model.load_weights('./Saved Models/Weights/unet4c-optimized.h5')

reducelr = ReduceLROnPlateau('val_loss', 0.5, 5, verbose = 1)
checkpoint = ModelCheckpoint('./Checkpoints/Tuyen_seg_model.h5', 'val_loss', save_best_only = True, verbose = 1)
train_model.fit(X_train, y_train, 4, epochs=50, callbacks=[reducelr, checkpoint], validation_data=(X_test, y_test))
# train_model.fit_generator(train_generator, train_generator.__len__(), 50, callbacks=[checkpoint, reducelr])