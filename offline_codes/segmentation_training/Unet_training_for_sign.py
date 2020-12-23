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

class UnetDataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, images_link, batch_size=32, dim=(240, 320), shuffle=True, preview=False):
    '''
    The images will come with shape (320, 240, 7) with channel 0-2 is the image and 3-6 is the label

    Image link structure:
    <images_link>
    |
    |-- files.npz
    '''

    self.dim = dim
    self.batch_size = batch_size
    self.images_link = images_link
    self.shuffle = shuffle
    self.preview = preview

    self.images = os.listdir(self.images_link)

    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.images) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    #print('Index: {}'.format(index))
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    list_images_temp = [self.images[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_images_temp)

    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.images))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_images_temp):
    '''
    Generate data with the specified batch size
    '''
    # Initialization
    X = np.empty((self.batch_size, *self.dim, 3), dtype=np.uint8)
    y = np.empty((self.batch_size, *self.dim, 4), dtype=np.int8)

    for i, image in enumerate(list_images_temp):
      temp = np.load(self.images_link + image)

      X[i,] = temp[:, :, 0:3]
      y[i,] = temp[:, :, 3:7]

    return X, y

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

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def triple_loss(y_true, y_pred):
  return rv_dice(y_true, y_pred) + weighted_categorical_cross_entropy(y_true, y_pred)

def prepare_model(input_size = (256, 256, 3)):
  inputs = Input(input_size)
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
  pool1 = MaxPool2D((2, 2))(conv1)

  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
  pool2 = MaxPool2D((2, 2))(conv2)

  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
  pool3 = MaxPool2D((2, 2))(conv3)

  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPool2D((2, 2))(drop4)

  # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
  # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
  # drop5 = Dropout(0.5)(conv5)

  # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D((2,2))(drop5))
  # merge6 = concatenate([drop4, up6], axis=3)
  # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
  # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

  up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D((2, 2))(drop4))
  merge7 = concatenate([conv3, up7], axis=3)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
  
  up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D((2, 2))(conv7))
  merge8 = concatenate([conv2, up8], axis=3)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

  up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D((2, 2))(conv8))
  merge9 = concatenate([conv1, up9], axis=3)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
  conv9 = Conv2D(6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
  conv10 = Conv2D(4, 1, activation='sigmoid')(conv9)

  model = Model(inputs=inputs, outputs=conv10)
  model.compile(optimizer=Adam(lr=1e-4), loss=rv_iou)
  model.summary()
  return model

def prepare_fcn_model(input_size = (256, 256, 3)):
  b = 4
  i = Input(input_size)
  # s = Lambda(lambda x: preprocess_input(x)) (i)
  n_classes=4
  c1 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (i)
  c1 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
  c1 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
  p1 = MaxPool2D((2, 2)) (c1)

  c2 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
  c2 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
  c2 = Dropout(0.1) (c2)
  c2 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
  p2 = MaxPool2D((2, 2)) (c2)

  c3 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
  c3 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
  c3 = Dropout(0.2) (c3)
  c3 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)

  p3 = MaxPool2D((2, 2)) (c3)

  c4 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
  c4 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
  c4 = Dropout(0.2) (c4)
  c4 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)

  p4 = MaxPool2D(pool_size=(2, 2)) (c4)

  c5 = Conv2D(2**(b+4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
  c5 = Conv2D(2**(b+4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
  c5 = Dropout(0.3) (c5)
  c5 = Conv2D(2**(b+4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
  c5 = Conv2D(2**(b+4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

  u6 = Conv2DTranspose(2**(b+3), (2, 2), strides=(2, 2), padding='same') (c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
  c6 = Dropout(0.2) (c6)
  c6 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
  c6 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

  u7 = Conv2DTranspose(2**(b+2), (2, 2), strides=(2, 2), padding='same') (c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
  c7 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
  c7 = Dropout(0.2) (c7)
  c7 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

  u8 = Conv2DTranspose(2**(b+1), (2, 2), strides=(2, 2), padding='same') (c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
  c8 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
  c8 = Dropout(0.1) (c8)
  c8 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

  u9 = Conv2DTranspose(2**b, (2, 2), strides=(2, 2), padding='same') (c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
  c8 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
  c9 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

  o = Conv2D(n_classes, (1, 1), activation='sigmoid') (c9)

  model = Model(inputs=i, outputs=o)
  # model.compile(optimizer=Adam(lr=1e-4), loss=weighted_categorical_crossentropy([0.2, 0.3, 0.3, 0.2]))
  model.compile(optimizer=Adam(lr=1e-4), loss=rv_iou)
  model.summary()
  return model

# train_generator = UnetDataGenerator('./Training Data/train/', 4, preview=True)
# val_generator = UnetDataGenerator('./Training Data/val/', 4, shuffle=False)

images = np.load('./Training Data/X_data/images.npy')
y_true = np.load('./Training Data/y_true/labels.npy')

y_true[:, :, :, 2] = y_true[:, :, :, 2] + y_true[:, :, :, 3]
y_true[:, :, :, 3] = y_true[:, :, :, 5]
y_true = y_true[:, :, :, 0:4]

# take = []
# count = 0
# for y in y_true:
#     print(count, end='\r')
#     # cv2.imshow('car', y[:, :, 1].astype('float'))
#     # if cv2.waitKey(1):
#     #     pass
#     if np.sum(y[:, :, 1]) > 50:
#         cv2.imshow('car', y[:, :, 1].astype('float'))
#         if cv2.waitKey(1):
#             pass
#         take.append(count)
    
#     elif np.sum(y[:, :, 2]) > 50:
#         cv2.imshow('sign', y[:, :, 2].astype('float'))
#         if cv2.waitKey(1):
#             pass
#         take.append(count)

#     count += 1
# print('Take {} images'.format(len(take)))
# take = np.array(take)
# images = images[take]
# y_true = y_true[take]

X_train, X_test, y_train, y_test = train_test_split(images, y_true)
del(images)
del(y_true)

train_model = prepare_fcn_model((240, 320, 3))

with open('./Saved Models/Model/Tuyen_seg_model.json', 'w') as save_file:
  save_file.write(train_model.to_json())

train_model.load_weights('./Checkpoints/Tuyen_seg_model_addition.h5')

reducelr = ReduceLROnPlateau('val_loss', 0.5, 12, verbose = 1)
checkpoint = ModelCheckpoint('./Checkpoints/Tuyen_seg_model_addition.h5', 'val_loss', save_best_only = True, verbose = 1)
train_model.fit(X_train, y_train, 4, epochs=250, callbacks=[reducelr, checkpoint], validation_data=(X_test, y_test))
# train_model.fit_generator(train_generator, train_generator.__len__(), 50, callbacks=[checkpoint, reducelr])