import os
import cv2
import json
import time
import numpy as np
import keras
import matplotlib.pyplot as plt
# from Models.Unet import *
from Models.refinenet.refinennet_model import *
# from Models.FCN_Resnet import *
# from Models.icnet import *
# from Models.enet.model import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import Adam
# from keras.layers import *
from Losses import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt

class NumpyDataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, images_link, batch_size=32, dim=(240, 320), shuffle=False):
    self.dim = dim
    self.batch_size = batch_size
    self.images_link = images_link
    self.shuffle = shuffle

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
    # X, y = self.__data_generation(list_images_temp)
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
    X = np.empty((self.batch_size, *self.dim, 3), dtype=np.float32)
    y = np.empty((self.batch_size, *self.dim, 2), dtype=np.float16)

    for i, image in enumerate(list_images_temp):
      temp = np.load(self.images_link + image)

      # X[i,] = cv2.resize(temp['img'].astype(np.float32), (256, 256))
      # y[i,] = cv2.resize(temp['anno'].astype(np.float32), (256, 256))
      temp_anno = temp['anno']
      temp_anno[:, :, 1] = np.ones(self.dim) - temp_anno[:, :, 0]
      X[i,] = temp['img']
      y[i,] = temp_anno[:, :, :2]

    return X, y

datagen = NumpyDataGenerator('./Data/CVTN/Training_data/train/', 8, (240, 320), True)
val_gen = NumpyDataGenerator('./Data/CVTN/Training_data/val/', 4, (240, 320))

# train_model = fcn_model((240, 320, 3), 6)
# train_model = enet_build(6, 320, 240)
# with open('./Checkpoints/fcn25.json', 'r') as segnet_json:
#   train_model = model_from_json(segnet_json.read())

model = Refinenet(n_classes= 2)
train_model = model.chose_model(0)[0]

train_model.summary()

# train_model.compile(optimizer=Adam(lr=1e-4), loss=double_loss([1, 1]))
train_model.compile(optimizer=Adam(lr=1e-4), loss=rv_iou)
reducelr = ReduceLROnPlateau('val_loss', 0.5, 5, verbose = 1)
checkpoint = ModelCheckpoint('./Checkpoints/cvtn_refine.h5', 'val_loss', save_best_only = True, verbose = 1)

with open('./Checkpoints/cvtn_refine.json', 'w') as model_file:
  model_file.write(train_model.to_json())

train_model.load_weights('./Checkpoints/cvtn_refine.h5')
# print('Loaded pretrained and start training')
history = train_model.fit_generator(datagen, int(datagen.__len__()/3), 50, callbacks = [reducelr, checkpoint], validation_data=val_gen)
plt.plot(history.history['val_loss'])
plt.show()