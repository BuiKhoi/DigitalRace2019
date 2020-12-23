import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import tensorflow
import keras
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from models import thieft_model

LEARNING_RATE = 1e-4
EPOCHS = 2000

TRAIN_FOLDER = './train_data/train/'
VAL_FOLDER = './train_data/val/'

LOAD_PRETRAINED = True
PRETRAINED_PATH = './checkpoints/weights/steer_direct_1412.2.h5'

WEIGHT_SAVE_PATH = './checkpoints/weights/steer_direct_1412.h5'
MODEL_SAVE_PATH = './checkpoints/models/steer_direct_1412.json'

INPUT_SHAPE = (128, 128)

class TrainingDataGenerator(keras.utils.Sequence):
  def __init__(self, images_folder, classes=13, batch_size=64, dim=(112, 112), shuffle=False):
    self.dim = dim
    self.batch_size = batch_size
    self.images_folder = images_folder
    self.shuffle = shuffle
    self.classes = classes

    self.images = [self.images_folder + f for f in os.listdir(self.images_folder)]
    print('Image generator on {} images'.format(len(self.images)))

    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return len(self.images) // self.batch_size

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
    y = np.empty((self.batch_size, 2), dtype=np.float32)

    for i, image in enumerate(list_images_temp):
      X[i,] = cv2.imread(image)[:, :, (2, 1, 0)] / 255
      _, spd, ste = image.split('/')[-1].split('.')[0].split('x')
      y[i,] = np.array([int(spd), int(ste)])

    return X, y

  def to_onehot(self, array, classes):
    one_hot = np.zeros((len(array), classes), dtype=np.uint8)
    for i, a in enumerate(array):
        one_hot[i][a] = 1
    
    return one_hot

if __name__ == '__main__':
    train_model = thieft_model((*INPUT_SHAPE, 3))
    train_model.summary()
    train_model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    train_gen = TrainingDataGenerator(TRAIN_FOLDER, dim=INPUT_SHAPE, shuffle=True, batch_size=128)
    val_gen = TrainingDataGenerator(VAL_FOLDER, dim=INPUT_SHAPE, batch_size=128)

    with open(MODEL_SAVE_PATH, 'w') as model_file:
      model_file.write(train_model.to_json())

    if LOAD_PRETRAINED:
      train_model.load_weights(PRETRAINED_PATH)
      print('Loaded pretrained')

    reducelr = ReduceLROnPlateau('val_loss', 0.5, verbose=1, patience = 5)
    checkpoint = ModelCheckpoint(WEIGHT_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(patience = 50, verbose=1)

    history = train_model.fit_generator(
        train_gen,
        train_gen.__len__(),
        EPOCHS,
        callbacks=[reducelr, checkpoint],
        validation_data=val_gen
    )

    plt.plot(history.history['val_loss'])
    plt.show()