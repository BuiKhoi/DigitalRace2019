import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, UpSampling2D, Input, Lambda, Conv2DTranspose, concatenate
# from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
from classification_models.tfkeras import Classifiers

CHECKPOINT_FILE = './checkpoints/fcn_embd_01.h5'

IMG_HEIGHT = 144
IMG_WIDTH = 400

OUT_HEIGHT = 18
OUT_WIDTH = 50

def make_fcn_model(IMG_HEIGHT, IMG_WIDTH):
    b = 4
    i = Input((IMG_HEIGHT, IMG_WIDTH, 3))
    # s = Lambda(lambda x: preprocess_input(x)) (i)
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

    o = Conv2D(4, (1, 1), activation='linear') (c6)

    model = Model(inputs=i, outputs=o)
    return model

class NumpyDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images_link, in_dim, out_dim, batch_size=32, shuffle=False):
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.images_link = images_link
        self.shuffle = shuffle
        self.images = self.get_all_images(images_link)
        self.indexes = None
        self.on_epoch_end()
        print('Data generator on {} images'.format(len(self.images)))

    def get_all_images(self, images_link):
        images = []
        for f in os.listdir(images_link):
            if '_lbl' not in f:
                images.append(images_link + f)
        return images

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.images)//self.batch_size

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
        X = np.empty((self.batch_size, *self.in_dim, 3), dtype=np.float32)
        y = np.empty((self.batch_size, *self.out_dim, 4), dtype=np.float32)

        for i, img in enumerate(list_images_temp):
            
            X[i,] = cv2.resize(np.load(img)['image'], (self.in_dim[1], self.in_dim[0]))
            y[i,] = cv2.resize(np.load(img.replace('.npz', '_lbl.npz'))['image'], (self.out_dim[1], self.out_dim[0]))

        return X, y


image_train_link = './training_images/train/'
image_val_link = './training_images/val/'

train_gen = NumpyDataGenerator(
    image_train_link, 
    (IMG_HEIGHT, IMG_WIDTH), 
    (OUT_HEIGHT, OUT_WIDTH), 
    batch_size=32,
    shuffle=True
)
val_gen = NumpyDataGenerator(
    image_val_link, 
    (IMG_HEIGHT, IMG_WIDTH), 
    (OUT_HEIGHT, OUT_WIDTH), 
    batch_size=32,
    shuffle=False
)

lanemodel = make_fcn_model(IMG_HEIGHT, IMG_WIDTH)
lanemodel.summary()

cbs = [
    ModelCheckpoint(CHECKPOINT_FILE, monitor="val_loss", verbose=1, save_best_only=True),
    ReduceLROnPlateau('val_loss', factor=0.5, patience=3, verbose=1)
]

# lanemodel.load_weights('./checkpoints/linemodel_03.h5')

lanemodel.compile(loss='mse', optimizer=optimizers.Adam(1e-4))
lanemodel.fit_generator(
    train_gen, 
    steps_per_epoch=train_gen.__len__(), 
    epochs=200, 
    verbose=1, 
    callbacks=cbs,
    validation_data=val_gen
)