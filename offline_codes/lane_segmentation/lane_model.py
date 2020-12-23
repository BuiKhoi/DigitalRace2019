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

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

CHECKPOINT_FILE = './checkpoints/fcn_big_01.h5'

IMG_HEIGHT = 144
IMG_WIDTH = 400

OUT_HEIGHT = 72
OUT_WIDTH = 200

def rv_iou(y_true, y_pred):

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  intersection = y_true * y_pred

  not_true = 1 - y_true
  union = y_true + (not_true * y_pred)

  #return intersection.sum() / union.sum()
  return 1 - K.sum(intersection) / K.sum(union)

def construct_model(IMG_HEIGHT, IMG_WIDTH):
    resnet = ResNet18((IMG_HEIGHT, IMG_WIDTH, 3), weights=None, include_top=False)

    output = resnet.layers[-1].output
    output = Flatten()(output)
    restnet = Model(inputs=resnet.input, outputs=output)

#     for layer in restnet.layers:
#         layer.trainable = False

    model = Sequential()
    model.add(resnet)

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Dropout(0.2))

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Dropout(0.2))
    model.add(Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal'))

    return model

def make_fcn_model(IMG_HEIGHT, IMG_WIDTH):
    b = 2
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

    # u9 = Conv2DTranspose(2**b, (2, 2), strides=(2, 2), padding='same') (c8)
    # u9 = concatenate([u9, c1], axis=3)
    # c9 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    # c8 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    # c9 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    o = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=i, outputs=o)
    return model

def get_moblilenet_model():
    from tensorflow.keras.models import model_from_json

    with open('./checkpoints/moblilenet_lane_01.json', 'r') as model_file:
        mini_model = model_from_json(model_file.read())
    return mini_model

class NumpyDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images_link, in_dim, out_dim, batch_size=32, shuffle=False):
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.images_link = images_link
        self.shuffle = shuffle
        self.images = [images_link + f for f in os.listdir(images_link)]
        self.indexes = None
        self.on_epoch_end()
        print('Data generator on {} images'.format(len(self.images)))

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
        y = np.empty((self.batch_size, *self.out_dim, 1), dtype=np.float32)

        for i, img in enumerate(list_images_temp):
            image = np.load(img)['image']
            anno = cv2.resize(image[:, :, -1], (self.out_dim[1], self.out_dim[0]))
            
            X[i,] = cv2.resize(image[:, :, :3], (self.in_dim[1], self.in_dim[0]))
            y[i,] = np.expand_dims(anno, -1)

        return X, y


image_train_link = '/media/buikhoi/01D58E5761595100/Data/training_images/train/'
image_val_link = '/media/buikhoi/01D58E5761595100/Data/training_images/val/'

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
# lanemodel = get_moblilenet_model()
lanemodel.summary()

cbs = [
    ModelCheckpoint(CHECKPOINT_FILE, monitor="val_loss", verbose=1, save_best_only=True),
    ReduceLROnPlateau('val_loss', factor=0.5, patience=3, verbose=1)
]

# lanemodel.load_weights('./checkpoints/fcn_med_01.h5')

lanemodel.compile(loss=rv_iou, optimizer=optimizers.Adam(1e-4))
lanemodel.fit_generator(
    train_gen, 
    steps_per_epoch=train_gen.__len__(), 
    epochs=200, 
    verbose=1, 
    callbacks=cbs,
    validation_data=val_gen
)