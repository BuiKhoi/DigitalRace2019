from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, Model

IMG_HEIGHT = 720
IMG_WIDTH = 1280

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))

output = resnet.layers[-1].output
output = Flatten()(output)
restnet = Model(inputs=resnet.input, outputs=output)

for layer in restnet.layers:
    layer.trainable = False

model = Sequential()
model.add(resnet)

model.add(UpSampling2D((2,2)))
model.add(Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))

model.add(UpSampling2D((2,2)))
model.add(Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))

model.add(UpSampling2D((2,2)))
model.add(Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Flatten())

model.summary()

