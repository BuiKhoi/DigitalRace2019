from keras.models import *
from keras.layers import *

def thieft_model(input_size = (240, 320, 3), classes = 2):
    inp = Input(input_size)
    
    cnv1 = Conv2D(16, (3, 3), activation='relu', padding = 'same')(inp)
    btn1 = BatchNormalization()(cnv1)
    btn1 = LeakyReLU()(btn1)
    mxp1 = MaxPool2D((2, 2))(btn1)
    
    cnv2 = Conv2D(32, (3, 3), activation='relu', padding = 'same')(mxp1)
    btn2 = BatchNormalization()(cnv2)
    btn2 = LeakyReLU()(btn2)
    mxp2 = MaxPool2D((2, 2))(btn2)
    
    cnv3 = Conv2D(64, (3, 3), padding = 'same')(mxp2)
    cnv3 = LeakyReLU()(cnv3)
    mxp3 = MaxPooling2D((2, 2))(cnv3)
    
    flat = Flatten()(mxp3)
    drp1 = Dropout(0.5)(flat)
    
    dns1 = Dense(64)(drp1)
    dns1 = LeakyReLU()(dns1)
    dns2 = Dense(64)(dns1)
    dns2 = LeakyReLU()(dns2)
    dns3 = Dense(classes, activation='linear')(dns2)
    
    return Model(inp, dns3)