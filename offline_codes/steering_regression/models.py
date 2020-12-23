from keras.models import *
from keras.layers import *

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

def steer_model_fcn(input_size = (144, 400, 3), classes = 13):
    fcn = make_fcn_model(input_size[0], input_size[1])
    for layer in fcn.layers:
        layer.trainable = False
        
    model = Sequential()
    model.add(fcn)
    model.add(Conv2D(4, 3, activation='relu', padding='same'))
    model.add(Conv2D(4, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    return model

def torch_steer_model(input_shape = (112, 112, 4), classes=13):
    inpt = Input(input_shape)

    cnv1 = Conv2D(16, 3, padding='same')(inpt)
    cnv1 = BatchNormalization()(cnv1)
    cnv1 = LeakyReLU()(cnv1)
    # mxp1 = MaxPool2D((2, 2))(cnv1)

    cnv2 = Conv2D(16, 3, padding='same')(cnv1)
    cnv2 = BatchNormalization()(cnv2)
    cnv2 = LeakyReLU()(cnv2)
    mxp2 = MaxPool2D((2, 2))(cnv2)

    cnv3 = Conv2D(64, 3, padding='same')(mxp2)
    cnv3 = BatchNormalization()(cnv3)
    cnv3 = LeakyReLU()(cnv3)
    # mxp3 = MaxPool2D((2, 2))(cnv3)

    cnv4 = Conv2D(64, 3, padding='same')(cnv3)
    cnv4 = BatchNormalization()(cnv4)
    cnv4 = LeakyReLU()(cnv4)

    cnv5 = Conv2D(16, 3, padding='same')(cnv4)
    cnv5 = BatchNormalization()(cnv5)
    cnv5 = LeakyReLU()(cnv5)
    mxp5 = MaxPool2D((2, 2))(cnv5)

    fltn = Flatten()(mxp5)
    drop = Dropout(0.3)(fltn)

    dns1 = Dense(64)(drop)
    dns1 = LeakyReLU()(dns1)
    dns2 = Dense(64)(dns1)
    dns2 = LeakyReLU()(dns2)
    dns3 = Dense(classes, activation='softmax')(dns2)

    return Model(inpt, dns3)


def thieft_model(input_size = (240, 320, 3), classes = 2):
    inp = Input(input_size)
    
    cnv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
    btn1 = BatchNormalization()(cnv1)
    btn1 = LeakyReLU()(btn1)
    mxp1 = MaxPool2D((2, 2))(btn1)
    
    cnv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(mxp1)
    btn2 = BatchNormalization()(cnv2)
    btn2 = LeakyReLU()(btn2)
    mxp2 = MaxPool2D((2, 2))(btn2)
    
    # cnv3 = Conv2D(64, (3, 3))(mxp2)
    # cnv3 = LeakyReLU()(cnv3)
    # mxp3 = MaxPooling2D((2, 2))(cnv3)
    
    flat = Flatten()(mxp2)
    drp1 = Dropout(0.5)(flat)
    
    dns1 = Dense(64)(drp1)
    dns1 = LeakyReLU()(dns1)
    dns2 = Dense(64)(dns1)
    dns2 = LeakyReLU()(dns2)
    dns3 = Dense(classes, activation='linear')(dns2)
    
    return Model(inp, dns3)