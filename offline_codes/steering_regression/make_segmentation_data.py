import os
import cv2
from models import *
import numpy as np
from shutil import rmtree
from keras.models import model_from_json

SEGMENT_MODEL = './models/fcn_big_01.h5'
SAVE_FOLDER = './segmented_images/'

def preprocess_images(image):
    image = cv2.resize(image, (400, 144))
    meann = image.mean()
    std = image.std()
    return (image - meann) / std

def get_all_images(image_folder):
    images = []
    for f in os.listdir(image_folder):
        if os.path.isdir(image_folder + f):
            images.extend(get_all_images(image_folder + f + '/'))
        else:
            if any([e in f for e in ['.png', '.jpg']]):
                images.append(image_folder + f)
    return images

def make_segmentation_data(image_folder = None):
    # with open('./models/fcn_mini.json', 'r') as model_file:
    #     model = model_from_json(model_file.read())
    model = make_fcn_model(144, 400)
    model.load_weights(SEGMENT_MODEL)
    model.summary()

    #clean target folder
    if os.path.exists(SAVE_FOLDER):
        rmtree(SAVE_FOLDER)
    os.mkdir(SAVE_FOLDER)

    #load images
    if image_folder is None:
        image_folder = './image_data/'

    images = get_all_images(image_folder)

    for idx, image in enumerate(images):
        print('Processing image {} of {}'.format(idx, len(images)), end='\r')
        file_name = image.split('/')[-1]
        image = cv2.imread(image)[:, :, (2, 1, 0)]
        pred = model.predict(np.expand_dims(preprocess_images(image), 0))[0, :, :, 0]
        cv2.imwrite(SAVE_FOLDER + file_name, pred * 255)
        # cv2.imshow('lane', cv2.resize(pred, (400, 114)))
        # if cv2.waitKey(5) & 0xFF == ord('q'):
        #     break

if __name__ == '__main__':
    make_segmentation_data()