import os
import cv2
import time
import numpy as np
from shutil import rmtree
import albumentations as A
from keras.models import model_from_json, load_model

from generate_images import get_files

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.7),
    A.RandomGamma(p=0.7),
    A.CLAHE(p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
])

def millis():
    return int(round(time.time() * 1000))

IMAGE_FOLDER = './images_data/'
TARGET_FOLDER = './segmented_data/'

CLEAR_DATA = False

COURSE_MODEL = './models/models/unet_course.json'
COURSE_MODEL_WEIGHT = './models/weights/unet_course.h5'

GENERATED_FILE = './generated_images.txt'

BATCH_SIZE = 10

def get_all_images(folder):
    return get_files(folder, GENERATED_FILE)

if __name__ == '__main__':
    # get all images
    images = get_all_images(IMAGE_FOLDER)
    
    if CLEAR_DATA:
        # clear destination folder
        if os.path.exists(TARGET_FOLDER):
            rmtree(TARGET_FOLDER)
        os.mkdir(TARGET_FOLDER)

    # load course model
    with open(COURSE_MODEL, 'r') as model_file:
        course_model = model_from_json(model_file.read())
    course_model.load_weights(COURSE_MODEL_WEIGHT)
    # course_model = load_model(COURSE_MODEL_WEIGHT, compile=False)
    course_model.summary()

    print('Processing on {} images'.format(len(images)))

    # make image batch and predict
    idx = 0
    image_batch = []
    image_names = []
    while True:
        print('Processing image {} of {}'.format(idx + 1, len(images)), end='\r')
        if len(image_batch) < BATCH_SIZE and idx < len(images):
            try:
                img = cv2.imread(images[idx])[:, :, (2, 1, 0)]
                image_names.append(images[idx].split('/')[-1])
                image_batch.append(cv2.resize(img, (128, 128)))
                idx += 1
            except IndexError:
                pass
        else:
            predictions = course_model.predict(np.array(image_batch))
            for pred, img, name in zip(predictions, image_batch, image_names):
                new_img = cv2.bitwise_and(img, img, mask=(pred[:, :, 0] * 255).astype('uint8'))
                mil = name.split('x')[0]
                name = name.replace(mil, str(millis()))
                cv2.imwrite(TARGET_FOLDER + name, new_img[:, :, (2, 1, 0)])

                for i in range(2):
                    transformed = transform(image=img)
                    mil = name.split('x')[0]
                    name = name.replace(mil, str(millis()))
                    new_img = cv2.bitwise_and(transformed['image'], transformed['image'], mask=(pred[:, :, 0] * 255).astype('uint8'))
                    cv2.imwrite(TARGET_FOLDER + name, new_img[:, :, (2, 1, 0)])
            image_batch = []
            image_names = []
