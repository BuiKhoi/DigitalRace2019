import os
import cv2
import math
import time
import json
import rospkg
import darkdll
import numpy as np
from PIL import Image
import tensorflow as tf
import keras.backend as K
from threading import Thread
from collections import deque
from time_metrics import TimeMetrics
# from trt_interface import TensorRTModel

def preprocess_input(image):
    image = cv2.resize(image, (400, 144))
    mean = image.mean()
    std = image.std()
    return (image - mean)/std

def min_max_scaler(image):
    minn = image.min()
    maxx = image.max()
    return (image - minn) / (maxx - minn)

import sys
sys.path.append('/home/lfb/Desktop/testing/Ultra-Fast-Lane-Detection/')
from lane_detection_inference import LaneInference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.system('clear')
print("\nWait for initial setup, please don't connect anything yet...\n")

def read_config(link):
    with open(link, 'r') as config_read:
        return json.loads(config_read.read())

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.allow_growth = True
# K.set_session(tf.Session(config=config))

import tensorflow as tf
from keras.models import *

class CarControl:

    prev_I = 0
    prev_error = 0
    last_itr = 0
    new_control = False
    out_video = None
    prev_constrain_angle = []

    start_turning = 0
    left_turn_count = 0
    right_turn_count = 0
    stop_count = 0
    bus = False
    stoppp = False
    turning = False
    turned_right = False
    turned_left = False
    no_turn_index = 0
    last_no_turn_spot = -1
    no_turn_signs = deque(maxlen=20)

    image_feed = None
    depth_feed = None
    sequence_feed = None
    fetching_image = True
    image_detected = False
    lane = None

    course = None
    final_speed = 0
    final_angle = 0
    
    ready = False
    force_stop = True
    prev_get_control = -1
    stopped = False
    force_stop_mil = -1
    last_human_spot = -1
    car_start = -1

    def get_run_model(self, model_link = '', weight_link = ''):
        '''
        Grab the model
        '''
        if model_link is not '':
            print('Loading model at: {}'.format(model_link))
            with open(model_link, 'r') as model_read:
                run_model = model_from_json(model_read.read())

            run_model.load_weights(weight_link)
            print('Model weights loaded')
            # run_model._make_predict_function()
            self.graph = tf.get_default_graph()
            # K.clear_session()
            return run_model
        else:
            run_model = load_model(weight_link)
            return run_model

    def parse_config(self):
        self.config_dict = read_config(self.ros_path + '/scripts/Config/car_control.json')
        self.recording = self.config_dict['recording']
        self.base_speed = self.config_dict['base_speed']
        self.constrain_speed = self.config_dict['constrain_speed']
        self.constrain_angle = [-self.config_dict['contrain_angle'], self.config_dict['contrain_angle']]
        self.prev_constrain_angle = self.constrain_angle
        self.kP = self.config_dict['kP']
        self.kI = self.config_dict['kI']
        self.kD = self.config_dict['kD']
        self.patience = self.config_dict['sign_patience']
        print('Speed multipiler: {}'.format(self.config_dict['speed_multipiler']))

    def refresh_image(self, image, depth):
        self.image_feed = image
        self.depth_feed = depth
        # cv2.imshow('rgb', self.image_feed)
        # cv2.imshow('depth', self.depth_feed)
        # if cv2.waitKey(0):
        #     pass
        self.fetching_image = False
        self.image_detected = False
        # print('New image hey hey')

    def cancel_operation(self):
        self.left_turn_count = 0
        self.right_turn_count = 0
        self.prev_error = 0
        self.prev_I = 0
        self.start_turning = 0
        self.turning = False
        self.stop_count = 0
        self.stoppp = False
        self.patience = self.config_dict['sign_patience']
        self.no_turn_index = 0
        self.last_no_turn_spot = -1
        self.no_turn_signs = 0
        self.turned_left = False
        self.turned_right = False
        self.stopped = False
        self.force_stop_mil = -1
        self.last_human_spot = -1
        self.car_start = self.tm.millis()
    
    def __init__(self):
        self.ros_path = rospkg.RosPack().get_path('team504')
        self.parse_config()
        self.tm = TimeMetrics()
        self.last_itr = self.tm.millis()

        # Thread(target=self.lane_thread).start()
        # Thread(target=self.steer_thread).start()

        Thread(target=self.sign_thread).start()

        # Thread(target=self.path_thread).start()
        # Thread(target=self.course_thread).start()

        Thread(target=self.predictive_thread).start()
    
        print('Car control instance initialized with the following parameters: \n   Speed limit: {}\n   Angle limit: {}'.format(self.constrain_speed, self.constrain_angle))

    def lane_thread(self):
        try:
            print('Loading lane model')
            lane_model = self.get_run_model(self.ros_path + self.config_dict['lane_model_link'], self.ros_path + self.config_dict['lane_model_weight'])
            print('Done loading lane model')
            pred_img = np.random.random((1, 144, 400, 3))
            lane_model.predict(pred_img)
            self.ready = True
            print('Lane thread ready')
            while True:
                if not self.fetching_image:
                    if self.image_feed is not None:
                        pred_img = preprocess_input(self.image_feed)
                        pred_img = np.expand_dims(pred_img, 0)
                        prediction = lane_model.predict(pred_img)[0]
                        self.lane = prediction
                        # cv2.imshow('lane', cv2.resize(prediction, (400, 144)))
                        # if cv2.waitKey(1):
                        #     pass
                    else:
                        time.sleep(0.001)
                    self.fetching_image = True
                else:
                    time.sleep(0.001)
        except Exception as e:
            print('Lane thread fucked up', '\n', e)

    def steer_thread(self):
        try:
            print('Steer thread loading')
            steer_model = self.get_run_model(self.ros_path + self.config_dict['steer_model_link'], self.ros_path + self.config_dict['steer_model_weight'])
            pred_img = np.zeros((1, 72, 200, 1))
            steer_model.predict(pred_img)
            print('Steer thread ready')
            while True:
                if self.lane is not None:
                    prediction = steer_model.predict(np.expand_dims(self.lane, 0))[0]
                    
                    self.final_angle = prediction[1] * -1
                    self.final_speed = (self.config_dict['base_speed'] * self.config_dict['speed_multipiler']) - (abs(self.final_angle) * self.config_dict['speed_reduce_ratio'])
                    self.new_control = True
                    self.lane = None

                    # print('Speed: {}, angle: {}'.format(self.final_speed, self.final_angle))

                    # process_time = self.tm.millis() - self.prev_get_control
                    # print('FPS: {}'.format(int(1000/process_time)))
                    # self.prev_get_control = self.tm.millis()
                else:
                    time.sleep(0.001)
        except Exception as e:
            print('Steer thread fucked up', '\n', e)

    def sign_thread(self):
        try:
            #Get sign model 
            configPath = self.ros_path + self.config_dict['sign_config_path']
            weightPath = self.ros_path + self.config_dict['sign_weight_path']
            metaPath = self.ros_path + self.config_dict['sign_meta_path']

            netMain = None
            metaMain = None
            altNames = None

            if netMain is None:
                netMain = darkdll.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
            if metaMain is None:
                metaMain = darkdll.load_meta(metaPath.encode("ascii"))
            if altNames is None:
                try:
                    with open(metaPath) as metaFH:
                        metaContents = metaFH.read()
                        import re
                        match = re.search("names *= *(.*)$", metaContents,
                                        re.IGNORECASE | re.MULTILINE)
                        if match:
                            result = match.group(1)
                        else:
                            result = None
                        try:
                            if os.path.exists(result):
                                with open(result) as namesFH:
                                    namesList = namesFH.read().strip().split("\n")
                                    altNames = [x.strip() for x in namesList]
                        except TypeError:
                            pass
                except Exception:
                    pass

            darknet_image = darkdll.make_image(darkdll.network_width(netMain), darkdll.network_height(netMain), 3)
            last_sign_detecting = self.tm.millis()
            print('Sign thread ready')
            
            while self.image_feed is None:
                time.sleep(0.001)
            while True:
                if not self.image_detected and self.tm.millis() - last_sign_detecting > self.config_dict['sign_detect_interval'] and self.image_feed is not None:
                    # Get predited shit
                    # mil = self.tm.millis()
                    frame_resized = cv2.resize(self.image_feed, (darkdll.network_width(netMain), darkdll.network_height(netMain)))
                    # print(frame_resized.shape)
                    # cv2.imshow('rgb2', self.image_feed)
                    # if cv2.waitKey(1):
                    #     pass
                    darkdll.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
                    detections = darkdll.detect_image(
                        netMain, 
                        metaMain, 
                        darknet_image, 
                        thresh = self.config_dict['detection_threshold'],
                        nms = 0.1
                    )
                    self.image_detected = True
                    # print('Sign predicted')
                    # print('Sign detection took {} millis'.format(self.tm.millis() - mil))
                    detected = self.filter_detections(detections)
                    # continue
                    # self.vote_signs(detected)
                    # if self.stop_count > self.config_dict['stop_threshold']:
                    #     self.stoppp = True
                    # If there are some sign
                    if b'BBDung' in detected: #if this is a stop sign
                        self.patience = self.config_dict['sign_patience']
                        self.stop_count += 1
                        print('Stop sign')
                    # elif b'BBPhai' in detected: #If this is a right sign
                    #     self.patience = self.config_dict['sign_patience']
                    #     if self.config_dict['right_side_course']:
                    #         self.right_turn_count += 1
                    #     else:
                    #         self.left_turn_count += 1
                    #     print('Right sign')
                    # elif b'BBTrai' in detected: #if this is a left sign
                    #     self.patience = self.config_dict['sign_patience']
                    #     if self.config_dict['right_side_course']:
                    #         self.right_turn_count += 1
                    #     else:
                    #         self.left_turn_count += 1
                    #     print('Left sign')
                    # elif b'VatCan' in detected: #if we found the bus
                    #     self.bus = True
                    #     self.last_bus_spot = self.tm.millis()
                    #     print('The bus')
                    # elif b'Human' in detected:
                    #     print('Hooman')
                    #     self.force_stop = True
                    #     if self.last_human_spot == -1:
                    #         self.last_human_spot = self.tm.millis()
                    #     else:
                    #         if (self.tm.millis() - self.last_human_spot) > 2000:
                    #             self.force_stop = False

                    #If no sign found
                    elif self.patience > 0 and (self.right_turn_count + self.left_turn_count + self.stop_count) > 0:
                        self.patience -= 1
                        print('Chill bro')
                        # self.constrain_speed = 15
                        time.sleep(0.01)
                    elif self.stop_count > 0:
                        if self.stop_count > self.config_dict['stop_threshold']:
                            self.stoppp = True
                        else:
                            self.stop_count = 0
                    elif self.right_turn_count + self.left_turn_count > 0 and not self.turning:
                        if max(self.right_turn_count, self.left_turn_count) > self.config_dict['stop_threshold']: 
                            self.turning = True
                            self.start_turning = self.tm.millis()
                            print('Start turning')
                        # else:
                        #     self.right_turn_count = 0
                        #     self.left_turn_count = 0
                    else:
                        self.constrain_speed = self.config_dict['constrain_speed']
                        time.sleep(0.001)
                    
                    last_sign_detecting = self.tm.millis()
                else:
                    time.sleep(0.001)
        except Exception as e:
            print('Sign thread got fucked up: {}'.format(e))

    def filter_detections(self, detections):
        results = []
        # print(detections)
        for detection in detections:

            if detection[0] == b'BBDung':
                bbox = detection[2]
                if bbox[0] > 120:
                    self.stoppp = True
                    if self.force_stop_mil == -1:
                        self.force_stop_mil = self.tm.millis()
                        print('Force stop')
                        self.constrain_angle = [-60, -10]

            bbox = detection[2]
            if bbox[2] >= self.config_dict['bbox_x_min_size'] and bbox[3] >= self.config_dict['bbox_y_min_size']:
                results.append(detection[0])
        return [d[0] for d in detections]

    def vote_signs(self, detected):
        if any([b in detected for b in [b'BBCRP', b'BBCRT']]):
            if (self.tm.millis() - self.last_no_turn_spot) > self.config_dict['sign_delay']:
                self.no_turn_signs = 1
                print('Start counting no turn signs')
            else:
                self.no_turn_signs += 1
            self.last_no_turn_spot = self.tm.millis()
        else:
            if (self.tm.millis() - self.last_no_turn_spot) < self.config_dict['sign_offset']:
                time.sleep(0.001)
            else:
                if self.no_turn_signs > 0:
                    if self.no_turn_signs > self.config_dict['sign_threshold']:
                        self.no_turn_index += 1
                        print('Tick tick')
                    self.no_turn_signs = -1

    def course_thread(self):
        try:
            course_model = self.get_run_model(self.ros_path + self.config_dict['course_model_link'], self.ros_path + self.config_dict['course_model_weight'])
            pred_img = np.zeros((1, 128, 128, 3))
            course_model.predict(pred_img)
            
            self.ready = True
            # while self.image_feed is None:
            #     time.sleep(0.001)
            print('Course thread ready')

            while True:
                if not self.fetching_image:
                    if self.image_feed is not None:
                        pred_img = np.expand_dims(cv2.resize(self.image_feed, (128, 128))/255, 0)
                        prediction = course_model.predict(pred_img)[0].astype(np.uint8)
                        self.course = cv2.bitwise_and(pred_img[0], pred_img[0], mask=prediction)
                        # cv2.imshow('course', self.course)
                        # if cv2.waitKey(1):
                        #     pass
                    self.fetching_image = True
                else:
                    time.sleep(0.001)
        except InterruptedError as e:
            print(e)
            print('Course thread fucked up: {}'.format(e))
            
    def path_thread(self):
        try:
            if self.config_dict['right_side_course']:
                path_model = self.get_run_model(self.ros_path + self.config_dict['path_model_link'], self.ros_path + self.config_dict['path_model_weight_right'])
            else:
                path_model = self.get_run_model(self.ros_path + self.config_dict['path_model_link'], self.ros_path + self.config_dict['path_model_weight_left'])
            pred_img = np.zeros((1, 128, 128, 3))
            path_model.predict(pred_img)
                
            self.ready = True
            print('Path thread ready')

            while True:
                if self.course is not None:
                    # mil = self.tm.millis()
                    prediction = path_model.predict(np.expand_dims(self.course, 0))[0]
                    self.final_speed = self.config_dict['base_speed'] * self.config_dict['speed_multipiler']
                    self.final_angle = -prediction[1]
                    self.new_control = True
                    self.course = None
                    # self.fetching_image = True
                    # print('Path regress took {} millis'.format(self.tm.millis() - mil))

                    # process_time = self.tm.millis() - self.prev_get_control
                    # print('FPS: {}'.format(int(1000/process_time)))
                    # self.prev_get_control = self.tm.millis()
                else:
                    time.sleep(0.001)
        except Exception as e:
            print('Path thread fucked up {}'.format(e))

    def predictive_thread(self):
        try:
            if self.config_dict['right_side_course']:
                predictive_model = self.get_run_model(self.ros_path + self.config_dict['predictive_model_link'], self.ros_path + self.config_dict['predictive_model_weight_right'])
            else:
                predictive_model = self.get_run_model(self.ros_path + self.config_dict['predictive_model_link'], self.ros_path + self.config_dict['predictive_model_weight_left'])
            pred_img = np.zeros((1, 128, 128, 3))
            predictive_model.predict(pred_img)
            
            self.ready = True
            # while self.image_feed is None:
            #     time.sleep(0.001)
            print('Predictive thread ready')

            while True:
                if not self.fetching_image:
                    if self.image_feed is not None:
                        pred_img = np.expand_dims(cv2.resize(self.image_feed, (128, 128))/255, 0)
                        prediction = predictive_model.predict(pred_img)[0]
                        pred_ste = prediction[0]
                        ste = prediction[1]

                        self.final_speed = self.config_dict['base_speed'] - self.config_dict['speed_reduce_ratio'] * abs(pred_ste)
                        self.final_angle = -ste
                        self.new_control = True

                        # print('New control: {}, {}'.format(self.final_speed, self.final_angle))
                    self.fetching_image = True
                else:
                    time.sleep(0.001)
        except InterruptedError as e:
            print(e)
            print('Course thread fucked up: {}'.format(e))

    def calc_pid(self, error, kP, kI, kD):
        '''
        Return the calculated angle base on PID controller
        '''
        if self.last_itr == 0:
            self.last_itr = self.tm.millis()
            return 0
        else:
            itr = self.tm.millis() - self.last_itr
            i_error = error + self.prev_I / itr
            d_error = (error - self.prev_error) / itr

            self.last_itr = self.tm.millis()
            self.prev_I = i_error
            self.prev_error = error
            pid_value = kP * error + kI * i_error + kD * d_error

            # print('Raw pid: {}'.format(pid_value))

            pid_value = np.clip(pid_value, self.constrain_angle[0], self.constrain_angle[1])
            # print('PID: {}'.format(pid_value))
            return pid_value

    def get_next_control(self): 
        '''
        Return [speed, streering angle] of the next control
        '''
        steer_offset = 0
        if self.bus:
            if (self.tm.millis() - self.last_bus_spot) < self.config_dict['obstacle_avoid_delay']:
                if self.turned_left and not self.config_dict['right_side_course']:
                    print('Avoiding the bus on left side')
                    steer_offset = -self.config_dict['obstacle_avoid_value']
                elif self.config_dict['right_side_course'] and not self.turned_right:
                    print('Avoiding the bus on right side')
                    steer_offset = -self.config_dict['obstacle_avoid_value']
            else:
                self.bus = False

        if (self.tm.millis() - self.last_no_turn_spot) < self.config_dict['sign_delay']: #when we got a no turn sign
            if self.no_turn_index == 1:
                if self.config_dict['right_side_course']:
                    new_angle = [-10, 60]
                    print('Departing')
                else:
                    pass

            elif self.no_turn_index == 2:
                if self.config_dict['right_side_course']:
                    new_angle = [0, 60]
                    print('Ok done the bottom')
                else:
                    pass

            elif self.no_turn_index == 3 or self.no_turn_index == 4:
                if self.config_dict['right_side_course']:
                    new_angle = [-10, 10]
                else:
                    pass
            
            else:
                if self.config_dict['right_side_course']:
                    if not self.turned_right:
                        new_angle = [-60, 20]
                    else:
                        new_angle = [-10, 10]
                    print('Finishing')
                else:
                    pass
        else:
            new_angle = [-self.config_dict['contrain_angle'], self.config_dict['contrain_angle']]

        if self.constrain_angle != new_angle:
                self.constrain_angle = new_angle
                print('Changed constrain angle to: {}'.format(self.constrain_angle))

        if self.force_stop_mil != -1:
            if (self.tm.millis() - self.force_stop_mil) > self.config_dict['sign_offset']:
                self.stopped = True
                self.force_stop_mil = -1
                print('Break')
                return [0, 0]

        elif self.force_stop or self.stoppp:
            if not self.stopped:
                self.stopped = True
                print('Breakkk')
                return [0, 0]
            
        elif self.turning:
            if self.tm.millis() - self.start_turning < self.config_dict['turn_delay']:
                if self.left_turn_count > self.right_turn_count:
                    self.turned_left = True
                    return [20, -60]
                elif self.right_turn_count > self.left_turn_count:
                    self.turned_right = True
                    return [20, 60]
            else:
                self.cancel_operation()
                print('Stop turning')
                return None
        elif self.new_control:
            kP = self.kP
            kI = self.kI
            kD = self.kD
            angle = self.calc_pid(self.final_angle, kP, kI, kD) + steer_offset
            speed = np.clip(self.final_speed, 11, self.constrain_speed) # - self.config_dict['speed_reduce_ratio'] * abs(angle)
            # print(angle)
            self.new_control = False
            if True:
                if self.tm.millis() - self.car_start > 220000:
                    speed = 15
            return [speed, angle]
        else:
            return None
