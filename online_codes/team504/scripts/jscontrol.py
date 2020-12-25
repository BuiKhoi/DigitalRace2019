#!/usr/bin/python3
#---Import---#
#---ROS

from primesense import openni2#, nite2
from primesense import _openni2 as c_api

import rospy,sys,os
import rospkg
import os
import gc
import cv2
import json
import time
import math
import zipfile
import shutil
import numpy as np
from threading import Thread
from datetime import datetime
from std_msgs.msg import String, Float32, Bool, Int8
from sensor_msgs.msg import CompressedImage, Image, LaserScan, Joy

path = rospkg.RosPack().get_path('team504')

openni2.initialize(path + '/scripts/modules')
dev = openni2.Device.open_any()
rgb_stream = dev.create_color_stream()
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))
rgb_stream.start()
depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 320, resolutionY = 240, fps = 30))
depth_stream.start()

spd = 0
ste = 0
recording = False
recording_folder = str(datetime.now())
prev_recording_folder = recording_folder

speed_pub = rospy.Publisher('/set_speed', Float32, queue_size=1)
steer_pub = rospy.Publisher('/set_angle', Float32, queue_size=1)

for f in os.listdir(path + '/Images/'):
    try:
        shutil.rmtree(path + '/Images/' + f)
    except:
        os.remove(path + '/Images/' + f)

def millis():
        return int(round(time.time() * 1000))

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def video_thread():
    global recording_folder
    os.mkdir(path + '/Images/' + recording_folder)
    while True:
        global spd
        global ste
        global recording
        mil = millis()
        bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
        rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb,(320, 240))
        rgb = cv2.flip(rgb, 1)
        # cv2.imshow('rgb', rgb)
        # if cv2.waitKey(1):
        #     pass

        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        img = np.frombuffer(frame_data, dtype=np.uint16)
        img.shape = (240, 320)
        img = cv2.flip(img,1)
        img2 = img.copy()
        img2 = (img2*1.0/2**8).astype(np.uint8)
        img2 = 255 - img2
        # print('{}, {}'.format(img2.min(), img2.max()))
        # cv2.imshow('depth', img2)
        # if cv2.waitKey(1):
        #     pass   

        tmp = np.zeros((240, 320, 4))
        tmp [:, :, :3] = rgb.copy()
        tmp [:, :, -1] = img2.copy()
        if recording and int(spd) > 10:
            cv2.imwrite(path + '/Images/' + recording_folder + '/' + str(int(mil)) + 'x' + str(int(spd)) + 'x' + str(int(ste)) + '.png', tmp)
        # print('Speed: {}, steer: {}'.format(spd, ste))

def lidar_process(data):
    img = np.zeros((240, 360), np.uint8)
    idx = 0
    for i in range(360):
        r = data.ranges[i]
        if r <300:
            img[:, idx] = r/12 * 255
        else:
            img[:, idx] = 255
        idx += 1
    # cv2.imshow('lidar', cv2.resize(img, (360, 240)))
    # if cv2.waitKey(1):
    #     pass

def process_joy(data):
    global spd
    global ste
    global recording
    global recording_folder
    global prev_recording_folder
    spd = data.axes[1] * -30
    if spd <= 5 and spd >= -5:
        spd = 0
    elif spd < 15 and spd > 5:
        spd = 15
    elif spd > -15 and spd < -5:
        spd = -15
    ste = data.axes[2] * 60
    speed_pub.publish(spd)
    steer_pub.publish(ste)
    if not recording and data.buttons[0]:
        print('Recording')
    elif recording and not data.buttons[0]:
        print('Stop recording at ' + recording_folder)
        prev_recording_folder = recording_folder
        recording_folder = str(datetime.now())
        os.mkdir(path + '/Images/' + recording_folder)
    recording = data.buttons[0]

    if data.buttons[1]:
        remove_folder()

def remove_folder():
    global recording_folder, recording, prev_recording_folder
    if recording:
        recording = False
        try:
            shutil.rmtree(path + '/Images/' + recording_folder)
            print('Removed folder {}'.format(recording_folder))
        except:
            print('Folder already deleted')
        prev_recording_folder = recording_folder
    else:
        try:
            shutil.rmtree(path + '/Images/' + prev_recording_folder)
            print('Removed folder {}'.format(prev_recording_folder))
        except:
            pass

if __name__ == '__main__':
    rospy.init_node('run', anonymous=True, disable_signals=True)
    rospy.Subscriber("/joy", Joy, process_joy, queue_size=1)
    Thread(target=video_thread).start()
    rospy.spin()
    # Run after code executed
    print('Compressing images')
    # shutil.make_archive(path + '/Images/' + str(datetime.now()), 'zip', path + '/Images/')
    # for f in os.listdir(path + '/Images/'):
    #     if not f.endswith('.zip'):
    #         shutil.rmtree(path + '/Images/' + f)
    dev.close()