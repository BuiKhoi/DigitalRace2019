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
import numpy as np
from threading import Thread
from datetime import datetime
from std_msgs.msg import String, Float32, Bool, Int8
from sensor_msgs.msg import CompressedImage, Image, LaserScan

path = rospkg.RosPack().get_path('team504')

openni2.initialize(path + '/scripts/modules')
dev = openni2.Device.open_any()
rgb_stream = dev.create_color_stream()
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))
rgb_stream.start()
depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 320, resolutionY = 240, fps = 30))
depth_stream.start()

dt = str(datetime.now())
out_rgb = cv2.VideoWriter(path + '/scripts/Videos/rgb_' + dt + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (320,240))
out_depth = cv2.VideoWriter(path + '/scripts/Videos/depth_' + dt + '.avi',cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 30, (320,240), 0)

def video_thread():
    while True:
        bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
        rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb,(320, 240))
        rgb = cv2.flip(rgb, 1)
        out_rgb.write(rgb)
        cv2.imshow('rgb', rgb)
        if cv2.waitKey(1):
            pass

        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        img = np.frombuffer(frame_data, dtype=np.uint16)
        img.shape = (240, 320)
        img = cv2.flip(img,1)
        img2 = img.copy()
        img2 = (img2*1.0/2**8).astype(np.uint8)
        # img2 = 255 - img2
        # print('{}, {}'.format(img2.min(), img2.max()))
        out_depth.write(img2)
        cv2.imshow('depth', img2)
        if cv2.waitKey(1):
            pass   

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
    cv2.imshow('lidar', cv2.resize(img, (360, 240)))
    if cv2.waitKey(1):
        pass

if __name__ == '__main__':
    rospy.init_node('run', anonymous=True, disable_signals=True)
    Thread(target=video_thread).start()
    # rospy.Subscriber("/scan", LaserScan, lidar_process, queue_size=1)
    # steer_pub = rospy.Publisher('/set_steer_car_api', Float32, queue_size=1)
    # while True:
    #     steer_pub.publish(float(40))
    #     time.sleep(3)
        
    #     steer_pub.publish(float(-40))
    #     time.sleep(3)
    rospy.spin()
    dev.close()
    out_rgb.release()
    out_depth.release()