#!/usr/bin/python3
#---Import---#
#---ROS
import rospy,sys,os
import rospkg
import os
import gc
import cv2
import json
import time
import math
import numpy as np
from std_msgs.msg import Float32
from threading import Thread
from sensor_msgs.msg import CompressedImage


os.system("rm -rf ~/.nv")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import tensorflow as tf
from keras.models import *
from keras.callbacks import *

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

pkg_path = rospkg.RosPack().get_path('team504')

try:
	os.chdir(os.path.dirname(__file__))	
	os.system('clear')
	print("\nWait for initial setup, please don't connect anything yet...\n")
	sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except: pass

def ros_print(content):
        rospy.loginfo(content)

class ROSControl:
    pubSpeed = None
    pubAngle = None
    subImage = None

    current_speed = 0
    current_angle = 0
    newControl = False

    newImage = False

    model_link = ''
    weight_link = ''

    tm = None

    def refresh_image(self, data):
        '''
        Callback function to refresh the image feed when there is one available
        '''
        try:
            # if self.cControl.fetching_image:
            if True:
                Array_JPG = np.fromstring(data.data, np.uint8)
                cv_image = cv2.imdecode(Array_JPG, cv2.IMREAD_COLOR)
                self.cControl.refresh_image(cv_image)
                self.newImage = True
        except BaseException as be:
            ros_print('{}'.format(be))
            self.Continue = True

    def __init__(self, teamName):
        '''
        ROSPY init function
        '''
        self.subImage = rospy.Subscriber(teamName + '/camera/rgb/compressed', CompressedImage, self.refresh_image)
        self.pubSpeed = rospy.Publisher(teamName + '/set_speed', Float32, queue_size=10)
        self.pubAngle = rospy.Publisher(teamName + '/set_angle', Float32, queue_size=10)
        rospy.init_node('talker', anonymous=True)
        Thread(target=self.drive_thread).start()
        Thread(target=self.publish_thread).start()
        self.tm = TimeMetrics()
        rospy.spin()
        if self.cControl.out_video is not None:
            self.cControl.out_video.release()

    def publish_thread(self):
        while True:
            if self.newControl:
                self.pubSpeed.publish(self.current_speed)
                self.pubAngle.publish(self.current_angle)
                self.newControl = False
            else:
                time.sleep(0.000001)

    def drive_thread(self):
        '''
        Thread for driving the car
        '''
        print('Drive thread online')
        self.cControl = CarControl()
        while True:
            if self.newImage:
                controls = self.cControl.get_next_control()
                self.current_speed = float(controls[0])
                self.current_angle = float(controls[1])
                self.newImage = False
                self.newControl = True
            else:
                time.sleep(0.1)

if __name__ == '__main__':
    # print(sys.version)
    rosControl = ROSControl('team504')
