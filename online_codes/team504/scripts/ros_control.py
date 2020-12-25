import cv2
import time
import json
import rospy,rospkg
import numpy as np
from threading import Thread
from primesense import openni2#, nite2
from car_control import CarControl
from time_metrics import TimeMetrics
from primesense import _openni2 as c_api
from std_msgs.msg import String, Float32, Bool, Int8
from sensor_msgs.msg import CompressedImage, LaserScan, Joy
from collections import deque

def read_config(link):
    with open(link, 'r') as config_read:
        return json.loads(config_read.read())

class ROSControl:
    ros_path = None
    config_dict = None

    prev_btn = [False] * 4
    curr_speed = 0
    curr_angle = 0
    recording = False
    newLcd = False
    cControl = None
    litsen_flag = False
    pub_flag = True
    last_joy_process = 0
    sensor_on = False

    led_pub = None

    def __init__(self):
        rospy.init_node('run', anonymous=True, disable_signals=True)
        self.ros_path = rospkg.RosPack().get_path('team504')
        self.led_pub = rospy.Publisher('/led_status', Bool, queue_size=1)
        self.config_dict = read_config(self.ros_path + '/scripts/Config/ros_control.json')
        self.cControl = CarControl()
        self.newLcd = True
        while not self.cControl.ready:
            time.sleep(0.0001)

        for i in range(3):
            self.set_led(Bool(True))
            time.sleep(0.2)
            self.set_led(Bool(False))
            time.sleep(0.1)
        
        self.init_sub()
        self.read_cc_config()
        self.newLcd = True
        self.control_override = False
        self.tm = TimeMetrics()

        Thread(target=self.publish_thread).start()
        Thread(target=self.image_thread).start()

        rospy.spin()
        if self.recording:
            self.cControl.out_video.release()

        speed_pub = rospy.Publisher('/set_speed', Float32, queue_size=1)
        steer_pub = rospy.Publisher('/set_angle', Float32, queue_size=1)
        steer_pub.publish(0)
        speed_pub.publish(0)

    def read_cc_config(self):
        if self.cControl.recording:
            self.recording = True

    def process_bt1_status(self, data):
        if data.data:
            print('Litsening for flag')
            self.litsen_flag = True
            self.pub_flag = False
            self.cControl.force_stop = True
        else:
            pass

    def process_bt2_status(self, data):
        if data.data:
            print('Parsing config again')
            self.cControl.parse_config()
        else:
            pass

    def process_bt3_status(self, data):
        if data.data:
            pass
        else:
            pass

    def process_bt4_status(self, data):
        if data.data:
            pass
        else:
            pass

    def process_ss(self, data):
        self.cControl.flag = True
        # if data.data:
        #     self.sensor_on = True
        # else:
        #     self.sensor_on = False
        if self.litsen_flag:
            if not data.data:
                self.litsen_flag = False
        else:
            if data.data and not self.pub_flag:
                self.cControl.sensor_onflag = False
                self.pub_flag = True
                self.cControl.cancel_operation()
                self.cControl.force_stop = False
                print('Go go go')

        self.set_led(not data.data)

    def process_joy(self, data):
        if not self.control_override:
            if self.tm.millis() - self.last_joy_process < 200:
                return
            else:
                self.last_joy_process = self.tm.millis()

        self.spd = data.axes[1] * -30
        if self.spd <= 5 and self.spd >= -5:
            self.spd = 0
        elif self.spd < 15 and self.spd > 5:
            self.spd = 15
        elif self.spd > -15 and self.spd < -5:
            self.spd = -15
        
        self.ste = data.axes[2] * 60

        if data.buttons[0] and not self.control_override:
            self.control_override = True
            print('Overriding control')
        elif self.control_override and not data.buttons[0]:
            self.control_override = False
            self.process_bt2_status(Bool(True))
            print('Stop overriding')

        if data.buttons[2]:
            self.process_ss(Bool(True))
        else:
            self.process_ss(Bool(False))
            # pass

        if data.buttons[1]:
            self.process_bt1_status(Bool(True))
        else:
            # self.process_bt1_status(Bool(False))
            pass

    def init_sub(self):
        rospy.Subscriber("/bt1_status", Bool, self.process_bt1_status, queue_size=1)
        rospy.Subscriber("/bt3_status", Bool, self.process_bt2_status, queue_size=1)
        # rospy.Subscriber("/bt2_status", Bool, self.process_bt3_status, queue_size=1)
        # rospy.Subscriber("/bt4_status", Bool, self.process_bt4_status, queue_size=1)
        rospy.Subscriber("/ss1_status", Bool, self.process_ss, queue_size=1)

        if self.config_dict['remote_control']:
            rospy.Subscriber("/joy", Joy, self.process_joy, queue_size=1)

        pass

    def set_led(self, led_stt):
        self.led_pub.publish(led_stt)
    
    def publish_thread(self):
        speed_pub = rospy.Publisher('/set_speed', Float32, queue_size=1)
        steer_pub = rospy.Publisher('/set_angle', Float32, queue_size=1)
        # lcd_pub = rospy.Publisher('/lcd_print', String, queue_size=1)
        prev_speed = 0
        while True:
            # if self.sensor_on:
            #     speed_pub.publish(0)
            #     steer_pub.publish(0)
            #     print('Obstacle')
            if self.control_override:
                speed_pub.publish(self.spd)
                steer_pub.publish(self.ste)
            elif self.cControl.new_control:
                control = self.cControl.get_next_control()
                # print('Returning speed: {}, angle: {}'.format(control[0], control[1]))
                if control is not None:
                    if control[0] != prev_speed:
                        speed_pub.publish(control[0])
                        prev_speed = control[0]
                    steer_pub.publish(-control[1])
                    self.curr_angle = control[1]
                    self.curr_speed = control[0]
                    self.newLcd = True
                    print('New publish')
                else:
                    time.sleep(0.01)
            # elif self.newLcd:
            #     if not self.cControl.ready:
            #         texts = ["00:0:Starting", "00:1:Pls wait", "00:2:Patient"]
            #         for c in range(20):
            #             for i in range(len(texts)):
            #                 text = texts[i]
            #                 space = (14 - len(text))*" "
            #                 text +=space
            #                 lcd_pub.publish(text)
            #     else:
            #         texts = ["00:0:Spd=", "00:1:Ang=", "00:2:Rec=", "10:0:MaxSp=", "10:1:MaxAng="]
            #         info = [str(self.curr_speed), str(self.curr_angle), str(self.recording), str(self.max_spd), str(self.max_ang)]
            #         for i in range(len(texts)):
            #             text = texts[i] + info[i][:6]
            #             space = (14 - len(text))*" "
            #             text +=space
            #             lcd_pub.publish(text)
            #             time.sleep(0.0001)
            #     self.newLcd = False
            else:
                time.sleep(0.001)

    def image_thread(self):
        openni2.initialize(self.ros_path + '/scripts/modules')
        dev = openni2.Device.open_any()
        rgb_stream = dev.create_color_stream()
        rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))
        rgb_stream.start()
        depth_stream = dev.create_depth_stream()
        depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 320, resolutionY = 240, fps = 30))
        depth_stream.start()
        rgbs = deque(maxlen=25)
        depths = deque(maxlen=25)
        last_camera_read = self.tm.millis()
        while True:
            if self.tm.millis() - last_camera_read >= 10:
                rgb  = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
                # rgb  = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb,(320, 240))
                rgb = cv2.flip(rgb, 1)
                rgbs.append(rgb)

                frame = depth_stream.read_frame()
                frame_data = frame.get_buffer_as_uint16()
                img = np.frombuffer(frame_data, dtype=np.uint16)
                img.shape = (240, 320)
                img = cv2.flip(img,1)
                img2 = img.copy()
                depth = (img2*1.0/2**8).astype(np.uint8)
                depths.append(depth)

                last_camera_read = self.tm.millis()

            if self.cControl.fetching_image:
                
                # cv2.imshow('depth', depth)
                # if cv2.waitKey(1):
                #     pass

                # rgb = cv2.imread(self.ros_path + '/scripts/Pictures/1579508788883.jpg')
                # rgb  = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
                # if len(rgbs) < 5:
                #     self.cControl.refresh_sequence(None, None)
                # else:
                #     self.cControl.refresh_sequence(self.get_image_sequence(rgbs, depths), rgbs[-1])
                if len(rgbs) > 0:
                    self.cControl.refresh_image(rgbs[-1], None)
                else:
                    self.cControl.refresh_image(None, None)
                # print('Refreshing images')
            else:
                time.sleep(0.0001)

    def get_image_sequence(self, rgbs, depths):
        seq = np.empty((1, 240, 320, 20))
        counter = 0
        idx = -1
        for i in range(5):
            try:
                temp = rgbs[idx]
            except IndexError:
                idx += 5
            # print(idx)
            seq[:, :, :, counter:counter+3] = rgbs[idx]
            counter += 3
            seq[:, :, :, counter] = depths[idx]
            counter += 1
            idx -= 5
        return seq