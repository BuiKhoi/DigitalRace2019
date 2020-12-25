from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
from . import darkdll
from threading import Thread
netMain = None
metaMain = None
altNames = None

def YOLO():
    global metaMain, netMain, altNames
    configPath ='./New/yolov4-tiny.cfg'
    weightPath = './New/yolov4-tiny_best.weights'
    metaPath = './New/yolo.data'
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
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
    # cap = cv2.VideoCapture(0)
    # video_path = 'D:/DATA_SET/Cap_roi/videos/VID_20200205_160720.mp4'
    # is_camera = 'rtsp://' in video_path
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"D:\Downloads\CDS\LHU2.avi")
    cap.set(3, 1280)
    cap.set(4, 720)
    # out = cv2.VideoWriter(
    #     "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #     (darknet.network_width(netMain), darknet.network_height(netMain)))
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darkdll.make_image(darkdll.network_width(netMain),
                                       darkdll.network_height(netMain), 3)

    fps = cap.get(cv2.CAP_PROP_FPS)
    pos_slider_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            prev_time = time.time()
            frame_ = frame.copy()
            frame_rgb = np.rot90(frame_, 4)
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darkdll.network_width(netMain),
                                        darkdll.network_height(netMain)),
                                       interpolation = cv2.INTER_LINEAR)

            darkdll.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            detections = darkdll.detect_image(netMain, metaMain, darknet_image, thresh = 0.25)

            shape_img = frame.shape
            data, abc = darkdll.cvDrawBoxesssss(detections, frame_rgb, shape_img, darkdll.network_height(netMain))
            image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            cv2.namedWindow("Demo", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Demo', image)
            cv2.namedWindow("Demo1", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Demo1', frame_)
            # print(1/(time.time()-prev_time))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    YOLO()
