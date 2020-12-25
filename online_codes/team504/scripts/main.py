#!/usr/bin/python3
#---Import---#
#---ROS

import os
from ros_control import ROSControl

# os.system("rm -rf ~/.nv")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == '__main__':
    # print(sys.version)
    rosControl = ROSControl()