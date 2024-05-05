import os
import cv2
import csv
import numpy as np
import time

from key_frame_detector import *

# path_video = "/home/abel/Desktop/video_toolbox/data/a_live_photo_eg/eg2/IMG_8820.MOV"
path_video = "../../data/a_live_photo_eg/eg1/IMG_4144.MOV"
# path_video = "/home/abel/Downloads/IMG_3903.MOV"
# path_video = "/home/abel/Downloads/selfie_iphone12pro_37frames_30ms.MOV"


dest = "."
keyframeDetection(path_video, dest, Thres=0.2, plotMetrics=True)