import os
import cv2
import csv
import numpy as np
import time
import peakutils
from utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics

import matplotlib.pyplot as plt


def keyframeDetection(source, dest, Thres, plotMetrics=False, verbose=False):
    
    keyframePath = dest+'/keyFrames'
    imageGridsPath = dest+'/imageGrids'
    csvPath = dest+'/csvFile'
    path2file = csvPath + '/output.csv'
    prepare_dirs(keyframePath, imageGridsPath, csvPath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("length: ", length)
  
    if (cap.isOpened()== False):
        print("Error opening video file")

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None
    Start_time = time.process_time()
    
    # Read until video is completed
    for i in range(length):
        ret, frame = cap.read()

        # import pdb; pdb.set_trace(); # debug

        grayframe, blur_gray = convert_frame_to_grayscale(frame)
        # grayframe.shape: (1920, 1440), same as blur_gray

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)

        # # 
        # import imageio
        # imageio.imwrite(f"diff_{i}.jpg", diff)

        diffMag = cv2.countNonZero(diff)
        print("diffMag: ", diffMag)

        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time-Start_time
        timeSpans.append(time_Span)
        lastFrame = blur_gray

    # import pdb; pdb.set_trace(); # debug

    cap.release()
    y = np.array(lstdiffMag) # (80,)


    base = peakutils.baseline(y, 2) # (80,)

    # plt.figure()
    # plt.plot(y, 'r')
    # plt.plot(base, 'g')
    # plt.plot(y - base, 'b')

    # plt.show()
    # plt.close()

    indices = peakutils.indexes(y-base, Thres, min_dist=1) # (18,)
    # array([ 3,  6,  8, 11, 13, 15, 17, 20, 22, 24, 26, 37, 39, 64, 66, 68, 70, 73])
    print("indices: ", indices)
    print("len(indices): ", len(indices))

    ##plot to monitor the selected keyframe
    if (plotMetrics):
        plot_metrics(y, indices, lstfrm, lstdiffMag)

    cnt = 1
    for x in indices:
        cv2.imwrite(os.path.join(keyframePath , 'keyframe'+ str(cnt) +'.jpg'), full_color[x])
        cnt +=1
        log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
        if(verbose):
            print(log_message)
        with open(path2file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(log_message)
            csvFile.close()

    cv2.destroyAllWindows()