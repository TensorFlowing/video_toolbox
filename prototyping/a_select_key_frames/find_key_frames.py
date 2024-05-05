import os
import cv2
import numpy as np
import peakutils


def video_to_frames(path_video, dir_save_frames):
    cap = cv2.VideoCapture(path_video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    if (cap.isOpened()== False):
        print("Error opening video file")
    os.makedirs(dir_save_frames, exist_ok=True)
    # Read until video is completed
    frames_all = []
    for i in range(num_frames):
        ret, frame = cap.read()
        path_save_frames = os.path.join(dir_save_frames, "frame_{:03d}.jpg".format(i))
        cv2.imwrite(path_save_frames, frame)
        frames_all.append(frame)
    return frames_all

def find_key_frames(frames_all, threshold=0.2):
    num_frames = len(frames_all)
    diff_curr_prev = np.zeros((num_frames,))

    for i in range(num_frames):
        frame = frames_all[i]
        # convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
        if i > 1:
            img_diff = cv2.subtract(gray, gray_prev)
            value_diff = cv2.countNonZero(img_diff)
            diff_curr_prev[i] = value_diff
        gray_prev = gray
    # Find the key frames by locating the local peaks
    baseline = peakutils.baseline(diff_curr_prev, 2) 
    indices_keyframe = peakutils.indexes(diff_curr_prev-baseline, threshold, min_dist=1) 

    return indices_keyframe




if __name__ == "__main__":

    path_video = "../../data/a_live_photo_eg/eg1/IMG_4144.MOV"
    dir_save_frames = "./tmp"
    frames_all = video_to_frames(path_video, dir_save_frames)

    indices_keyframe = find_key_frames(frames_all, threshold=0.2)
    print("indices_keyframe: ", indices_keyframe)
