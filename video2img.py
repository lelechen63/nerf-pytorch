import cv2
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--v_path', type=str, default='/home/cxu-serve/u1/lchen63/github/nerf-pytorch/data/lele_data/video/IMG_0267.MOV',
                    help='input scene directory')
args = parser.parse_args()


def video2img(v_path, img_path):
    raw = cv2.VideoCapture(v_path)
    ret = True
    count = 0
    while ret:
        count += 1
        ret,frame = raw.read()
        if not ret:
            break
        print(ret,count)
        print(frame.shape)

video2img(args.v_path, None)