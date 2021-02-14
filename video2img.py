import cv2
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('v_path', type=str,
                    help='input scene directory')
args = parser.parse_args()


def video2img(v_path, img_path):
    raw = cv2.VideoCapture(v_path)
    ret = True
    while ret:
        ret,frame = raw.read()
        print(ret)
        print(frame.shape)

video2img('/home/cxu-serve/u1/lchen63/github/nerf-pytorch/data/lele_data/video/IMG_0267.MOV', None)