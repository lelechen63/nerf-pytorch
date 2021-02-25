import cv2
import sys

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--v_path', type=str, default='/home/cxu-serve/u1/lchen63/github/nerf-pytorch/data/lele_data3/video/IMG_0298.MOV',
                    help='input video path')
parser.add_argument('--v_id', type=str, default='IMG_0298',
                    help='input video path')
parser.add_argument('--img_path', type=str, default='/home/cxu-serve/u1/lchen63/github/nerf-pytorch/data/lele_data3/images',
                    help='input video path')

args = parser.parse_args()


def video2img(v_path = None, v_id = None,  img_path = None, step = 5):
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    raw = cv2.VideoCapture(v_path)
    ret = True
    count = 0
    # imgs = []
    # store_step = [0, 40, 80, 120, 80, 85, 90, 95, 100,105, 110, 115, 120, 125, 390,400, 405, 410,420,430]
    # print(len(store_step))
    while ret:
        ret,frame = raw.read()
        if not ret:
            break
        print(ret,count)
        # print(frame.shape)
        # if count in  store_step:
        if count % step == 0:
            img_name = os.path.join( img_path, v_id + '_%05d.png'%count)
            print (img_name)
            cv2.imwrite(img_name, frame)
        count += 1
    # return imgs
video2img(args.v_path, args.v_id, args.img_path)