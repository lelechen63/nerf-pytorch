
import cv2

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.tddfa_util import _parse_param
from utils.render_ctypes import render 
import argparse
import os
visualize_result = True

def process_face_image(image_p):
    tddfa = TDDFA()
    # Given a still image path and load to BGR channel
    face_boxes = FaceBoxes()

    if os.path.isdir(image_p):
        for img_p in os.listdir(image_p):
            image_path = os.path.join( image_p, img_p )
            # Given a still image path and load to BGR channel
            img = cv2.imread(image_path)

            # Detect faces, get 3DMM params and roi boxes
            boxes = face_boxes(img)
            n = len(boxes)
            if n == 0:
                print(f'No face detected, exit')
                sys.exit(-1)
            print(f'Detect {n} faces')

            param_lst, roi_box_lst = tddfa(img, boxes)

            if visualize_result:
                save_path = image_path.replace('images', 'renderred')
                save_folder = os.path.dirname(save_path)
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
                #######  visualize result #######
                ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
                render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=True, wfp = save_path)
                #################################

            param = param_lst[0]
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            print("exp code: ", alpha_exp)
    else:
        image_path = image_p
        # Given a still image path and load to BGR channel
        img = cv2.imread(image_path)

        # Detect faces, get 3DMM params and roi boxes
        boxes = face_boxes(img)
        n = len(boxes)
        if n == 0:
            print(f'No face detected, exit')
            sys.exit(-1)
        print(f'Detect {n} faces')

        param_lst, roi_box_lst = tddfa(img, boxes)

        if visualize_result:
            save_path = image_path.replace('images', 'renderred')
            save_folder = os.path.dirname(save_path)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            #######  visualize result #######
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=True, wfp = save_path)
            #################################

        param = param_lst[0]
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        print("exp code: ", alpha_exp)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_f', type=str, default="")

    return parser.parse_args()

if __name__ == '__main__':
    config = parse_args()
    process_face_image(config.img_f)
