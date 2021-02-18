
import cv2

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.tddfa_util import _parse_param

visualize_result = True

def process_face_image(image_path):
    tddfa = TDDFA()
    face_boxes = FaceBoxes()
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
        from utils.render_ctypes import render 
        #######  visualize result #######
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=True, wfp="render.jpg")
        #################################

    param = param_lst[0]
    R, offset, alpha_shp, alpha_exp = _parse_param(param)
    print("exp code: ", alpha_exp)


if __name__ == '__main__':
    process_face_image("/home/cxu-serve/p1/lchen63/nerf/data/1/1_neutral_processed/images/1.png")