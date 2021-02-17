import os 
import cv2



def get_imgs(img_folder, save_folder):
    img_lists = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,\
                  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,\
                 27, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 53, 54]

    for i in img_lists:
        img_path = os.path.join( img_folder, '%d.jpg'%i )
        print (i, img_path)
        img = cv2.imread(img_path)
        img =  cv2.resize(img, (1728, 2592), interpolation = cv2.INTER_AREA)
        cv2.imwrite( os.path.join(save_folder, '%d.png'%i),  img)
img_path = '/home/cxu-serve/p1/lchen63/nerf/data/1/1_neutral'
get_imgs(img_path, save_folder = img_path +'_processed')