import os
import cv2 as cv
import numpy as np


file_path = "E:/cvfinal/yolov5/datasets/VOC/images/"
path_list = os.listdir(file_path)
save_path = "E:/cvfinal/yolov5/datasets/VOC/images_argu/"
for image_name in path_list:
    image_path = file_path + image_name
    image = cv.imread(image_path, 1)
    data0 = np.asarray(image, dtype=float)
    data1 = data0 * 0.9
    data2 = data0 * 1.1
    out_name = image_name.split('.')[0]
    out_path0 = save_path + out_name + "00.jpg"
    out_path1 = save_path + out_name + "01.jpg"
    out_path2 = save_path + out_name + "02.jpg"
    cv.imwrite(out_path0, data0)
    cv.imwrite(out_path1, data1)
    cv.imwrite(out_path2, data2)
