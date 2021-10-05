

import os
from glob import glob
import numpy as np

import cv2
import torch
from PIL import Image
from matplotlib.pyplot import imshow

if __name__ == "__main__":

    target_path = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_test"

    target_list = glob(target_path + "/*")
    for file_idx, target_file in enumerate(target_list):
        target_3D = torch.load(target_file)
        #print(target_3D["class_num"])

        target = target_3D["target"]
        print(target.shape)

        # normalize to grayscale: 0 ~ 255
        target_tmp = target - np.min(target)
        target_normalize = (target_tmp * 255) / np.max(target_tmp)
        #print(target_normalize[:][:][0].shape)

        width, height, depth = target_normalize.shape
        for slice_idx in range(depth):
            #print(target_normalize[:, :, slice_idx].shape)

            image_pixel = Image.fromarray(target_normalize[:, :, slice_idx].astype(np.int), "I")
            image_pixel.save(f"./temp_test_image/test_{file_idx}_{slice_idx}.gif")

            #cv2.imshow("test", target_normalize[:][:][slice_idx])
            #cv2.waitKey(0)

            #break
        #break