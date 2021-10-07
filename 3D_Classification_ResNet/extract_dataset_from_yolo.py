
import os
import shutil
from glob import glob

import torch

def datalist_from_yolo(text_path: str, temp_dataset_path):

    data_list = []
    for target_line in open(text_path, "r").readlines():
        target_id = target_line.split("\\")[-3]
        if target_id not in data_list:
            data_list.append(target_id)

    target_list = []
    for target_path in glob(temp_dataset_path + "/*.pt"):
        target_3D = torch.load(target_path)
        source = target_3D["source"]

        if source[0] in data_list:
            target_list.append(target_path)

    return target_list

if __name__ == "__main__":

    train_txt_path = "C:/Users/SNUBH/SP_work/Python_Project/yolov3_DICOM/data/billion/bone_coronal_20210914/train_0.txt"
    val_txt_path = "C:/Users/SNUBH/SP_work/Python_Project/yolov3_DICOM/data/billion/bone_coronal_20210914/val_0.txt"

    dataset_path = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp"
    save_path = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp"

    train_list = []
    for target_line in open(train_txt_path, "r").readlines():
        target_id = target_line.split("\\")[-3]
        if target_id not in train_list:
            train_list.append(target_id)
        #break
    #print(len(train_list))

    val_list = []
    for target_line in open(val_txt_path, "r").readlines():
        target_id = target_line.split("\\")[-3]
        if target_id not in val_list:
            val_list.append(target_id)

    for target_path in glob(dataset_path + "/*.pt"):
        target_3D = torch.load(target_path)
        source = target_3D["source"]
        #print(source)

        if source[0] in train_list:
            #print(f"train - {source[0]}")
            #shutil.move(target_path, save_path + f"/train/{os.path.basename(target_path)}")
            shutil.copy(target_path, save_path + f"/train/{os.path.basename(target_path)}")
        elif source[0] in val_list:
            #print(f"val - {source[0]}")
            #shutil.move(target_path, save_path + f"/val/{os.path.basename(target_path)}")
            shutil.copy(target_path, save_path + f"/val/{os.path.basename(target_path)}")
        else:
            print(f"Not included - {source[0]}")

        #break