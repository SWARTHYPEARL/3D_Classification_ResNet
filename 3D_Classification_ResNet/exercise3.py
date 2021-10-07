

import torch
from glob import glob

dataset_train_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp/train"
pt_list = glob(dataset_train_dir + "/*.pt")

for target_path in pt_list:
    torch.load(target_path)
