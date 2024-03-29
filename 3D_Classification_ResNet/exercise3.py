
import os

from torchvision import transforms

import datasets_coronal
from datasets_coronal import ToTensor3D, Padding3D

if __name__ == "__main__":

    data_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_normal_candidate"
    save_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_crop_64_linear_12normal"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    datasets_coronal.Transform_Only(data_dir, save_dir,
                                    transform=transforms.Compose([
                                        ToTensor3D(),
                                        Padding3D((64, 64, 64))])
                                    )