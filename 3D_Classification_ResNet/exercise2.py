
import os

if __name__ == "__main__":

    target_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/remove_test"

    for file_target in os.scandir(target_dir):
        os.remove(file_target.path)