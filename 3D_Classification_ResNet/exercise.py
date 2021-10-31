
import torch
from glob import glob

if __name__ == "__main__":

    target_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_new"
    for target_path in glob(target_dir + "/*"):
        data = torch.load(target_path)
        target, class_num, source = data["target"].float(), data["class_num"], data["source"]

        print(target.shape)

        #target_1 = torch.flip(target, [1, 2])
        #print(target_1.shape)

        #target_2 = torch.flip(target, [1, 3])
        #print(target_2.shape)

        target_3 = torch.flip(target, [2, 3])
        print(target_3.shape)

        target_4 = torch.fliplr(target)
        print(target_4.shape)

        break

    '''
    x = torch.arange(8).view(2, 2, 2)
    print(x)

    # sagittal rotate
    x1 = torch.flip(x, [0, 1])
    print(x1)

    # axial rotate
    x2 = torch.flip(x, [0, 2])
    print(x2)

    # coronal rotate
    x3 = torch.flip(x, [1, 2])
    print(x3)

    # mirror
    x4 = torch.fliplr(x)
    print(x4)
    '''