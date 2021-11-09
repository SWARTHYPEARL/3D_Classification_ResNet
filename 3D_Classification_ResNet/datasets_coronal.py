
import platform
if platform.system() == "Windows":
    import pydicom

import os
import uuid
import numpy as np

from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import img_crop, read_dicom, xywh2xyxy

import pandas as pd
#f_excel = pd.read_excel("Y:/SP_work/bone_mets_data/bone_mets_0504-01_이희진_중복삭제.xlsx", sheet_name="new_list", index_col="id")
f_excel = pd.read_excel("bone_mets_0504-01.xlsx", sheet_name="new_list", index_col="id")
SPINE_LEVEL_LIST = ["C7", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2"]
SPINE_STATUS = ("m", "b", "sct", "so", "x")
#print(f_excel["C7"][10021150])


class Dicom3D_Coronal_Dataset(Dataset):

    def __init__(self, data_dir: str, dicom_dir: str, label_dir: str, dicom_HU_level: int, dicom_HU_width: int,
                 data_savepath: str, data_saved: bool = False, transform=None, cache_num: int = 2000, set_status = None):
        """
        :param data_dir: absolute path of directories which contain data
        :param dicom_dir: relative path of dicom directory. ex) /images
        :param label_dir: relative path of label directory. ex) /label
        :param dicom_HU_level: Hounsfield Unit for dicom handling
        :param dicom_HU_width: Hounsfield Unit for dicom handling
        => dicom_dir/"directories"/label_dir/labels.file

        :param data_savepath: Reformed 3D data save path
        :param data_saved: whether reformed 3D data exist or not
        :param transform: torch transform
        :param cache_num: number of data for caching
        :param set_status: set status of all dataset

        Per data_dir, one directory -> one study
        Each data contains directory name and spine level to identify
        Dicom file name must have extension: '.dcm'
        Label file must be yolo format.
        """
        self.data_dir = data_dir
        self.dicom_dir = dicom_dir
        self.label_dir = label_dir
        self.dicom_HU_level = dicom_HU_level
        self.dicom_HU_width = dicom_HU_width
        self.data_savepath = data_savepath
        self.data_saved = data_saved
        self.transform = transform

        self.cache_num = cache_num
        self.cache_dict = {}

        self.set_status = set_status

        self.data_label_pathlist = []
        self.torchTensor_fullpathlist = []

        if self.data_saved is False:
            #dicom_dir_list = glob(self.data_dir + "/*")
            target_dir_list = glob(self.data_dir + "/*")
            #for target_CT_path in dicom_dir_list: # repeat per patient
            for target_dir_path in target_dir_list: # repeat per patient
                if not os.path.isdir(target_dir_path):
                    continue
                # print(target_CT_path)

                #patient_num = os.path.basename(target_CT_path)
                target_dir_name = os.path.basename(target_dir_path)
                label_dict = {
                    # Spine level - C: Cervical, T: Thoracic, L: Lumbar
                    # each level contains [dicom file path, bbox info]
                    "C7": [], "T1": [], "T2": [], "T3": [], "T4": [], "T5": [], "T6": [], "T7": [], "T8": [], "T9": [], "T10": [], "T11": [], "T12": [],
                    "L1": [], "L2": []
                }

                # extract and bind with dicom file and bounding box information
                target_dicom_list = glob(target_dir_path + self.dicom_dir + "/*")
                target_dicom_list_sorted = sorted(target_dicom_list, key=lambda x: pydicom.dcmread(x).SliceLocation, reverse=True)
                for target_dicom_path in target_dicom_list_sorted:
                    target_label_path = target_dir_path + self.label_dir + "/" + os.path.basename(target_dicom_path).split(".dcm")[0] + ".txt"
                    if os.path.isfile(target_label_path):
                        f_target_label = open(target_label_path, "r")
                        for label_line in f_target_label.readlines():
                            label_line_split = label_line.rstrip().split(" ")
                            label_class = int(label_line_split[0])
                            label_bbox = list(map(float, label_line_split[1:]))

                            label_dict[SPINE_LEVEL_LIST[label_class]].append([target_dicom_path, label_bbox])
                    else:
                        continue

                # Add lesion information of each spine level
                for target_spine_level in label_dict.keys():
                    if len(label_dict[target_spine_level]) == 0:
                        continue

                    if self.set_status is None:
                        target_spine_status = f_excel[target_spine_level][int(os.path.basename(target_dir_name))]
                    else:
                        target_spine_status = self.set_status
                    if target_spine_status in SPINE_STATUS:
                        label_dict[target_spine_level].insert(0, f"{target_spine_level}-{target_spine_status}-{target_dir_name}")
                    else:
                        #print(f"nothing-{target_spine_status}: {label_dict[target_spine_level]}")
                        continue
                    self.data_label_pathlist.append(label_dict[target_spine_level])

            # save temp .pt files for ssd caching
            print(f"total dataset: {len(self.data_label_pathlist)}")
            for target_idx in range(len(self.data_label_pathlist)):
                target_Tensor = self.load_data(target_idx)

                temp_filename = str(uuid.uuid4())
                temp_fullpath = self.data_savepath + "/" + temp_filename + ".pt"
                torch.save(target_Tensor, temp_fullpath)

                self.torchTensor_fullpathlist.append(temp_fullpath)
        else:
            self.torchTensor_fullpathlist = glob(self.data_savepath + "/*.pt")

    def load_data(self, idx: int):
        """
        :param idx: 3D data num in each dicom directory
        :return: torch Tensor data with Rescaled
        """

        # get 3D data
        target_3D = {}
        target_path_list = self.data_label_pathlist[idx]
        #target_class, patient_num = target_path_list[0].split("-")
        target_spine_level, target_spine_status, target_dir_name = target_path_list[0].split("-")

        wh_max = [0, 0] # ratio of (width, height)
        for target_path in target_path_list[1:]:
            target_bbox = target_path[1]

            if wh_max[0] < target_bbox[2]:
                wh_max[0] = target_bbox[2]
            if wh_max[1] < target_bbox[3]:
                wh_max[1] = target_bbox[3]

        pixel_width, pixel_height, _ = read_dicom(target_path_list[1][0], self.dicom_HU_width,
                                                  self.dicom_HU_level).shape
        for target_path in target_path_list[1:]:
            target_dicom_path, target_bbox = target_path[0], target_path[1]
            dicom_pixel = read_dicom(target_dicom_path, self.dicom_HU_width, self.dicom_HU_level)

            target_bbox = target_path[1]
            target_bbox[2], target_bbox[3] = wh_max
            tbox = xywh2xyxy(target_bbox)
            tbox_length = tbox * np.array([pixel_width, pixel_height, pixel_width, pixel_height],
                                                  dtype=np.float)

            # crop exception of 'out of bounds' position
            if tbox_length[0] < 0:
                tbox_length[2] -= tbox_length[0]
                tbox_length[0] = 0.0
            if tbox_length[1] < 0:
                tbox_length[3] -= tbox_length[1]
                tbox_length[1] = 0.0
            if tbox_length[2] > pixel_width:
                tbox_length[0] -= (tbox_length[2] - pixel_width)
                tbox_length[2] = float(pixel_width)
            if tbox_length[3] > pixel_height:
                tbox_length[1] -= (tbox_length[3] - pixel_height)
                tbox_length[3] = float(pixel_height)
            # crop type: img[y1:y2, x1:x2]
            dicom_crop = img_crop(dicom_pixel, tbox_length)
            #print(f"{dicom_crop.shape} - {tbox_length}")

            if "target" not in target_3D:
                target_3D["target"] = dicom_crop
                #target_3D["class_num"] = 0 if target_class == "normal" else 1
                target_3D["class_num"] = SPINE_STATUS.index(target_spine_status)
                #target_3D["source"] = [patient_num, os.path.basename(target_dicom_path)]
                target_3D["source"] = [target_dir_name, target_spine_level]
            else:
                target_3D["target"] = np.append(target_3D["target"], dicom_crop, axis=2)

        if self.transform:
            target_3D = self.transform(target_3D)

        return target_3D

    def load_data_zero_padding(self, idx:int):
        """
        :param idx: 3D data num in each dicom directory
        :return: torch Tensor data with Rescaled
        """

        # get 3D data
        target_3D = {}
        target_path_list = self.data_label_pathlist[idx]
        # target_class, patient_num = target_path_list[0].split("-")
        target_spine_level, target_spine_status, target_dir_name = target_path_list[0].split("-")

        wh_max = [0, 0]  # ratio of (width, height)
        for target_path in target_path_list[1:]:
            target_bbox = target_path[1]

            if wh_max[0] < target_bbox[2]:
                wh_max[0] = target_bbox[2]
            if wh_max[1] < target_bbox[3]:
                wh_max[1] = target_bbox[3]

        pixel_width, pixel_height, _ = read_dicom(target_path_list[1][0], self.dicom_HU_width,
                                                  self.dicom_HU_level).shape
        for target_path in target_path_list[1:]:
            target_dicom_path, target_bbox = target_path[0], target_path[1]
            dicom_pixel = read_dicom(target_dicom_path, self.dicom_HU_width, self.dicom_HU_level)

            target_bbox = target_path[1]
            #target_bbox[2], target_bbox[3] = wh_max
            tbox = xywh2xyxy(target_bbox)
            tbox_length = tbox * np.array([pixel_width, pixel_height, pixel_width, pixel_height],
                                          dtype=np.float)

            # crop exception of 'out of bounds' position
            if tbox_length[0] < 0:
                tbox_length[2] -= tbox_length[0]
                tbox_length[0] = 0.0
            if tbox_length[1] < 0:
                tbox_length[3] -= tbox_length[1]
                tbox_length[1] = 0.0
            if tbox_length[2] > pixel_width:
                tbox_length[0] -= (tbox_length[2] - pixel_width)
                tbox_length[2] = float(pixel_width)
            if tbox_length[3] > pixel_height:
                tbox_length[1] -= (tbox_length[3] - pixel_height)
                tbox_length[3] = float(pixel_height)
            # crop type: img[y1:y2, x1:x2]
            dicom_crop = img_crop(dicom_pixel, tbox_length)
            # print(f"{dicom_crop.shape} - {tbox_length}")

            # zero padding
            pad_width, pad_height = wh_max * np.array([pixel_width, pixel_height], dtype=np.float)
            crop_height, crop_width = dicom_crop.shape
            top_padding = int((pad_height - crop_height) / 2)
            bottom_padding = int((pad_height - crop_height)) - top_padding
            left_padding = int((pad_width - crop_width) / 2)
            right_padding = int((pad_width - crop_width)) - left_padding

            dicom_padding = np.pad(dicom_crop, ((top_padding, bottom_padding), (left_padding, right_padding)), "constant", constant_values=0)

            if "target" not in target_3D:
                #target_3D["target"] = dicom_crop
                target_3D["target"] = dicom_padding
                # target_3D["class_num"] = 0 if target_class == "normal" else 1
                target_3D["class_num"] = SPINE_STATUS.index(target_spine_status)
                target_3D["source"] = [target_dir_name, target_spine_level]
            else:
                #target_3D["target"] = np.append(target_3D["target"], dicom_crop, axis=2)
                target_3D["target"] = np.append(target_3D["target"], dicom_padding, axis=2)

        if self.transform:
            target_3D = self.transform(target_3D)

        return target_3D


    def __len__(self):
        return len(self.torchTensor_fullpathlist)

    def __getitem__(self, idx: int):
        """
        :param idx: 3D data num in each dicom directory
        :return: torch Tensor data with Rescaled
        """

        return torch.load(self.torchTensor_fullpathlist[idx])




class ToTensor3D(object):
    """
    convert numpy array to tensor(torch)
    """

    def __call__(self, target_3D):
        target, class_num = target_3D["target"], target_3D["class_num"]
        source = target_3D["source"]

        # numpy HWD
        # torch DHW
        target = target.transpose((2, 0, 1))
        # target = np.expand_dims(target, axis=0)  # expand dim for add channels
        # target = np.expand_dims(target, axis=0)  # expand dim for mini-batch. squeeze after converting torch Tensor
        return {"target": torch.from_numpy(target),
                # "class_num": torch.from_numpy(np.array(class_num, dtype=np.int)),
                "class_num": torch.Tensor([class_num]),
                "source": source}


class Rescale3D(object):
    """
    3D data rescale in torch Tensor

    """

    def __init__(self, output_size):
        """
        :param output_size: tuple - (D, H, W)
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, target_3D):
        target, class_num = target_3D["target"], target_3D["class_num"]
        source = target_3D["source"]

        # generate mini-batch & channels dim to resize torch Tensor
        target = target.unsqueeze(0)
        target = target.unsqueeze(0)

        # input dimensions: mini-batch x channels x [optional depth] x [optional height] x width
        target_resize = torch.nn.functional.interpolate(target, self.output_size)
        target_resize = target_resize.squeeze(0)  # squeeze to remove mini-batch dim

        return {"target": target_resize,
                "class_num": class_num,
                "source": source}


class Padding3D(object):
    """
        3D data rescale in torch Tensor

        """

    def __init__(self, output_size):
        """
        :param output_size: tuple - (D, H, W)
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, target_3D):
        target, class_num = target_3D["target"], target_3D["class_num"]
        source = target_3D["source"]
        print(f"init shape: {target.shape}")

        target_depth, target_height, target_width = target.shape
        target_resize = None
        # resize (H, W)
        for idx_depth in range(target_depth):
            target_slice = target[idx_depth, :, :]

            target_slice = target_slice.unsqueeze(0)
            target_slice = target_slice.unsqueeze(0)

            target_slice_resize = torch.nn.functional.interpolate(target_slice, self.output_size[1:])
            if target_resize is None:
                target_resize = target_slice_resize.detach().clone()
            else:
                target_resize = torch.cat([target_resize, target_slice_resize], dim=1)

        # resize (D)
        for count_padding in range(self.output_size[0] - target_depth):
            slice_zero = torch.zeros(1, 1, self.output_size[1], self.output_size[2])
            if (count_padding % 2) == 1:
                target_resize = torch.cat([target_resize, slice_zero], dim=1)
            else:
                target_resize = torch.cat([slice_zero, target_resize], dim=1)

        #print(f"resized shape: {target_resize.shape}")
        return {"target": target_resize,
                "class_num": class_num,
                "source": source}


class Tensor3D_Dataset(Dataset):

    def __init__(self, tensor_dir: str = None, tensor_list: list = None, cache_num: int = 2000, flip_dir = None):
        """
        Load Tensor format datasets
        :param tensor_dir: directory path of .pt data
        :param tensor_list: path list of .pt data
        :param cache_num: number of data for caching
        """

        self.tensor_dir = tensor_dir
        self.tensor_list = glob(self.tensor_dir + "/*.pt") if tensor_list is None else tensor_list

        self.cache_num = cache_num
        self.cache_dict = {}

        self.flip_dir = flip_dir

        # Deadlock must be solved on distributed parallel processing
        #if self.flip_dir is not None:
        #    if not os.path.isdir(self.flip_dir):
        #        os.makedirs(self.flip_dir)
        #    self.Tensor3D_flip()


    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx: int):
        """
        :param idx: 3D data num in each dicom directory
        :return: torch Tensor data with Rescaled
        """

        if self.cache_num == 0:
            return torch.load(self.tensor_list[idx])

        cache_key = str(idx)
        if cache_key not in self.cache_dict:
            if len(self.cache_dict) == self.cache_num:
                self.cache_dict.popitem()
            self.cache_dict[cache_key] = torch.load(self.tensor_list[idx])

        return self.cache_dict[cache_key]

    def Tensor3D_flip(self):
        size_dataset = self.__len__()
        for target_idx in range(size_dataset):
            data = self.__getitem__(target_idx)
            target, class_num, source = data["target"].float(), data["class_num"], data["source"]

            target_flip = torch.flip(target, [2, 3])
            target_fliplr = torch.fliplr(target)
            target_flip_fliplr = torch.fliplr(target_flip)

            target_flip_dict = {"target": target_flip, "class_num": class_num, "source": source}
            target_fliplr_dict = {"target": target_fliplr, "class_num": class_num, "source": source}
            target_flip_fliplr_dict = {"target": target_flip_fliplr, "class_num": class_num, "source": source}

            temp_filename = str(uuid.uuid4())
            temp_fullpath = self.flip_dir + "/" + temp_filename + ".pt"
            torch.save(target_flip_dict, temp_fullpath)
            self.tensor_list.append(temp_fullpath)

            temp_filename = str(uuid.uuid4())
            temp_fullpath = self.flip_dir + "/" + temp_filename + ".pt"
            torch.save(target_fliplr_dict, temp_fullpath)
            self.tensor_list.append(temp_fullpath)

            temp_filename = str(uuid.uuid4())
            temp_fullpath = self.flip_dir + "/" + temp_filename + ".pt"
            torch.save(target_flip_fliplr_dict, temp_fullpath)
            self.tensor_list.append(temp_fullpath)




    def get_tensor_list(self):
        return self.tensor_list

    def set_tensor_list(self, tensor_list: list):
        self.tensor_list = tensor_list

    def pop_tensor_list(self, idx: int):
        del self.tensor_list[idx]

class Transform_Only(Dataset):
    def __init__(self, data_dir: str, save_dir: str, transform=None):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.transform = transform

        for target_path in glob(data_dir + "/*.pt"):
            target_3D = torch.load(target_path)
            target_Tensor = self.transform(target_3D)

            temp_filename = str(uuid.uuid4())
            temp_fullpath = self.save_dir + "/" + temp_filename + ".pt"
            torch.save(target_Tensor, temp_fullpath)


if __name__ == "__main__":

    target_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/datasets/normal_candidate/Coronal"
    #target_dir = "C:/Users/SNUBH/SP_work/Python_Project/yolov3_DICOM/data/billion/bone_coronal_20210914"
    #target_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/datasets"
    set_status = "x"
    dicom_dir = "/dicom"
    #dicom_dir = "/images"
    label_dir = "/yolo"
    #label_dir = "/labels_yolo"
    dicom_HU_level = 300
    dicom_HU_width = 2500
    data_savepath = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_normal_candidate"
    #data_savepath = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_crop_original"
    #data_savepath = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_new_256"
    #temp_savepath = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_test"
    data_saved = False

    dataset = Dicom3D_Coronal_Dataset(
        data_dir=target_dir,
        dicom_dir=dicom_dir,
        label_dir=label_dir,
        dicom_HU_level=dicom_HU_level,
        dicom_HU_width=dicom_HU_width,
        data_savepath=data_savepath,
        data_saved=data_saved,
        set_status=set_status
        #transform=transforms.Compose([
            #ToTensor3D(),
            #Rescale3D((64, 192, 256))
            #Rescale3D((64, 256, 256))
        #])
    )

    height, width, depth = (0, 0, 0)
    for target in dataset:

        target_height, target_width, target_depth = target["target"].shape

        #height += target_height
        #width += target_width
        #depth += target_depth

        if height < target_height:
            height = target_height
        if width < target_width:
            width = target_width
        if depth < target_depth:
            depth = target_depth
    #print(np.array([height, width, depth]) / len(dataset))
    print((height, width, depth))