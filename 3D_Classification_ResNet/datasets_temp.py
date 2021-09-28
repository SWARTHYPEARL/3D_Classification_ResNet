
import os
import uuid

from glob import glob
from torch.utils.data import Dataset

from utils import *


class Dicom3D_CT_Crop_Dataset(Dataset):

    def __init__(self, dicom_dir: str, label_dir: str, dicom_HU_level: int, dicom_HU_width: int, cache_num: int,
                 temp_savepath: str, isTest: bool = False, transform=None):
        """
        :param dicom_dir: absolute path of directories which are in dicom files
        :param label_dir: relative path of label directory. ex) /label
        :param dicom_HU_level: Hounsfield Unit for dicom handling
        :param dicom_HU_width: Hounsfield Unit for dicom handling
        => dicom_dir/"directories"/label_dir/labels.file
        """
        self.dicom_dir = dicom_dir
        self.label_dir = label_dir
        self.dicom_HU_level = dicom_HU_level
        self.dicom_HU_width = dicom_HU_width
        self.cache_num = cache_num
        self.temp_savepath = temp_savepath
        self.isTest = isTest
        self.transform = transform

        self.cache_dict = {}

        self.data_label_pathlist = []
        self.torchTensor_fullpathlist = []
        if isTest is False:
            # check data for counting and ...
            dicom_dir_list = glob(self.dicom_dir + "/*")
            for target_CT_path in dicom_dir_list:
                # print(target_CT_path)

                dicom_label_list = glob(target_CT_path + self.label_dir + "/FILE*.txt")
                target_3Ddataset = {}
                # 집어 넣을 때, label 파일 순서가 아닌 dicom 정보를 통해 정렬되는 방식으로 수정할 것.
                for path_label in dicom_label_list:

                    if os.path.isfile(target_CT_path + "/" + os.path.basename(path_label).split(".txt")[0] + ".dcm"):
                        file_label = open(path_label, "r")
                        label_class = file_label.readline().split(" ")[0]
                        if label_class not in target_3Ddataset:
                            target_3Ddataset[label_class] = [path_label]
                            # print(target_3Ddataset[label_class])
                        else:
                            target_3Ddataset[label_class].append(path_label)

                    else:
                        continue

                self.data_label_pathlist.extend(list(target_3Ddataset.values()))

            # save temp .np files for ssd caching
            for target_idx in range(len(self.data_label_pathlist)):
                target_Tensor = self.load_data(target_idx)

                temp_filename = str(uuid.uuid4())
                temp_fullpath = temp_savepath + "/" + temp_filename + ".pt"
                torch.save(target_Tensor, temp_fullpath)

                self.torchTensor_fullpathlist.append(temp_fullpath)
        else:
            self.torchTensor_fullpathlist = glob(self.temp_savepath + "/*.pt")

    def load_data(self, idx: int):
        """
        :param idx: 3D data num in each dicom directory
        :return: torch Tensor data with Rescaled
        """

        # relative label path count
        target_backpath = self.label_dir
        backpath_count = 0
        while os.path.dirname(self.label_dir) != target_backpath:
            backpath_count += 1
            target_backpath = os.path.dirname(self.label_dir)
        # print(backpath_count)

        # get 3D data
        target_3D = {}
        target_path = self.data_label_pathlist[idx]
        for path_label in target_path:

            file_label = open(path_label, "r")

            target_CT_path = os.path.dirname(path_label)
            for _ in range(backpath_count):
                target_CT_path = os.path.dirname(target_CT_path)
            dicom_pixel = read_dicom(target_CT_path + "/" + os.path.basename(path_label).split(".txt")[0] + ".dcm",
                                     self.dicom_HU_width, self.dicom_HU_level)

            # class, x(ratio), y(ratio), w(ratio), h(ratio)
            label_LabelImg_YOLOv3 = np.array(file_label.readline().split(" "), dtype=np.float)
            pixel_width, pixel_height, _ = dicom_pixel.shape

            tbox = xywh2xyxy(label_LabelImg_YOLOv3[1:]) * np.array(
                [pixel_width, pixel_height, pixel_width, pixel_height], dtype=np.float)

            # crop type: img[y1:y2, x1:x2]
            dicom_crop = img_crop(dicom_pixel, tbox)

            label_class = str(int(label_LabelImg_YOLOv3[0]))
            if "target" not in target_3D:
                target_3D["target"] = dicom_crop
                #target_3D["class_num"] = "0" if int(label_class) < 12 else "1"
                target_3D["class_num"] = np.array([1, 0]) if int(label_class) < 12 else np.array([0, 1])
                target_3D["source"] = path_label
            else:
                target_3D["target"] = np.append(target_3D["target"], dicom_crop, axis=2)

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

        if self.cache_num == 0:
            return torch.load(self.torchTensor_fullpathlist[idx])

        cache_key = str(idx)
        if cache_key not in self.cache_dict:
            if len(self.cache_dict) == self.cache_num:
                self.cache_dict.popitem()
            self.cache_dict[cache_key] = torch.load(self.torchTensor_fullpathlist[idx])

        return self.cache_dict[cache_key]


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
                "class_num": torch.from_numpy(class_num),
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


class Tensor3D_Dataset(Dataset):

    def __init__(self, tensor_dir: str):
        """
        Load Tensor format datasets
        :param tensor_dir:
        """

        self.tensor_dir = tensor_dir
        self.torchTensor_list = glob(self.tensor_dir + "/*.pt")

    def __len__(self):
        return len(self.torchTensor_list)

    def __getitem__(self, idx: int):
        return torch.load(self.torchTensor_list[idx])
