
import os
import uuid

from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *

import pandas as pd
f_excel = pd.read_excel("Y:/SP_work/bone_mets_data/bone_mets_0504-01_이희진_중복삭제.xlsx", sheet_name="new_list", index_col="id")
anatomic_level_list = ["C7", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2"]
#print(f_excel["C7"][10021150])


class Dicom3D_Coronal_Dataset(Dataset):

    def __init__(self, data_dir: str, dicom_dir: str, label_dir: str, dicom_HU_level: int, dicom_HU_width: int, cache_num: int,
                 temp_savepath: str, isTest: bool = False, transform=None):
        """
        :param data_dir: absolute path of directories which are in datas
        :param dicom_dir: relative path of dicom directory. ex) /images
        :param label_dir: relative path of label directory. ex) /label
        :param dicom_HU_level: Hounsfield Unit for dicom handling
        :param dicom_HU_width: Hounsfield Unit for dicom handling
        => dicom_dir/"directories"/label_dir/labels.file
        """
        self.data_dir = data_dir
        self.dicom_dir = dicom_dir
        self.label_dir = label_dir
        self.dicom_HU_level = dicom_HU_level
        self.dicom_HU_width = dicom_HU_width
        self.cache_num = cache_num
        self.temp_savepath = temp_savepath
        self.isTest = isTest
        self.transform = transform

        self.cache_dict = {}

        self
        self.data_label_pathlist = []
        self.torchTensor_fullpathlist = []
        if isTest is False:
            # check data for counting and ...
            dicom_dir_list = glob(self.data_dir + "/*")
            for target_CT_path in dicom_dir_list: # repeat per patient
                if not os.path.isdir(target_CT_path):
                    continue
                # print(target_CT_path)

                patient_num = os.path.basename(target_CT_path)
                label_dict = {
                    # 0: cervical_7, 1: thoracic_1, 2: thoracic_2, 3: thoracic_3, 4: thoracic_4, 5: thoracic_5, 6: thoracic_6, 7: thoracic_7, 8: thoracic_8, 9: thoracic_9, 10: thoracic_10, 11: thoracic_11, 12: thoracic_12, 13: lumber_1, 14: lumber_2
                    0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                    13: [], 14: []
                }
                dicom_label_list = glob(target_CT_path + self.label_dir + "/FILE*.txt")
                #target_3Ddataset = {}
                # 집어 넣을 때, label 파일 순서가 아닌 dicom 정보를 통해 정렬되는 방식으로 수정할 것.
                for path_label in dicom_label_list:
                    target_dicom_path = target_CT_path + self.dicom_dir + "/" + os.path.basename(path_label).split(".txt")[0] + ".dcm"
                    #print(target_dicom_path)
                    if os.path.isfile(target_dicom_path): # check dicom file exist
                        file_label = open(path_label, "r")
                        # label_class = file_label.readline().split(" ")[0]
                        for label_line in file_label.readlines():
                            label_line_split = label_line.rstrip().split(" ")
                            label_class = int(label_line_split[0])
                            label_bbox = list(map(float, label_line_split[1:]))
                            #print(label_bbox)

                            label_dict[label_class].append([target_dicom_path, label_bbox])
                        #print(label_dict)

                    else:
                        continue

                for label_dict_key in label_dict.keys():
                    if len(label_dict[label_dict_key]) == 0:
                        continue
                    anatomical_status = f_excel[anatomic_level_list[label_dict_key]][int(os.path.basename(target_CT_path))]
                    if anatomical_status == "m":
                        #print(label_dict[label_dict_key])
                        label_dict[label_dict_key].insert(0, f"lesion-{patient_num}")
                    elif anatomical_status == "b":
                        #print(label_dict[label_dict_key])
                        label_dict[label_dict_key].insert(0, f"lesion-{patient_num}")
                    elif anatomical_status == "sct":
                        #print(label_dict[label_dict_key])
                        label_dict[label_dict_key].insert(0, f"lesion-{patient_num}")
                    elif anatomical_status == "so":
                        #print(label_dict[label_dict_key])
                        label_dict[label_dict_key].insert(0, f"lesion-{patient_num}")
                    elif anatomical_status == "x":
                        #print(label_dict[label_dict_key])
                        label_dict[label_dict_key].insert(0, f"normal-{patient_num}")
                    else:
                        #print(f"nothing-{anatomical_status}: {label_dict[label_dict_key]}")
                        continue
                    self.data_label_pathlist.append(label_dict[label_dict_key])

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

        # get 3D data
        target_3D = {}
        target_path_list = self.data_label_pathlist[idx]
        target_class, patient_num = target_path_list[0].split("-")

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
                target_3D["class_num"] = 0 if target_class == "normal" else 1
                target_3D["source"] = [patient_num, os.path.basename(target_dicom_path)]
            else:
                target_3D["target"] = np.append(target_3D["target"], dicom_crop, axis=2)

        '''
        # tbox_max - ratio of (lt_x, lt_y, rb_x, rb_y)
        
        for target_path in target_path_list[1:]:
            #target_dicom_path = target_path[0]
            target_bbox = target_path[1]
            #print(f"path: {os.path.basename(target_dicom_path)}, bbox: {target_bbox}")

            tbox = xywh2xyxy(target_bbox)
            #target_lt_x = round(tbox[0] - (tbox[2] / 2))
            #target_lt_y = round(tbox[1] - (tbox[3] / 2))
            #target_rb_x = round(tbox[0] + (tbox[2] / 2))
            #target_rb_y = round(tbox[1] + (tbox[3] / 2))
            target_lt_x, target_lt_y, target_rb_x, target_rb_y = tbox
            if tbox_max[0] > target_lt_x: # left-top-x update
                tbox_max[0] = target_lt_x
            if tbox_max[1] > target_lt_y: # left-top-y update
                tbox_max[1] = target_lt_y
            if tbox_max[2] < target_rb_x: # right-bottom-x update
                tbox_max[2] = target_rb_x
            if tbox_max[3] < target_rb_y: # right-bottom-y update
                tbox_max[3] = target_rb_y
        # tbox_max convert - ratio of (x, y, w, h)
        #tbox_max[0] = round((tbox_max[0] + tbox_max[2]) / 2)
        #tbox_max[1] = round((tbox_max[1] + tbox_max[3]) / 2)
        #tbox_max[2] = round((tbox_max[2] - tbox_max[0]) * 2)
        #tbox_max[3] = round((tbox_max[3] - tbox_max[1]) * 2)

        pixel_width, pixel_height, _ = read_dicom(target_path_list[1][0], self.dicom_HU_width, self.dicom_HU_level).shape
        tbox_max_length = tbox_max * np.array([pixel_width, pixel_height, pixel_width, pixel_height], dtype=np.float)
        for target_path in target_path_list[1:]:
            target_dicom_path = target_path[0]
            dicom_pixel = read_dicom(target_dicom_path, self.dicom_HU_width, self.dicom_HU_level)

            #crop type: img[y1:y2, x1:x2]
            dicom_crop = img_crop(dicom_pixel, tbox_max_length)

            if "target" not in target_3D:
                target_3D["target"] = dicom_crop
                #target_3D["class_num"] = "0" if int(label_class) < 12 else "1"
                #target_3D["class_num"] = np.array([1, 0]) if target_class == "normal" else np.array([0, 1])
                target_3D["class_num"] = 0 if target_class == "normal" else 1
                target_3D["source"] = [patient_num, os.path.basename(target_dicom_path)]
            else:
                target_3D["target"] = np.append(target_3D["target"], dicom_crop, axis=2)
                #target_3D["source"].append(target_dicom_path)
        #print(target_3D)
        '''

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
                "class_num": class_num,
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


if __name__ == "__main__":

    #target_dir = "C:/Users/SNUBH/SP_work/Python_Project/yolov3_DICOM/data/billion/bone_coronal_20210914"
    target_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/datasets"
    dicom_dir = "/images"
    label_dir = "/labels_yolo"
    dicom_HU_level = 300
    dicom_HU_width = 2500
    cache_num = 100
    temp_savepath = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_test"
    isTest = False

    dataset = Dicom3D_Coronal_Dataset(
        data_dir=target_dir,
        dicom_dir=dicom_dir,
        label_dir=label_dir,
        dicom_HU_level=dicom_HU_level,
        dicom_HU_width=dicom_HU_width,
        cache_num=cache_num,
        temp_savepath=temp_savepath,
        isTest=isTest
        #transform=transforms.Compose([
        #    ToTensor3D(),
        #    Rescale3D((64, 128, 128))
        #])
    )

    height, width, depth = (0, 0, 0)
    for target in dataset:

        _, target_height, target_width, target_depth = target["target"].shape

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