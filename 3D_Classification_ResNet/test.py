
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from ResNet_3D import r3d_18, r3d_34, load_pretrained_model
from datasets_coronal import Tensor3D_Dataset, datalist_from_yolo

from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interpolate

import time
import argparse
from utils import AverageMeter, ProgressMeter
from collections import defaultdict

import pandas as pd
import numpy as np


def test_model(opt):
    opt.device = opt.device if type(opt.device) is int else opt.device[0]

    test_dataset = Tensor3D_Dataset(opt.dataset_dir, opt.data_pathlist)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = None
    if opt.arch == "r3d_18":
        model = r3d_18().float()
    elif opt.arch == "r3d_34":
        model = r3d_34().float()

    print("I was here..")
    if opt.multiprocessing_distributed:
        #dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:50000", world_size=1, rank=1)
        #torch.cuda.set_device(opt.device)
        #model = model.to(opt.device)
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.device])
        model = torch.nn.parallel.DistributedDataParallel(model)
    torch.cuda.set_device(opt.device)
    model = model.to(opt.device)
    print(opt.device)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(test_loader),
        # [batch_time, data_time, losses, top1, top5],
        [batch_time, data_time],
        prefix="[{}/{}]\t".format(len(test_loader)))

    model.eval()

    activation = nn.Softmax(dim=1).to(opt.device)

    results = {"results": defaultdict(list)}
    end_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # measure data loading time
            data_time.update(time.time() - end_time)

            inputs, labels = data["target"].to(opt.device), data["class_num"].to(opt.device)
            source = data["source"]
            # print(f"inputs: {inputs}")
            # print(f"labels: {labels}")

            outputs = model(inputs.float())
            outputs_act = activation(outputs)
            # print(outputs_act)

            # _, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.max(outputs_act.data, 1)
            #label_truth = torch.tensor([0 if label_list[0] == 1 else 1 for label_list in labels]).to(device)  # just for binary class
            label_truth = labels

            #print(outputs_act)
            #print(label_truth)
            #print(predicted)
            #print(outputs_act[range(label_truth.size(0)), label_truth])

            label_id = source[0][0]
            label_file = source[1][0]
            for batch_idx in range(len(outputs)):
                results["results"][f"{label_id}_{label_file}"].append({
                    #"predict": outputs_act[batch_idx][predicted.item()].item(),
                    "predict": outputs_act[batch_idx][1].item(),
                    "label": label_truth[batch_idx].item() # 0 is normal, 1 is lesion
                })

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            progress.display(i)

    del model

    predict_list = []
    label_list = []
    for target_result in results["results"].items():
        #print(target_result)
        label_id_file, target_predict_list = target_result
        for target_predict in target_predict_list:
            #predict_list.append(target_predict["predict"] if target_predict["label"] == 1 else 1.0 - target_predict["predict"])
            predict_list.append(target_predict["predict"])
            label_list.append(target_predict["label"])
    #print(np.array(predict_list, dtype=np.float))
    #print(np.array(label_list, dtype=np.int))

    ROC_curve_plot([predict_list], [label_list], opt.test_save_path)


def ROC_curve_plot(predict_list: list, label_list: list, save_path: str):
    # sklearn ROC calcalte higher label as positive case

    fig = plt.figure(figsize=(10, 7), dpi=600)

    for idx in range(len(predict_list)):
        predict_tensorlist = predict_list[idx]
        label_tensorlist = label_list[idx]

        # print(metrics.roc_auc_score(label_tensorlist, predict_tensorlist))
        auc_score = metrics.roc_auc_score(label_tensorlist, predict_tensorlist)
        print("AUC: %0.3f" % auc_score)

        fpr, tpr, thresholds = metrics.roc_curve(label_tensorlist, predict_tensorlist)
        # print(fpr)
        # print(tpr)
        # print(thresholds)

        fig = plt.figure(figsize=(4, 4), dpi=600)

        f1 = interpolate.interp1d(fpr, tpr)
        fpr_new = np.linspace(fpr.min(), fpr.max(), num=20, endpoint=True)
        # print(type(fpr_new))
        # print(type(f1(fpr_new)))

        x_final = np.append([0], fpr_new)
        y_final = np.append([0], f1(fpr_new))
        plt.plot(x_final, y_final, color="red", linewidth=2, label="ROC curve (area = %0.3f)" % auc_score)
        plt.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")

        plt.fill_between(x_final, 0, y_final, color="gray", alpha=0.2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=8)
        plt.ylabel('True Positive Rate', fontsize=8)
        plt.legend(loc="lower right")

        # plt.show()
        fig.savefig(f"{save_path}_plot_{idx}.png")

def parse_opts_excel():

    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", default="train_sheet.xlsx", type=str, help="excel sheet of train list")
    opt = parser.parse_args()

    f_excel = pd.read_excel(opt.excel_path, sheet_name="Sheet1")
    f_excel = f_excel.fillna("")
    opt_list = []
    for excel_row, excel_case in enumerate(f_excel["case"]):
        if excel_case == "":
            break

        opt = parser.parse_args()

        opt.device = [int(x.strip()) for x in str(f_excel["device"][excel_row]).split(",")]

        opt.arch = f_excel["arch"][excel_row]
        opt.pretrained_path = None if f_excel["pretrained_path"][excel_row] == "" else f_excel["pretrained_path"][excel_row]

        opt.dataset_dir = f_excel["dataset_dir"][excel_row]
        opt.class_normal = [x.strip() for x in f_excel["class_normal"][excel_row].split(",")]
        opt.class_abnormal = [x.strip() for x in f_excel["class_abnormal"][excel_row].split(",")]
        opt.data_pathlist = None if f_excel["data_pathlist_txt"][excel_row] == "" else datalist_from_yolo(f_excel["data_pathlist_txt"][excel_row], f_excel["dataset_dir"][excel_row], opt.class_normal, opt.class_abnormal)

        opt.test_save_path = f_excel["test_save_path"][excel_row] + f"/{int(excel_case)}"

        opt.multiprocessing_distributed = True if str(
            f_excel["multiprocessing_distributed"][excel_row]) == "True" else False

        opt_list.append(opt)

    return opt_list

if __name__ == "__main__":

    opt_list = parse_opts_excel()

    for opt in opt_list:
        test_model(opt)
