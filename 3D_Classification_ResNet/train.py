
import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from ResNet_3D import r3d_18
from datasets_coronal import Tensor3D_Dataset, SPINE_STATUS
from utils import AverageMeter, calculate_accuracy, Logger, get_lr
from glob import glob
import pandas as pd


def train_model(opt, log_list):

    train_logger, train_batch_logger, val_logger, tb_writer = log_list

    train_dataset = Tensor3D_Dataset(opt.dataset_train_dir, opt.datapath_train_list, opt.cache_num)
    val_dataset = Tensor3D_Dataset(opt.dataset_val_dir, opt.datapath_val_list, opt.cache_num)

    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.train_num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.train_num_workers, pin_memory=True)

    model = r3d_18().float()
    if opt.multiGPU:
        model = nn.DataParallel(model)
    model = model.to(opt.device)

    if opt.train_optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.train_initial_lr, weight_decay=opt.train_weight_decay)#, momentum=0.9)
    elif opt.train_optimizer == "SGD":
        if opt.train_nesterov:
            opt.train_dampening = 0.0
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.train_initial_lr, momentum=opt.train_momentum, dampening=opt.train_dampening, weight_decay=opt.train_weight_decay, nesterov=opt.train_nesterov)
    if opt.train_scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=opt.train_plateau_patience, min_lr=0.000001)
    elif opt.train_scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.train_multistep_milestones)

    criterion = nn.CrossEntropyLoss().to(opt.device)

    print(f"Total epoch: {opt.train_epoch}")
    for _epoch in range(opt.train_epoch):
        # training step
        print("\n >>> Train - Step")
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        end_time = time.time()

        current_lr = get_lr(optimizer)
        for idx_batch, data in enumerate(train_loader, 0):
            data_time.update(time.time() - end_time)

            target, class_num, source = data["target"].float().to(opt.device), data["class_num"], data["source"]

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(target)

            #loss = criterion(output, class_num)
            #class_num = torch.eq(torch.transpose(class_num, 0, 1).squeeze(0), 4).long().to(opt.device)
            class_num = torch.transpose(class_num, 0, 1).squeeze(0)
            for class_num_idx, class_num_elmt in enumerate(class_num):
                if SPINE_STATUS[int(class_num_elmt)] in opt.class_normal:
                    class_num[class_num_idx] = 0
                elif SPINE_STATUS[int(class_num_elmt)] in opt.class_abnormal:
                    class_num[class_num_idx] = 1
                else:
                    raise IndexError
            class_num = class_num.long().to(opt.device)
            loss = criterion(output, class_num)

            # accuracy update
            acc = calculate_accuracy(output, class_num)
            losses.update(loss.item(), target.size(0))
            accuracies.update(acc, target.size(0))

            loss.backward()
            optimizer.step()

            # time update
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if train_batch_logger is not None:
                train_batch_logger.log({
                    'epoch': _epoch,
                    'batch': idx_batch + 1,
                    'iter': (_epoch - 1) * len(train_loader) + (idx_batch + 1),
                    'loss': losses.val,
                    'acc': accuracies.val,
                    'lr': current_lr
                })
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(_epoch,
                                                             idx_batch + 1,
                                                             len(train_loader),
                                                             batch_time=batch_time,
                                                             data_time=data_time,
                                                             loss=losses,
                                                             acc=accuracies))
        if train_logger is not None:
            train_logger.log({
                'epoch': _epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': current_lr
            })
        if tb_writer is not None:
            tb_writer.add_scalar('train/loss', losses.avg, _epoch)
            tb_writer.add_scalar('train/acc', accuracies.avg, _epoch)
            tb_writer.add_scalar('train/lr', current_lr, _epoch)


        # validation step
        print("\n >>> Validation - Step")
        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        end_time = time.time()

        with torch.no_grad():
            for idx_batch, data in enumerate(val_loader, 0):
                data_time.update(time.time() - end_time)

                target, class_num, source = data["target"].float().to(opt.device), \
                                            data["class_num"], \
                                            data["source"]

                output = model(target)

                #loss = criterion(output, class_num)
                #class_num = torch.eq(torch.transpose(class_num, 0, 1).squeeze(0), 4).long().to(opt.device)
                class_num = torch.transpose(class_num, 0, 1).squeeze(0)
                for class_num_idx, class_num_elmt in enumerate(class_num):
                    if SPINE_STATUS[int(class_num_elmt)] in opt.class_normal:
                        class_num[class_num_idx] = 0
                    elif SPINE_STATUS[int(class_num_elmt)] in opt.class_abnormal:
                        class_num[class_num_idx] = 1
                    else:
                        raise IndexError
                class_num = class_num.long().to(opt.device)
                loss = criterion(output, class_num)
                #val_loss = loss

                # accuracy update
                acc = calculate_accuracy(output, class_num)
                losses.update(loss.item(), target.size(0))
                accuracies.update(acc, target.size(0))

                # time update
                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    _epoch,
                    idx_batch + 1,
                    len(val_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies))
        if val_logger is not None:
            val_logger.log({'epoch': _epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        if tb_writer is not None:
            tb_writer.add_scalar('val/loss', losses.avg, _epoch)
            tb_writer.add_scalar('val/acc', accuracies.avg, _epoch)
        if opt.train_scheduler == "ReduceLROnPlateau":
            scheduler.step(losses.avg)
        elif opt.train_scheduler == "MultiStepLR":
            scheduler.step()

        if (_epoch + 1) % opt.train_checkpoint == 0:
            ckpt_path = opt.train_save_path + f"/ckpt_{_epoch}.pt"
            ckpt = {
                "epoch": _epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            if opt.train_scheduler is not None:
                ckpt["scheduler"] = scheduler.state_dict()
            torch.save(ckpt, ckpt_path)

def datalist_from_yolo(text_path: str, temp_dataset_path: str, class_normal: list, class_abnormal: list):

    data_list = []
    for target_line in open(text_path, "r").readlines():
        target_id = target_line.split("\\")[-3]
        if target_id not in data_list:
            data_list.append(target_id)

    target_list = []
    for target_path in glob(temp_dataset_path + "/*.pt"):
        target_3D = torch.load(target_path)
        class_num, source = target_3D["class_num"], target_3D["source"]

        if (SPINE_STATUS[int(class_num)] not in class_normal) and (
                SPINE_STATUS[int(class_num)] not in class_abnormal):
            continue

        # print(source[0])
        if source[0] in data_list:
            target_list.append(target_path)

    return target_list

def parse_opts_excel(opt):

    f_excel = pd.read_excel(opt.excel_path, sheet_name="Sheet1")
    f_excel = f_excel.fillna("")
    #print(f_excel["case"][0])
    opt_list = []
    for excel_row, excel_case in enumerate(f_excel["case"]):
        if excel_case == "":
            break

        #parser = argparse.ArgumentParser()
        #opt = parser.parse_args()

        opt.device = f_excel["device"][excel_row]
        opt.multiGPU = False

        opt.dataset_train_dir = f_excel["train_dir"][excel_row]
        opt.dataset_val_dir = f_excel["val_dir"][excel_row]
        opt.cache_num = f_excel["cache_num"][excel_row]

        opt.class_normal = [x.strip() for x in f_excel["class_normal"][excel_row].split(",")]
        opt.class_abnormal = [x.strip() for x in f_excel["class_abnormal"][excel_row].split(",")]
        if f_excel["train_yolo_txt"][excel_row] == "":
            opt.datapath_train_list = None
        else:
            opt.datapath_train_list = datalist_from_yolo(f_excel["train_yolo_txt"][excel_row], f_excel["train_dir"][excel_row], opt.class_normal, opt.class_abnormal)
        if f_excel["val_yolo_txt"][excel_row] == "":
            opt.datapath_val_list = None
        else:
            opt.datapath_val_list = datalist_from_yolo(f_excel["val_yolo_txt"][excel_row], f_excel["val_dir"][excel_row], opt.class_normal, opt.class_abnormal)

        opt.train_save_path = f_excel["train_save_path"][excel_row] + f"/{int(excel_case)}"
        opt.train_epoch = int(f_excel["train_epoch"][excel_row])
        opt.train_batch_size = int(f_excel["train_batch_size"][excel_row])
        opt.train_num_workers = int(f_excel["train_num_workers"][excel_row])
        opt.train_initial_lr = float(f_excel["train_initial_lr"][excel_row])
        opt.train_checkpoint = int(f_excel["train_checkpoint"][excel_row])
        opt.train_optimizer = f_excel["train_optimizer"][excel_row]
        opt.train_momentum = float(f_excel["train_momentum"][excel_row])
        opt.train_dampening = float(f_excel["train_dampening"][excel_row])
        opt.train_weight_decay = float(f_excel["train_weight_decay"][excel_row])
        opt.train_nesterov = True if f_excel["train_nesterov"][excel_row] == "True" else False
        opt.train_scheduler = None if f_excel["train_scheduler"][excel_row] == "None" else f_excel["train_scheduler"][excel_row]
        opt.train_plateau_patience = int(f_excel["train_plateau_patience"][excel_row])
        opt.train_multistep_milestones = [eval(step) for step in f_excel["train_multistep_milestones"][excel_row].split(",")]

        #print(opt)
        print("I was here, too.")
        #break

        opt_list.append(opt)
        #break

    #return parser.parse_args()
    #return opt
    return opt_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", default="train_sheet.xlsx", type=str, help="excel sheet of train list")
    main_opt = parser.parse_args()

    #opts_excel_path = "train_sheet.xlsx"
    opt_list = parse_opts_excel(main_opt)
    #print(opt_list)

    for opt in opt_list:
        if not os.path.isdir(opt.train_save_path):
            os.makedirs(opt.train_save_path)

        train_logger = Logger(opt.train_save_path + "/train.log", ["epoch", "loss", "acc", "lr"])
        train_batch_logger = Logger(opt.train_save_path + "/train_batch.log", ["epoch", "batch", "iter", "loss", "acc", "lr"])
        val_logger = Logger(opt.train_save_path + "/val.log", ["epoch", "loss", "acc"])
        tb_writer = SummaryWriter(log_dir=opt.train_save_path)
        log_list = [train_logger, train_batch_logger, val_logger, tb_writer]

        train_model(opt, log_list)

    exit(0)
'''
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.multiGPU = False
    #opt.device = "cpu"
    print(opt.device)

    #opt.dataset_target_dir = "C:/Users/SNUBH/SP_work/Python_Project/yolov3_DICOM/data/billion/bone_coronal_20210914"
    #opt.dataset_train_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp/train"
    opt.dataset_train_dir = "Y:/SP_work/billion/ubuntu/Python_Project/3D_Classification_ResNet/temp/train"
    #opt.dataset_val_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp/val"
    opt.dataset_val_dir = "Y:/SP_work/billion/ubuntu/Python_Project/3D_Classification_ResNet/temp/val"
    opt.datapath_train_list = datalist_from_yolo("C:/Users/SNUBH/SP_work/Python_Project/yolov3_DICOM/data/billion/bone_coronal_20210914/train_1.txt", "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_new_256")
    opt.datapath_val_list = datalist_from_yolo("C:/Users/SNUBH/SP_work/Python_Project/yolov3_DICOM/data/billion/bone_coronal_20210914/val_1.txt", "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_new_256")

    opt.train_save_path = "./results/train_20211021"
    opt.train_epoch = 300
    opt.train_batch_size = 4
    opt.train_num_workers = 0
    opt.train_initial_lr = 0.01
    opt.train_checkpoint = 10
    opt.train_optimizer = "SGD"
    opt.train_momentum = 0.9
    opt.train_dampening = 0.0
    opt.train_weight_decay = 1e-3
    opt.train_nesterov = False
    opt.train_scheduler = None
    opt.train_plateau_patience = 10
    opt.train_multistep_milestones = [50, 100, 150]
'''




