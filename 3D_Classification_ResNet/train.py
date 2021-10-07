
import os
import time
import argparse
import hashlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from ResNet_3D import r3d_18
from datasets_coronal import Tensor3D_Dataset
from utils import AverageMeter, calculate_accuracy, Logger, get_lr
from glob import glob

#from sklearn import metrics
#import matplotlib.pyplot as plt
#from scipy import interpolate

CACHE_LIST = []
CACHE_HASH = {}

def train_model(opt, log_list):

    train_logger, train_batch_logger, val_logger, tb_writer = log_list

    '''
    target_dataset = Dicom3D_Coronal_Dataset(
        data_dir=opt.dataset_target_dir,
        dicom_dir=opt.dataset_dicom_dir,
        label_dir=opt.dataset_label_dir,
        dicom_HU_level=opt.dataset_dicom_HU_level,
        dicom_HU_width=opt.dataset_dicom_HU_width,
        cache_num=opt.dataset_cache_num,
        temp_savepath=opt.dataset_temp_savepath,
        isTest=opt.dataset_isTest,
        transform=transforms.Compose([
            ToTensor3D(),
            Rescale3D((64, 128, 128))
        ])
    )
    target_dataset_size = len(target_dataset)
    val_dataset_size = int(target_dataset_size * opt.dataset_val_ratio)
    train_dataset_size = target_dataset_size - val_dataset_size
    train_dataset, val_dataset = torch.utils.data.random_split(target_dataset, [train_dataset_size, val_dataset_size])
    '''

    train_dataset = Tensor3D_Dataset(opt.dataset_train_dir, opt.dataset_cache_num)
    #train_dataset = Tensor3D_Dataset(opt.dataset_train_dir, opt.dataset_cache_num, opt.CACHE_LIST, opt.CACHE_HASH)
    val_dataset = Tensor3D_Dataset(opt.dataset_val_dir, opt.dataset_cache_num)
    #val_dataset = Tensor3D_Dataset(opt.dataset_val_dir, opt.dataset_cache_num, opt.CACHE_LIST, opt.CACHE_HASH)

    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.train_num_workers, pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.train_num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.train_num_workers, pin_memory=True)

    model = r3d_18().float()
    if opt.multiGPU:
        model = nn.DataParallel(model)
    model = model.to(opt.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.train_initial_lr)#, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=opt.train_plateau_patience, min_lr=0.000001)

    criterion = nn.CrossEntropyLoss().to(opt.device)

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

            target, class_num, source = data["target"].float().to(opt.device),\
                                        data["class_num"].long().to(opt.device),\
                                        data["source"]

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(target)
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
                                            data["class_num"].long().to(opt.device), \
                                            data["source"]

                output = model(target)
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
        scheduler.step(losses.avg)

        if (_epoch + 1) % opt.train_checkpoint == 0:
            ckpt_path = opt.train_save_path + f"/ckpt_{_epoch}.pt"
            ckpt = {
                "epoch": _epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }
            torch.save(ckpt, ckpt_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.multiGPU = False
    #opt.device = "cpu"
    print(opt.device)

    #opt.dataset_target_dir = "C:/Users/SNUBH/SP_work/Python_Project/yolov3_DICOM/data/billion/bone_coronal_20210914"
    #opt.dataset_train_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp/train"
    opt.dataset_train_dir = "Y:/SP_work/billion/ubuntu/Python_Project/3D_Classification_ResNet/temp/train"
    #opt.dataset_val_dir = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp/val"
    opt.dataset_val_dir = "Y:/SP_work/billion/ubuntu/Python_Project/3D_Classification_ResNet/temp/val"

    opt.dataset_dicom_dir = "/images"
    opt.dataset_label_dir = "/labels_yolo"
    opt.dataset_dicom_HU_level = 300
    opt.dataset_dicom_HU_width = 2500
    opt.dataset_cache_num = 2000
    #opt.dataset_cache_num = 0
    opt.dataset_temp_savepath = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp"
    opt.dataset_isTest = True
    opt.dataset_val_ratio = 0.1
    opt.dataset_kfold = 1

    opt.train_save_path = "./results/train_test"
    opt.train_epoch = 1000
    opt.train_batch_size = 6
    opt.train_num_workers = 0
    opt.train_initial_lr = 0.1
    opt.train_plateau_patience = 10
    opt.train_checkpoint = int(opt.train_epoch / 100)
    if not os.path.isdir(opt.train_save_path):
        os.makedirs(opt.train_save_path)

    train_logger = Logger(opt.train_save_path + "/train.log", ["epoch", "loss", "acc", "lr"])
    train_batch_logger = Logger(opt.train_save_path + "/train_batch.log", ["epoch", "batch", "iter", "loss", "acc", "lr"])
    val_logger = Logger(opt.train_save_path + "/val.log", ["epoch", "loss", "acc"])
    tb_writer = SummaryWriter(log_dir=opt.train_save_path)
    log_list = [train_logger, train_batch_logger, val_logger, tb_writer]

    '''
    pt_list = glob(opt.dataset_train_dir + "/*.pt")
    for target_idx, target_pt in enumerate(pt_list):
        print(f"[{target_idx + 1}/{len(pt_list)}]Caching: {target_pt}")
        cache_key = hashlib.blake2b(target_pt.encode("utf-8"), digest_size=64).hexdigest()
        cache_idx = len(CACHE_LIST)
        CACHE_HASH[cache_key] = cache_idx
        CACHE_LIST.insert(cache_idx, torch.load(target_pt))
    opt.CACHE_LIST = CACHE_LIST
    opt.CACHE_HASH = CACHE_HASH
    '''

    train_model(opt, log_list)



