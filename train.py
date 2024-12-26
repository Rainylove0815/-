import copy
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from model import AlexNet


def Get_Image_Size(in_size):
    # get Input-Imagee size
    width, high = 224, 224
    if isinstance(in_size, int):
        width = in_size
        high = in_size
    elif isinstance(in_size, tuple):
        width = in_size[0]
        high = in_size[1]
    return width, high


def Get_Batch_Size_For_TrainVal(batch):
    # get batch size
    train_batch, val_batch = 32, 16
    if isinstance(batch, int):
        train_batch = batch
        val_batch = batch
    elif isinstance(batch, tuple):
        train_batch = batch[0]
        val_batch = batch[1]
    return train_batch, val_batch


# 加载数据集
def Data_Loading(root, in_size, batch):
    width, high = Get_Image_Size(in_size)
    train_batch, val_batch = Get_Batch_Size_For_TrainVal(batch)
    # transform configuration
    data_transform = {
        "train": transforms.Compose([transforms.Resize((width, high)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((width, high)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }
    # Image File
    train_path = os.path.join(root, "train")
    val_path = os.path.join(root, "val")
    # Dataset
    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transform['val'])
    # Loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True, num_workers=0,
                              drop_last=True)
    val_loader = DataLoader(dataset=train_dataset, batch_size=val_batch, shuffle=False, num_workers=0, drop_last=True)
    # return variable
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def Train_Process(model, Train_DataLoader, Val_DataLoader, Learning_Rate, epoch_num, pth_name):
    # gpu/cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Learning_Rate)
    # Loss-交叉熵损失
    Loss = nn.CrossEntropyLoss()
    # 模型指认到设备中
    model.to(device)
    # Copy current model param
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 保存当前时间
    Time_Start = time.time()

    for epoch in range(epoch_num):
        print("epoch{}/{}".format(epoch, epoch_num - 1))

        # initialize parameter
        train_loss = 0.0
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0

        train_num = 0
        val_num = 0

        for step, (feature, label) in enumerate(Train_DataLoader):
            feature = feature.to(device)
            label = label.to(device)
            # set train model
            model.train()

            # 前向传播过程中输出一个batch输入一个batch中对应的预测
            output = model(feature)
            # 查找每一行中概率最大的结果
            predict = torch.argmax(input=output, dim=1)
            # 通过预测和标签计算损失函数值,每个batch的损失
            loss = Loss(output, label)

            # 将梯度初始化为0
            optimizer.zero_grad()

            # 反向传播
            loss.backward()
            # 更新网络参数
            optimizer.step()

            train_loss += loss.item() * feature.size(0)
            train_corrects += torch.sum(predict == label.data)
            train_num += feature.size(0)

            # print train process
            rate = (step + 1) / len(Train_DataLoader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            # print("Training:\n")
            print("\rTraining {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        for step, (feature, label) in enumerate(Val_DataLoader):
            # evaluate
            model.eval()
            feature = feature.to(device)
            label = label.to(device)
            output = model(feature)

            predict = torch.argmax(input=output, dim=1)
            loss = Loss(output, label)
            val_loss += loss.item() * feature.size(0)
            val_corrects += torch.sum(predict == label)
            val_num += feature.size(0)
            # print train process
            rate = (step + 1) / len(Val_DataLoader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            # print('Validate:\n')
            print("\rValidating: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train acc:{:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss:{:.4f} Val acc:{:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            # 保持当前参数
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - Time_Start
        print('训练耗时{:.1f}min{:.1f}s/epoch'.format(time_use // 60, time_use % 60))

    # 选择最优模型保存-加载最高准确率下的参数
    model.load_state_dict(best_model_wts)
    root_path = os.getcwd()
    torch.save(obj=best_model_wts,
               f=os.path.join(root_path, 'pth_save', pth_name))

    train_process = pd.DataFrame(data={'epoch': range(epoch_num),
                                       'train_loss_all': train_loss_all,
                                       'val_loss_all': val_loss_all,
                                       'train_acc_all': train_acc_all,
                                       'val_acc_all': val_loss_all, })
    return train_process


def plot(train_process, Title, Save_Path):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'ro-', label='train loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('train-loss')

    plt.subplot(2, 2, 3)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'bs-', label='Val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('val-loss')

    plt.subplot(2, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'ro-', label='train acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('train-acc')

    plt.subplot(2, 2, 4)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'bs-', label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('val-acc')

    # plt.title(Title)
    # plt.show()
    plt.savefig(Save_Path)


if __name__ == "__main__":

    root_path = os.getcwd()

    net = AlexNet()

    train_dataloader, val_dataloader, train_num, val_num = Data_Loading(root=os.path.join(root_path, 'data_enhance'),in_size=(227, 227), batch=(32, 32))
    train_process = Train_Process(model=net, Train_DataLoader=train_dataloader, Val_DataLoader=val_dataloader,Learning_Rate=0.0001, epoch_num=10, pth_name='AlexModel_test.pth')
    plot(train_process=train_process, Title='lr=0.0001 epoch=10',Save_Path='plt/res.png')