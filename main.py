# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from config import config as c
from process import FaceDataset, data_annotation
from net.ResNet import resnet18


device = c.device

def train(train_data, val_data, batch_size, epochs, num_class, learning_rate, wt_decay):
    # 给图像标注上标签，分割训练测试集
    data_annotation(c.train_data, c.val_data)
    # 数据集实例化(创建数据集)
    train_dataset = FaceDataset(train_data)
    val_dataset = FaceDataset(val_data)

    print("num_train_data:", len(train_dataset), "num_val_data:", len(val_dataset))
    # 载入网络
    # model = FaceCNN(num_class)
    # 残差网络
    model = resnet18(num_class)
    model.to(device)

    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 逐轮训练
    acc_best = 0
    for epoch in range(epochs):
        # 载入数据并分割batch
        train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size, shuffle=True)
        # 打印acc和loss的步长
        logs = 0
        # scheduler.step() # 学习率衰减
        model.train()  # 模型训练
        for images, labels in train_loader:
            logs += 1
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            output = model.forward(images)
            # 误差计算
            loss = loss_function(output, labels)

            if logs % 50 == 0 or logs == len(train_loader):
                output = torch.argmax(output, dim=-1)
                pred_acc = torch.mean(torch.eq(output, labels).float()).item()
                print('epoch={}, step={}, loss={:.3f}, pred acc={:.3f}'.format(epoch, logs, loss.item(), pred_acc))

            # 梯度清零
            optimizer.zero_grad()
            # 误差的反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

        print('===================================')
        model.eval()  # 模型评估
        acc_val = validate(model, val_loader)
        print('After {} epochs , the acc_val is :{:.6f} '.format(epoch, acc_val))
        print('===================================')

        if acc_val >= acc_best:
            # 保存模型
            torch.save(model.state_dict(), c.model_save_path + '/model_net.pth')


def validate(model, val_loader):
    # result, num = 0.0, 0
    pre_label = []
    true_label = []
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        # 模型预测
        pred = model.forward(images)

        #取最大元素下标得到结果
        output = torch.argmax(pred, dim=-1)
        pre_label.append(output)
        true_label.append(labels)

    # 预测标签对比正确标签得到准确率
    acc = torch.mean(torch.eq(torch.cat(pre_label), torch.cat(true_label)).float()).item()

    return acc


if __name__ == '__main__':
    # 超参数可自行指定
    train(c.train_data, c.val_data, c.batch_size, c.epoch, c.num_class, c.learning_rate, c.wt_decay)


