import numpy as np
import pandas as pd
import cv2
import torch
import os
import random
import torch.utils.data as data
from config import config as c



class FaceDataset(data.Dataset):
    # 初始化
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        df_path = pd.read_csv(root + '/dataset.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '/dataset.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]
        self.img_size = c.img_size

    # 读取图片，item为索引号
    def __getitem__(self, item):
        # 图像数据用于训练，需为tensor类型，label用numpy或list均可
        face = cv2.imread(c.images + '/' + self.path[item],0)
        face = cv2.resize(face,(self.img_size,self.img_size))
        # 50% 概率均值模糊
        if random.random() > 0.5:
            face = cv2.medianBlur(face, 9)
        # 50% 概率翻转
        if random.random() > 0.5:
            face = cv2.flip(face, 1)
        # 直方图均衡化
        face_hist = cv2.equalizeHist(face)
        """
        像素值标准化
        读出的数据是112*112的，而后续卷积神经网络中nn.Conv2d() API所接受的数据格式是(batch_size, channel, width, higth)，
        本次图片通道为1，因此我们要将112*112 reshape为1*112*112。
       """
        face_normalized = face_hist.reshape(1, self.img_size, self.img_size) / 255.0
        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')
        label = self.label[item]
        return face_tensor, label

    # 获取数据集样本个数
    def __len__(self):
        return self.path.shape[0]


def data_annotation(train_path,val_path):
    # 读取label文件
    df_label = pd.read_csv(c.root + '/labels.csv', header=None)
    # 查看该文件夹下所有文件
    files_dir = os.listdir(c.images)

    # 用于存放图片名
    train_data_list = []
    val_data_list = []
    # 用于存放图片对应的label
    train_label_list = []
    val_label_list = []

    # 遍历该文件夹下的所有文件
    for file_dir in files_dir:
        # 如果某文件是图片，则将其文件名以及对应的label取出
        if os.path.splitext(file_dir)[1] == ".jpg":
            if random.random() < 0.9:
                train_data_list.append(file_dir)
                index = int(os.path.splitext(file_dir)[0])
                train_label_list.append(df_label.iat[index, 0])
            else:
                val_data_list.append(file_dir)
                index = int(os.path.splitext(file_dir)[0])
                val_label_list.append(df_label.iat[index, 0])

    # 将训练数据写进dataset.csv文件
    train_data = pd.Series(train_data_list)
    train_label = pd.Series(train_label_list)
    train_df = pd.DataFrame()
    train_df['train_data'] = train_data
    train_df['train_label'] = train_label
    train_df.to_csv(train_path + '/dataset.csv', index=False, header=False)

    # 将测试数据写进dataset.csv文件
    val_data = pd.Series(val_data_list)
    val_label = pd.Series(val_label_list)
    val_df = pd.DataFrame()
    val_df['val_data'] = val_data
    val_df['val_label'] = val_label
    val_df.to_csv(val_path + '/dataset.csv', index=False, header=False)

    print("数据标注完毕...")

