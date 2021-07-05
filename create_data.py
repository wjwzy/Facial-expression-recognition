import os
import cv2
import pandas as pd
import numpy as np
from config import config as c


def create_folder():
    # 创建目录
    root = c.root
    if not os.path.exists(root):
        os.mkdir(root)

    images = c.images
    if not os.path.exists(images):
        os.mkdir(images)

    train_data = c.train_data
    if not os.path.exists(train_data):
        os.mkdir(train_data)

    val_data = c.val_data
    if not os.path.exists(val_data):
        os.mkdir(val_data)

    model_save_path = c.model_save_path
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    model_path = c.model_path
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print("目录创建完成！")

    # 将label和像素数据分离
    data_division(c.root + '/facial_data.csv', c.root + '/labels.csv', c.root + '/images.csv', )

    # 将像素整合成图像
    write_images(c.root + '/images.csv', c.images)



def data_division(path, label_save_path, image_save_path):

    path = path  # 原数据路径
    # 读取数据
    df = pd.read_csv(path)
    # 提取label数据
    df_y = df[['label']]
    # 提取feature（即像素）数据
    df_x = df[['feature']]
    # 将label写入label.csv
    df_y.to_csv(label_save_path, index=False, header=False)
    # 将feature数据写入data.csv
    df_x.to_csv(image_save_path, index=False, header=False)
    print("表格分割完成：{}".format(c.root))


def write_images(path, save_path):
    # 读取像素数据
    data = np.loadtxt(path)
    # 按行取数据
    for i in range(data.shape[0]):
        face_array = data[i, :].reshape((48, 48))  # reshape
        cv2.imwrite(save_path + '/{}.jpg'.format(i), face_array)  # 写图片
        print(i)

    print("图片写入完成：{}".format(save_path))

if __name__ == '__main__':
    create_folder()
