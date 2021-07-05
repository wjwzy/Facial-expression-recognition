## 项目是基于pytorch框架下开发的人脸表情识别算法

通过opencv调取摄像头对每一帧图像进行人脸定位、裁剪，再提取特征分类识别；
项目设计了的是Net目录下的Resnet.py，是18层的残差网络，输入的shape为1×112×112。


## 环境需求

可参照requirements.txt文件对比

```
pandas==0.24.0
torch==1.7.0+cu101
opencv_python==4.5.1.48
numpy==1.19.4
```

## 使用方法
### 一、准备数据集及相关配置
数据是data目录下的facial_data.csv文件，文件中，写有图像和对应的标签；

类别：{0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

config.py是配置文件，里面写了目录以及训练预测需要的配置项。

```
运行create_data.py，会创建存放数据以及模型的目录；
并且会将facial_data.csv文件中的图像和标签分开得到data.csv、label.csv，下标是两者的关联；
最后读取data.csv文件中的像素，保存成以下标命名的图像。
```

### 二、训练
运行main.py，运行时首先会调用process.py中的data_label方法，将数据分割成训练集和校验集，并将图像与label.csv文件通过下标进行标注；

具体预处理操作在process.py中；

```
python main.py
```


### 三、视频预测
视频预测时用到的人脸定位，人脸库在resource目录下的haarcascade_frontalface_default.xml
```
python video_predict.py
```
