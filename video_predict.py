# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from statistics import mode
from net.Resnet import resnet18,resnet34
from config import config as c


def video_predict():
    try:
        detection_model_path = 'resource/haarcascade_frontalface_default.xml'
        img_size = c.img_size
        # emotion_labels = {0: '愤怒', 1: '厌恶', 2: '恐惧', 3: '开心', 4: '悲伤', 5: '惊讶', 6: '无表情'}
        emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

        # 加载人脸检测模型
        face_detection = cv2.CascadeClassifier(detection_model_path)

        # 加载模型与权重
        model = resnet34(c.num_class)
        model.load_state_dict(torch.load(c.model_path + '/model_net.pt'))

        device = c.device
        model.to(device)
        model.eval()

        # 调摄像头
        video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.startWindowThread()
        cv2.namedWindow('window_frame')

        while True:
            emotion_window = []
            # 读取一帧
            _, frame = video_capture.read()
            frame = frame.copy()
            frame = cv2.flip(frame, 1)  # 镜像
            # 获得灰度图，并且在内存中创建一个图像对象
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 获取当前帧中的全部人脸
            faces = face_detection.detectMultiScale(gray,1.3,5)
            # 对于所有发现的人脸
            for (x, y, w, h) in faces:
                # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
                cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)

                # 获取人脸图像
                face = gray[y:y+h,x:x+w]

                try:
                    face = cv2.resize(face, (img_size, img_size))
                except:
                    continue

                # 扩充维度，shape变为(1,112,112,1)
                #将（1，112，112，1）转换成为(1,1,112,112)
                face = np.expand_dims(face,0)
                face = np.expand_dims(face,0)
                # 人脸数据归一化，将像素值从0-255映射到0-1之间
                face = face/255.0
                new_face = torch.from_numpy(face)
                new_face = new_face.float().requires_grad_(False)

                # 调用我们训练好的表情识别模型，预测分类
                emotion_arg = torch.argmax(model(new_face.to(device)), dim=-1)
                emotion = emotion_labels[emotion_arg.item()]

                emotion_window.append(emotion)

                try:
                    # 获得出现次数最多的分类
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                # 在矩形框上部，输出分类文字
                cv2.putText(frame,emotion_mode,(x,y-30), font, .7,(255,0,0),1,cv2.LINE_AA)

                # 将图片从内存中显示到屏幕上
                cv2.imshow('window_frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    video_predict()

