import os
import pickle
import face_recognition as fr

#读取所有数据集文件
images_root = "D:\DataSet/SCUT-FBP5500_v2/Images"
labels_file = "D:\DataSet/SCUT-FBP5500_v2/train_test_files/All_labels.txt"

face_list = []
with open(labels_file, encoding = "utf8") as fi:
    for line in fi:
        line = line.strip() #过滤掉txt每行两边多余空格字符
        filename, score = line.split(" ") #按照字符串之间的空格拆分字符
        full_path = os.path.join(images_root, filename) #拼接路径并全部返回
        if not os.path.exists(full_path):
            print("Error: image file not found: ", full_path)
            continue
        image = fr.load_image_file(full_path)
        assert image is not None
        encs = fr.face_encodings(image) #利用开源库face_recognition返回图像中每个面的128维人脸编码
        if len(encs) != 1:
            print("%s has %d faces." % (filename, len(encs)))
            continue

        item = {
            'enc': encs[0],
            'score': float(score)
        }
        face_list.append(item)

#把标签数据写入pkl文件
print("Total faces: %d." % len(face_list))
with open("training.pkl", "wb") as fo:
    pickle.dump(face_list, fo)