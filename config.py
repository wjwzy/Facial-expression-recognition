__all__ = ['config', ] # import 可以导入的接口

class DictObj(object):
    # 私有变量是map
    # 设置变量的时候 初始化设置map
    def __init__(self, mp):
        self.map = mp
        # print(mp)

# set 可以省略 如果直接初始化设置
    def __setattr__(self, name, value):
        if name == 'map':# 初始化的设置 走默认的方法
            object.__setattr__(self, name, value)
            return
        self.map[name] = value
# 之所以自己新建一个类就是为了能够实现直接调用名字的功能。
    def __getattr__(self, name):
        return  self.map[name]


config = DictObj({
    'root': '../facial_data', # 主目录
    'images': '../facial_data/images', # 图像路径
    'train_data': '../facial_data/train_data', # 训练数据路径
    'val_data': '../facial_data/val_data', # 校验数据路径
    'model_save_path':'../facial_data/save_model', # 训练时保存的模型
    'model_path':'../facial_data/model', # 预测时模型文件的路径
    'img_size': 112, # 图像大小
    'num_class' : 7, # 类别数
    'batch_size': 32, # 批量大小
    'epoch' : 10, # 训练循环次数
    'learning_rate' : 0.0005, # 优化器学习率
    'wt_decay' : 0.0001, # 权重衰退
    'device' : 'cuda:0', # 指定的GPU（cpu、0、1...）
})