import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

# 1 定义超参数
BATCH_SIZE = 128 #每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")#决定训练硬件
EPOCHS = 50 #训练轮次

# 2 构建pipeline，主要是对图像做变换
pipeline = transforms.Compose([
    transforms.ToTensor(),#转化为张量
    transforms.Normalize((0.1307,),(0.3081,))#张量标准化（均值，标准差）神秘数据从官方文档抄录
])

# 3 下载、加载数据

# 4 构建网络模型

# 5 定义优化器

# 6 定义训练方法

# 7 定义测试方法

# 8 调用 方法6/7