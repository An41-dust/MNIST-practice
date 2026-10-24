import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

# 超参数
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25

# 管道
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)) #张量标准化（均值，标准差）神秘数据从官方文档抄录
])

# 数据集
##下载
import os
import gzip
import urllib.request

URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
} #pytorch镜像源

def download(root="./dataset/raw"):
    os.makedirs(root, exist_ok=True)
    for name, url in URLS.items():
        gz_path = os.path.join(root, name)
        out_path = gz_path.removesuffix(".gz")
        if os.path.exists(out_path):
            continue
        if not os.path.exists(gz_path):
            print(f"Download {name} ...")
            urllib.request.urlretrieve(url, gz_path)
        print(f"Unzip {name} ...")
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())

##解析
import struct
import numpy as np

def read_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"images magic ERROR: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
    return data

def read_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8)) #尽管没有调用num，但为了从8字节之后读取，必须定义num
        assert magic == 2049, f"labels magic ERROR: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

##自定义dataset
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root="./dataset/raw", train=True):
        if train:
            img_file = os.path.join(root, "train-images-idx3-ubyte")
            lbl_file = os.path.join(root, "train-labels-idx1-ubyte")
        else:
            img_file = os.path.join(root, "t10k-images-idx3-ubyte")
            lbl_file = os.path.join(root, "t10k-labels-idx1-ubyte")

        self.images = read_images(img_file)
        self.labels = read_labels(lbl_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        img = pipeline(img)
        return img, label

##加载
from torch.utils.data import DataLoader

download()

train_dataset = MNISTDataset(train=True)
test_dataset = MNISTDataset(train=False)

train_loader = DataLoader(train_dataset,BATCH_SIZE,shuffle=True,num_workers=0)
test_loader = DataLoader(test_dataset,BATCH_SIZE,shuffle=False,num_workers=0)
#DataLoader时调用了魔法方法__len__

for images,labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break
#for循环迭代时调用了魔法方法__getitem__

# 神经网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5,padding=2) #二层卷积 1→32
        self.conv2 = nn.Conv2d(32,64,3,padding=1) #→64
        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.fc1 = nn.Linear(64*7*7,256) #→256
        self.drop = nn.Dropout(0.5) #丢失数据，防止过拟合
        self.fc2 = nn.Linear(256,10) #→10

    def forward(self,x):
        x = F.relu(self.conv1(x)) #激活函数
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return F.log_softmax(x,dim=1) #输出为概率

# 优化器
model = Digit().to(DEVICE)
optimizer = optim.Adam(model.parameters())

# 训练方法
def train_model(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_index,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
    print(f"Train Epoch : {epoch} \t Loss : {loss.item():.6f}")

# 测试方法
def test_model(model,device,test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output,target,reduction='sum').item()
            pred = output.max(1,keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"Test Average Loss : {test_loss:.4f},Accuracy : {accuracy:.3f}")
    return accuracy

# 运行并保存
os.makedirs("./model",exist_ok=True)
best_acc = 0.0

for epoch in range(1,EPOCHS + 1):
    train_model(model,DEVICE,train_loader,optimizer,epoch)
    acc = test_model(model,DEVICE,test_loader)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(),"./model/best_mnist_cnn.pth")
    print(f"Best acc={best_acc:.3f}")

print(f"Done. Best acc: {best_acc:.3f}, saved to ./model/best_mnist_cnn.pth")