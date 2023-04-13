from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}") # 查看medmnist版本

data_flag = 'organmnist3d' # 这里选择肠道数据集

download = True # 默认下载

num_epochs = 100 # 训练轮数
batch_size = 128 # 批数
lr = 1e-3 # 学习速率

info = INFO[data_flag] # 查看数据集信息
task = info['task'] # 查看数据集task
n_channels = info['n_channels'] # 查看数据集的通道
n_classes = len(info['label']) # 查看数据集总共的标签的数量

data_class = getattr(medmnist, info['python_class'])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [.5], std = [.5]),
]) # 对数据进行变化，先转为tensor(), 再进行正则化

train_dataset = data_class(split = 'train', transform = data_transform, download = download)
test_dataset = data_class(split = 'test', transform = data_transform, download = download)

pil_dataset = data_class(split = 'train', download = download)

train_loader = data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
train_loader_at_eval = data.DataLoader(dataset = train_dataset, batch_size = 2 * batch_size, shuffle = False)
test_loader = data.DataLoader(dataset = test_dataset, batch_size = 2 * batch_size, shuffle = False)
# 对训练集，验证集，测试集进行加载

print(train_dataset)
print("****************************************")
print(test_dataset)
# 打印训练集，测试卷结构

res = [0] * 11
for i in range(len(train_dataset.labels)):
    res[train_dataset.labels[i][0]] += 1
print(res)
img1 = train_dataset.montage(length=10)

img1[10].show()
# 查看数据集一部分，取40张

# 对网络进行定义
# 与手写数据集的网络定义相似，加入batchnorm可以时模型泛化能力变强和加快收敛的速度
class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1) # 将张量平铺后输入线性层
        x = self.fc(x)
        return x

model = Net(in_channels = n_channels, num_classes = n_classes).cuda()
# 将模型送入GPU

# 当task时二分类时使用BCELoss，因为二分类此损失函数效果好，交叉熵用于多分类
if task == 'multi-label, binary-class':
    criterion = nn.BCEWithLogitsLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss().cuda()

optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

# 定义优化器和开始训练

for epochs in range(num_epochs):
    model.train() # 训练模式
    for inputs, targets in tqdm(train_loader):
        inputs = inputs.cuda() # 将参数送入gpu
        targets = targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long() # 将target压缩维度为1的维度再将张量转换为long型
            loss = criterion(outputs, targets) # 放入交叉熵

        loss.backward() # 反向传播
        optimizer.step()

def test(split):
    model.eval().cuda() # 模型评估模式
    y_score = torch.tensor([]).cuda()
    y_true = torch.tensor([]).cuda()

    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim = -1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim = -1) # 取最后一维
                targets = targets.float().resize_(len(targets), 1) #修改tensor和其的shape

            y_score = torch.cat((y_score, outputs), 0) # 拼接张量
            y_true = torch.cat((y_true, targets), 0)

        y_true = y_true.cpu()
        y_score = y_score.cpu()
        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score) # 使用medmnist的库函数进行评估

        print('%s   auc:%.5f      acc:%.5f   ' % (split, *metrics))

test('train') # 展示验证集的auc和acc
test('test') # 展示测试集的auc和acc