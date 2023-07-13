import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 首先读取数据
# 分别构建训练集和测试集

# 定义超参数
input_size = 28  # 图像的总尺寸是28*28
num_class = 10
num_epoch = 3
batch_size = 64
# 训练集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
# 测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
# 构建batch数据
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU()
        )
        self.out = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # view 相当与 reshape操作 x的维度是 batch,channel,height,weight -1自动计算
        output = self.out(x)
        return output


# 准确率作为评估标准
"""
在给定的代码中，`view_as()`是一个PyTorch张量方法，它用于返回一个具有与给定张量相同大小的新张量，但具有不同的视图。`view_as()`方法通常用于将一个张量的形状调整为与另一个张量相匹配。

在这种情况下，`labels.data`是一个张量，`pred`是另一个张量。`view_as(pred)`的目的是将`labels.data`张量的形状调整为与`pred`张量的形状相匹配。这样做是为了确保两个张量在进行比较时具有相同的形状。

具体来说，在比较`pred`和`labels.data.view_as(pred)`时，`pred.eq()`方法用于比较两个张量的元素是否相等，返回一个具有相同形状的布尔值张量。然后，`.sum()`方法用于计算布尔值张量中为`True`的元素的数量，即预测正确的数量。
"""


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, dim=1)[1]  # 第一个“1”表示的是维度 0是横着的 1是竖着的
    right = pred.eq(labels.data.view_as(pred)).sum()  # 第二个‘1’ 是因为max中返回2个值 一个是最大值的值，另一个是位置 我们只需要知道
    return right, len(labels)


# 训练模型

net = CNN()

# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(
    params=net.parameters(),
    lr=0.001
)
# 开始训练循环
for epoch in range(num_epoch):
    train_rights=[]
    for batch_idx,(data,target) in enumerate(train_loader):
        net.train()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = accuracy(output, target)
        train_rights.append(right)

        if batch_idx % 100 == 0:

            net.eval()
            val_rights = []

            for (data, target) in test_loader:
                output = net(data)
                right = accuracy(output, target)
                val_rights.append(right)

            # 准确率计算
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].numpy() / train_r[1],
                       100. * val_r[0].numpy() / val_r[1]))