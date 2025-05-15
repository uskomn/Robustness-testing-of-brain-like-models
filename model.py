import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2=nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3=nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1=nn.Linear(in_features=128*18*18, out_features=512)
        self.fc2=nn.Linear(in_features=512, out_features=1)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.pool2(x)
        x=F.relu(self.conv3(x))
        x=torch.flatten(x, 1)
        in_features = x.size(1)  # 获取当前 x 的特征数
        self.fc1=nn.Linear(in_features=in_features, out_features=512)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

