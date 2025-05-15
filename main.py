import torch
import os
import  pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from model import ResNet,CNN


class BrainTumorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, dtype=str).applymap(lambda x: x.strip() if isinstance(x, str) else x)
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = {"tumor": 1, "normal": 0}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 1]
        class_name=self.annotations.iloc[idx, 2]
        sub_str="Brain Tumor" if class_name=="tumor" else "Healthy"
        img_path = os.path.join(self.root_dir, sub_str, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.class_map[self.annotations.iloc[idx, 2]]

        if self.transform:
            image = self.transform(image)

        return image, label


transform=transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)])
dataSet=BrainTumorDataset(csv_file='./data/metadata.csv',root_dir='./data/Brain Tumor Data Set',transform=transform)

train_length=int(0.8*len(dataSet))
test_length=len(dataSet)-train_length
train_dataSet,test_dataSet=random_split(dataSet,[train_length,test_length])

train_loader=DataLoader(train_dataSet,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataSet,batch_size=32,shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=ResNet().to(device)
criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

epoches=5
for epoch in range(epoches):
    model.train()
    train_loss=0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device).view(-1,1)
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
    print(f"Epoch {epoch+1}/{epoches}, Training Loss: {train_loss/len(train_loader):.4f}")


model.eval()
with torch.no_grad():
    test_loss=0
    correct=0
    total=0
    for images, labels in test_loader:
        images,labels=images.to(device),labels.float().to(device).view(-1,1)
        outputs=model(images)
        predicted=torch.round(torch.sigmoid(outputs))
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)
        loss=criterion(outputs,labels)
        test_loss+=loss.item()

print("Test Loss: {0:.3f}".format(test_loss/len(test_loader)))
print("Accuracy: {0:.2f}%".format(100*correct/total))

torch.save(model.state_dict(),'model.pth')
print("Model saved")