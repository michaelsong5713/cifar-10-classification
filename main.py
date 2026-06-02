import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0),"\n")

training_data = datasets.CIFAR10(
    root = "data",
    download = True,
    train = True
)
testing_data = datasets.CIFAR10(
    root = "data",
    download = True,
    train = False
)

mean = [0.4914, 0.4822, 0.4465]
std  = [0.2470, 0.2435, 0.2616]

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.RandomCrop(32, padding=4),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std)
])
test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std)
])
training_data.transform = train_transform
testing_data.transform = test_transform

training_len = len(training_data)
train_size = int(0.85 * training_len)
val_size = int(0.15 * training_len)
training_dataset,validation_dataset = random_split(
    training_data,
    [train_size,val_size]
)

train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(validation_dataset,batch_size=64, shuffle=False)
test_dataloader = DataLoader(testing_data,batch_size=64, shuffle = False)

class conv_block(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.block(x)
        return x

class neural_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3,32), #32x32
            conv_block(32,64),
            conv_block(64,128),
            nn.MaxPool2d(kernel_size=2,stride=2), #16x16
            conv_block(128,256), 
            conv_block(256,256), 
            nn.MaxPool2d(kernel_size=2,stride=2), #8x8
            conv_block(256,512), 
            conv_block(512,512),
            nn.MaxPool2d(kernel_size=2,stride=2), #4x4 
        )
        self.classifier = nn.Sequential(
            #nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512*4*4,512*4*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512*4*2,10)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
generations = 100
model = neural_network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=generations)
val_accuracies = []


for i in range(generations):
    cur = i+1
    print(f"{cur}")
    correct = 0
    total = 0
    model.train()
    for batch_d,batch_l in train_dataloader:
        batch_d, batch_l = batch_d.to(device), batch_l.to(device)
        output = model(batch_d)
        #print(f"{output.shape}")
        loss = loss_fn(output,batch_l)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prediction = torch.argmax(output,dim=1)
        correct += torch.sum(prediction==batch_l)
        total+=prediction.size(0)
    accuracy = correct/total*100
    print(f"Train Accuracy: {accuracy}")
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_d,batch_l in val_dataloader:
            batch_d, batch_l = batch_d.to(device), batch_l.to(device)
            output = model(batch_d)
            loss = loss_fn(output,batch_l)
            prediction = torch.argmax(output,dim=1)
            correct += torch.sum(prediction==batch_l)
            total+=prediction.size(0)
    accuracy = correct/total*100
    val_accuracies.append(accuracy.item())
    print(f"Val Accuracy: {accuracy}")
    scheduler.step()

correct = 0
total = 0
model.eval()

with torch.no_grad():
    for batch_d,batch_l in test_dataloader:
        batch_d, batch_l = batch_d.to(device), batch_l.to(device)
        output = model(batch_d)
        prediction = torch.argmax(output,dim=1)
        correct += torch.sum(prediction==batch_l)
        total+=prediction.size(0)
accuracy = correct/total*100

print(f"Accuracy: {accuracy}\n")
torch.save(model.state_dict(), "model.pth")

# vibe coded plot to show accuracy
epochs = range(1, generations + 1)
 
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, val_accuracies, color="#5563DE", linewidth=2, marker="o",
        markersize=4, label="Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Validation accuracy over training")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("val_accuracy_curve.png", dpi=150)