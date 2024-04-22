"""
To Do: 
    1. Resize the images and make them more smaller.
    2. Re - adjust the channels and other paramters of the model.
"""


import os
import shutil
from PIL import Image

import torch
# import wandb
from torch.optim import AdamW, Adam, SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from transfomer.transformer import Restormer
from datasets.raindrop import RainDrop



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

loss = torch.nn.MSELoss()

train_set = RainDrop(split='train')
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

c,h,w = train_set.get_image_dimension()


def train_single_step(model, train_loader, optimizer):
    train_loss  = 0
    for batch in train_loader:
        x, y = batch

        x = x.to(device)
        y = y.to(device)

        o = model(x)
        l = loss(o, y)
        
        l.backward()
        optimizer.step()

        train_loss+=l.detach()
    
    return train_loss



@torch.no_grad()
def evaluate(model, val_loader):
    val_loss = 0
    for batch in val_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        o = model(x)
        l = loss(o, y)

        val_loss+=l.detach()
    
    return val_loss


def results(model, device):
    total_images = len(os.listdir('./results'))//3
    resize = transforms.Resize(size=(128,128), antialias=True)
    toTensor = transforms.ToTensor()
    toPILImage= transforms.ToPILImage()
    
    destination = './results/'

    with Image.open('./data/RainDrop/train/data/' + str(total_images + 1) + "_rain.png") as targetImg:
        targetImgTensor = torch.unsqueeze(resize(toTensor(targetImg)).to(device), dim=0)
        ouputImg = toPILImage(model(targetImgTensor).squeeze(dim=0))

        targetImg.save(destination + str(total_images + 1) + '_target.png')
        ouputImg.save(destination + str(total_images + 1) + '_output.png')
    
    shutil.copy(src="./data/RainDrop/train/gt/" + str(total_images + 1) + "_clean.png", dst=destination + str(total_images + 1) + '_clean.png')



if __name__ == '__main__':
    epochs = 500
    learning_rate = 3e-4
    model = Restormer(channels=8,
                      heads=1,
                      height=h,
                      width=w)
   
    model.to(device)
    print(model)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss = train_single_step(model, train_loader, optimizer)

        print(f"Epoch: {epoch} | Loss: {train_loss:.6f}")
        
    
    results(model, device)

    torch.save(model, './checkpoint/best.pt')

