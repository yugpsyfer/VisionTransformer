import torch
import wandb
from torch.optim import AdamW

from transfomer.transformer import Restormer
from datasets.raindrop import RainDrop
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

loss = torch.nn.MSELoss()

train_set = RainDrop(split='train')
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

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



if __name__ == '__main__':
    epochs = 20
    learning_rate = 3e-4
    model = Restormer(channels=8,
                      heads=2,
                      height=h,
                      width=w)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss = train_single_step(model, train_loader, optimizer)

        print(f"Epoch: {epoch} ,  Loss: {train_loss}")
