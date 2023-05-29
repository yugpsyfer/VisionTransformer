import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, pc_in, pc_out) -> None:
        super().__init__()
        
        self.PC = nn.Conv2d(in_channels=pc_in, out_channels=pc_out, kernel_size=(1,1), bias=False, padding=1)
        self.DC = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), bias=False)

    def forward(self, x):
        o = self.PC(x)
        channels = [self.DC(torch.unsqueeze(o[:,i,:,:], dim=1)).squeeze() for i in range(o.shape[1])]

        o = torch.stack(channels, dim=1)
    
        return o



