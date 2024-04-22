import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, pc_in, pc_out) -> None:
        super().__init__()
        
        self.PC = nn.Conv2d(in_channels=pc_in, out_channels=pc_out, kernel_size=(1,1), bias=False) # Normal convolution
        self.DC = nn.Conv2d(in_channels=pc_out, out_channels=pc_out, kernel_size=(3,3), bias=False, groups=pc_out, padding=1)  #depth wise convolution

    def forward(self, x):
        o = self.PC(x)
        return self.DC(o)



