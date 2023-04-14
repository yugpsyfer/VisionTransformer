import torch
import torch.nn as nn
from .gdfn import GDFN
from .mdta import MDTA


class ConvolutionBlock(nn.Module):
    def __init__(self, pc_in, pc_out) -> None:
        super().__init__()
        
        self.PC = nn.Conv2d(in_channels=pc_in, out_channels=pc_out, kernel_size=(1,1), bias=False)
        self.DC = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), bias=False)

    def forward(self, x):
        o = self.PC(x)

        channels = [self.DC(o[:,:,:,i]) for i in range(o.shape[2])]

        o = torch.stack(channels, dim=3)

        return o



class TransformerBlock(nn.Module):

    def __init__(self, heads, channels, height, width) -> None:
        super().__init__()

        self.mdta = MDTA(input_height=height,
                         input_width=width,
                         input_channels=channels,
                         num_heads=heads)
        
        self.gdfn = GDFN(input_channels=channels)


    def forward(self, x):
        return self.gdfn(self.mdta(x))
