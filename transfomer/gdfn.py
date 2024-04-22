import torch.nn as nn
from .blocks.convolution_block import ConvolutionBlock as convBlock


class GDFN(nn.Module):
    """
    Instead of linear layer the authors have used
    the Gated D-Conv feed forward network.

    parameters:-
    
    input_channels - 
    channel expansion factor - 4 (Mentioned in the paper)

    """
    def __init__(self, input_height, input_width, input_channels, channel_expansion_factor=2) -> None:
        super().__init__()
        
        self.LN = nn.LayerNorm(normalized_shape=input_channels)
        
        self.conv_1 = convBlock(pc_in=input_channels,
                                pc_out=input_channels*channel_expansion_factor)
        
        self.conv_2 = convBlock(pc_in=input_channels,
                                pc_out=input_channels*channel_expansion_factor)
        
        self.conv_final = nn.Conv2d(kernel_size=(1,1),
                                    in_channels=input_channels*channel_expansion_factor,
                                    out_channels=input_channels)
        self.gelu = nn.GELU()

    def gating(self, x):
        x = x.permute(0,3,2,1)
        
        x = self.LN(x)
        o = x.permute(0,3,2,1)

        o1 = self.gelu(self.conv_1(o))
        
        o2 = self.conv_2(o)
        
        return o1 * o2
        
    def forward(self, x):
        out = self.gating(x)

        return self.conv_final(out) + x 


