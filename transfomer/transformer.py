import torch 
import torch.nn as nn
from .blocks.transformer_block import TransformerBlock as TBLK


class Restormer(nn.Module):
    def __init__(self, channels, heads, height, width) -> None:
        super().__init__()
        self.downsample = nn.PixelShuffle(2)
        self.upsample = nn.PixelUnshuffle(2)

        self.initial_convolution_layer = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3)
        self.final_convolution_layer = nn.Conv2d(in_channels=channels*2, out_channels=3, kernel_size=3)

        self.encoder_level_1 = TBLK(heads=heads,
                                    channels=channels*2,
                                    height=height,
                                    width=width)
        
        self.encoder_level_2 = TBLK(heads=heads,
                                    channels=channels*4,
                                    height=height,
                                    width=width)

        self.middle = TBLK(heads=heads,
                           channels=channels*8,
                           height=height,
                           width=width)

        self.decoder_level_1 = TBLK(heads=heads,
                                    channels=channels*4,
                                    height=height,
                                    width=width)
        
        self.decoder_level_2 = TBLK(heads=heads,
                                    channels=channels*2,
                                    height=height,
                                    width=width)

    def forward(self, x):
        
        o = self.initial_convolution_layer(x)

        # Encoders 
        
        o1 = self.encoder_level_1(o)
        o2 = self.encoder_level_2(self.downsample(o1))

        o3 = self.middle(self.downsample(o2))

        #decoders
        o3 = self.upsample(o3)
        o3 = torch.cat(o3,o2, dim=-1)

        o2 = self.decoder_level_2(o3)

        o1 = torch.cat(o2,o1, dim=-1)
        
        o1 = self.decoder_level_1(o1)

        o1 = self.final_convolution_layer(o1)

        return o1 + x




