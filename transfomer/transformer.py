import torch 
import torch.nn as nn
from .blocks.transformer_block import TransformerBlock as TBLK


class Restormer(nn.Module):
    def __init__(self, channels, heads, height, width) -> None:
        super().__init__()
        self.upsample = nn.PixelShuffle(2)   # REDUCE THE NUMBER OF CHANNELS
        self.downsample = nn.PixelUnshuffle(2)   # INCREASE THE NUMBER OF CHANNELS

        self.initial_convolution_layer = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1)
        self.final_convolution_layer = nn.Conv2d(in_channels=channels*2, out_channels=3, kernel_size=3, padding=1 )

        self.encoder_level_1 = TBLK(heads=heads,
                                    channels=channels,
                                    height=height,
                                    width=width)
        
        self.middle = TBLK(heads=heads,
                           channels=channels*4,
                           height=height//2,
                           width=width//2)

        self.decoder_level_1 = TBLK(heads=heads,
                                    channels=channels*2,
                                    height=height,
                                    width=width)

    def forward(self, x):
        
        o = self.initial_convolution_layer(x)

        # Encoders 
        
        o1 = self.encoder_level_1(o)


        o3 = self.middle(self.downsample(o1))

        #decoders
        o3 = self.upsample(o3)

        o1 = torch.cat((o3,o1), dim=1)

        
        o1 = self.decoder_level_1(o1)

        o1 = self.final_convolution_layer(o1)

        return o1 + x




