import torch 
import torch.nn as nn
from .blocks.transformer_block import TransformerBlock as TBLK


class Restormer(nn.Module):
    def __init__(self, channels, heads, height, width) -> None:
        super().__init__()
        self.upsample = nn.PixelShuffle(2)   # REDUCE THE NUMBER OF CHANNELS
        self.downsample = nn.PixelUnshuffle(2)   # INCREASE THE NUMBER OF CHANNELS

        self.initial_convolution_layer = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1)
        self.final_convolution_layer = nn.Conv2d(in_channels=channels*3, out_channels=3, kernel_size=3, padding=1 )

        self.encoder_level_1 = TBLK(heads=heads,
                                    channels=channels,
                                    height=height,
                                    width=width)
        
        self.encoder_level_2 = TBLK(heads=heads,
                                    channels=channels*4,
                                    height=height//2,
                                    width=width//2)

        self.middle = TBLK(heads=heads,
                           channels=channels*16,
                           height=height//4,
                           width=width//4)

        self.decoder_level_1 = TBLK(heads=heads,
                                    channels=channels*3,
                                    height=height,
                                    width=width)

        self.decoder_level_2 = TBLK(heads=heads,
                                    channels=channels*8,
                                    height=height//2,
                                    width=width//2)

    def forward(self, x):
        o = self.initial_convolution_layer(x)

        # Encoders 
        o1 = self.encoder_level_1(o)
        o2 = self.encoder_level_2(self.downsample(o1))
        o3 = self.middle(self.downsample(o2))

        # Decoders
        o3 = self.upsample(o3)
        o2 = torch.cat((o3,o2), dim=1)
        o2 = self.decoder_level_2(o2)
        o2 = self.upsample(o2)
        o1 = torch.cat((o2,o1), dim=1)
        o1 = self.decoder_level_1(o1)
        
        result = self.final_convolution_layer(o1)

        return result + x




