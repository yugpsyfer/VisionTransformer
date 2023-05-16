import torch.nn as nn

from transfomer.gdfn import GDFN
from transfomer.mdta import MDTA


class TransformerBlock(nn.Module):

    def __init__(self, heads, channels, height, width) -> None:
        super().__init__()

        self.mdta = MDTA(input_height=height,
                         input_width=width,
                         input_channels=channels,
                         num_heads=heads)
        
        self.gdfn = GDFN(input_height=height,
                         input_width=width,
                         input_channels=channels)


    def forward(self, x):
        return self.gdfn(self.mdta(x))