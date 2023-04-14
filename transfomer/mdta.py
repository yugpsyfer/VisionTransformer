import torch 
import torch.nn as nn
from .block import ConvolutionBlock as convBlock


class MDTA(nn.Module):
    """
    Purpose:- To find cross covariance between the channels at a given level of the encoder/decoder block

    This class takes (B,H,W,C) input where B is batch, H is height and W is widt and C is channels.
    The output and input of each block is going to be of same dimension.
    The Downsampling / Upsampling will be handled by a different class.

    Value - B,H*W,C
    Key - B,C,H*W
    Query - B,H*W,C

    The self attention is applied on the channels and not the pixels so we get (C,C) instead of (H*W,H*W) matrix.

    """

    def __init__(self, input_height, input_width, num_heads, input_channels) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.input_height = input_height
        self.input_width = input_width
    
        self.channels = input_channels

        self.alpha = torch.tensor(data=1, requires_grad=True)
        self.LN = nn.LayerNorm()
        
        self.conv_block_Q = convBlock(pc_in=input_channels,
                                      pc_out=input_channels)
        
        self.conv_block_K = convBlock(pc_in=input_channels,
                                      pc_out=input_channels)
        
        self.conv_block_V = convBlock(pc_in=input_channels,
                                      pc_out=input_channels)

        self.conv_final = nn.Conv2d(kernel_size=(1,1),
                                    out_channels=input_channels,
                                    in_channels=input_channels)
        
        self.softmax = nn.Softmax(dim=-1)


    def make_attention_heads(self, Q, K, V):
        """
        Divides channels into attention heads
        """
        b,t,c = K.size()

        s = c//self.num_heads

        K = K.view(b, t, self.num_heads, s)
        V = V.view(b, t, self.num_heads, s)
        Q = Q.view(b, t, self.num_heads, s)

        K = (K.transpose(1,2).view(b*self.num_heads, t, s)).transpose(1,2).contigous().view(b*self.num_heads, t, s)
        V = (V.transpose(1,2).view(b*self.num_heads, t, s)).transpose(1,2).contigous().view(b*self.num_heads, t, s)
        
        Q = Q.transpose(1,2).view(b*self.num_heads, t, s)
        
        return K,Q,V, b,t,s


    def forward(self, x):
        x = self.LN(x)
        
        Q = self.conv_block_Q(x).view(-1,
                                      self.input_height*self.input_width,
                                      self.channels)
        
        K = self.conv_block_K(x).view(-1,
                                      self.input_height*self.input_width,
                                      self.channels)

        V = self.conv_block_V(x).view(-1,
                                      self.input_height*self.input_width,
                                      self.channels)

        K,Q,V,b,t,s = self.make_attention_heads(K,Q,V)

        dot_prod = (torch.bmm(V, self.softmax(torch.bmm(K,Q)/self.alpha))).view(b,self.num_heads,t,s)

        dot_prod = dot_prod.transpose(1,2).view(b,t,self.num_heads*s)
        dot_prod = dot_prod.contiguous().view(b,self.input_height,self.input_width,self.num_heads*s)

        x_cap = self.conv_final(dot_prod) + x

        return x_cap


