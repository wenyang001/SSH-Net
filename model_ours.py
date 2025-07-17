## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math
from einops import rearrange
from timm.models.layers import trunc_normal_, to_2tuple
import os

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

##########################################################################
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    

##  Multi-Scale Feed-forward Layer (MFL)
class MSF(nn.Module):
    def __init__(self, dim, bias):
        super(MSF, self).__init__()

        # hidden_features = int(dim * ffn_expansion_factor)
        # hidden_features = 180
        hidden_features = 120

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features, bias=bias)

        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu7 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)
        self.dwconv7x7_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.relu7_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)

        x1_3, x2_3, x3_3 = self.relu3(self.dwconv3x3(x)).chunk(3, dim=1)
        x1_5, x2_5, x3_5 = self.relu5(self.dwconv5x5(x)).chunk(3, dim=1)
        x1_7, x2_7, x3_7 = self.relu7(self.dwconv7x7(x)).chunk(3, dim=1)

        x1 = torch.cat([x1_3, x1_5, x1_7], dim=1)
        x2 = torch.cat([x2_3, x2_5, x2_7], dim=1)
        x3 = torch.cat([x3_3, x3_5, x3_7], dim=1)  #

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))
        x3 = self.relu7_1(self.dwconv7x7_1(x3))

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.project_out(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape  # C=30，即通道数

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # b 1 C C

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        # print(111, mask1.scatter_(-1, index, 1.))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)  # [1 6 30 30]
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class HAB(nn.Module):
    r""" Modified Hybrid Attention Block: TTSA + WSA.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        # self.conv_block = Attention(dim, num_heads, qkv_bias)  # replace CA with TTSA
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForward(dim, mlp_ratio, qkv_bias)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        attn_x = attn_x.view(b, h * w, c)
        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x).view(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous().view(b, h * w, c))

        return x

    
class AttenBlocks(nn.Module):
    """ A series of attention blocks for one RHAG.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            HAB(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'])

        # x = self.overlap_attn(x, x_size, params['rpi_oca'])

        if self.downsample is not None:
            x = self.downsample(x)
        return x
    

class RHAG(nn.Module):
    """Residual Hybrid Attention Group (RHAG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=1,
                 resi_connection='1conv'):
        super(RHAG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)


    def forward(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        attn_mask = calculate_mask(x_size).to(x.device)
        params['attn_mask'] = attn_mask
        x = self.patch_embed(x)
        x = self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))
        return x

WS = 16
SS = WS // 2
def calculate_mask(x_size):
    # calculate attention mask for SW-MSA
    h, w = x_size
    img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
    h_slices = (slice(0, -WS), slice(-WS, -SS), slice(-SS, None))
    w_slices = (slice(0, -WS), slice(-WS, -SS), slice(-SS, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, WS)  # nw, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, WS * WS)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class TransformerBlock1(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention1(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.proj = nn.Sequential(
        #     nn.Conv2d(in_c, embed_dim, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2))

    def forward(self, x):
        x = self.proj(x)

        return x

class OverlapPatchRestore(nn.Module):
    def __init__(self, in_c=48, out_c=3, bias=False):
        super(OverlapPatchRestore, self).__init__()
        # 使用 ConvTranspose2d 执行转置卷积（反卷积）
        self.restore = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias)

    def forward(self, x):
        # 执行反卷积操作，尝试还原原始图像
        x = self.restore(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2)) # C --> 2C
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat): # dim --> dim/2
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
class Downsample_Only(nn.Module): # dim --> dim
    def __init__(self, n_feat):
        super(Downsample_Only, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.ReLU(inplace=True))      
                        
    def forward(self, x):
        return self.body(x)
    
class Upsample_Only(nn.Module): # dim --> dim hw --> 2h 2w
    def __init__(self, n_feat):
        super(Upsample_Only, self).__init__()
        self.body = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
                                    nn.ReLU(inplace=True))      
                        
    def forward(self, x):
        return self.body(x)




class GatedFeatureCombination(nn.Module):
    def __init__(self, in_channels):
        super(GatedFeatureCombination, self).__init__()
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.last_gate_signal = None  # 保存最近一次的 gate

    def forward(self, x1, x2): # x1 是 denoising feature ， x2 是 out_dec_level1_watermark_feature
        gate_signal = torch.sigmoid(self.gate(x2))
        self.last_gate_signal = gate_signal.detach().cpu()  # 用于可视化
        x1 = gate_signal * x1
        return x1 + x2


class Upsample_New(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample_New, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(PixelShuffleBlock, self).__init__()
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample_New(upscale_factor, 64)
        self.conv_last = nn.Conv2d(64, out_channels, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        return x



class SharedEncoder(nn.Module):
    def __init__(self,
        inp_channels=3, 
        out_channels=3, 
        dim = 24,  # 48
        num_blocks = [1, 1, 1, 1], 
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',  ## Other option 'BiasFree'
    ):
        super(SharedEncoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # shared encoder 
        self.encoder_level1 = nn.Sequential(*[NAFBlock(c=int(dim)) for i in range(num_blocks[0])]) 
        self.down1_2 = Downsample_Only(int(dim)) ## From Level 1 to Level 2
        
        self.encoder_level2 = nn.Sequential(*[NAFBlock(c=int(dim)) for i in range(num_blocks[1])])
        self.down2_3 = Downsample_Only(int(dim)) ## From Level 2 to Level 3
        
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim)) ## From Level 3 to Level 4

        self.encoder_level4 = nn.Sequential(*[TransformerBlock(dim=int(dim*2), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.down4_5 = Downsample(int(dim*2)) ## From Level 3 to Level 4
        
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*4), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[4])])

    def forward(self, inp_img):
        # shared encoder for denoising & watermarking
        inp_enc_level1 = self.patch_embed(inp_img)  # 48 channels

        out_enc_level1 = self.encoder_level1(inp_enc_level1) # 48 channels
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2) # 48 channels
        inp_enc_level3 = self.down2_3(out_enc_level2)   # 1x48x64x64

        # print(inp_enc_level3.shape)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) # 1x48x64x64
        # print(out_enc_level3.shape)
        inp_enc_level4 = self.down3_4(out_enc_level3)  # 1x96x32x32
        # print(inp_enc_level4.shape)


        out_enc_level4 = self.encoder_level4(inp_enc_level4) # 1x96x32x32
        inp_enc_level5 = self.down4_5(out_enc_level4)  # 1x192x16x16
        # print(inp_enc_level5.shape)

        latent = self.latent(inp_enc_level5) # 48 channels

        return latent, inp_enc_level3, [out_enc_level4, out_enc_level3, out_enc_level2, out_enc_level1]

    
class WatermarkDecoder(nn.Module):
    def __init__(self,
        inp_channels=3, 
        out_channels=3, 
        dim = 48,  # 48
        num_blocks = [1, 1, 1, 1], 
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',  ## Other option 'BiasFree'
        refine_layers = 2,
    ):
        super(WatermarkDecoder, self).__init__()

        self.up5_4 = Upsample(int(dim*4)) ## From Level 5 to Level 4
        self.reduce_chan_level4 = nn.Conv2d(int(dim*4), int(dim*2), kernel_size=1, bias=bias)
        self.decoder_level4 = nn.Sequential(*[TransformerBlock(dim=int(dim*2), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample_Only(int(dim)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[NAFBlock(c=int(dim)) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample_Only(int(dim))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[NAFBlock(c=int(dim)) for i in range(num_blocks[0])])
    
        
    def forward(self, latent, encoder_feature):
        out_enc_level4, out_enc_level3, out_enc_level2, out_enc_level1 = encoder_feature # x2, x1, x1, x1

        inp_dec_level4 = self.up5_4(latent) # dim*2
        # print(inp_dec_level4.shape, 'laa')
        inp_dec_level4 = torch.cat([inp_dec_level4, out_enc_level4], 1) # dim*4
        inp_dec_level4 = self.reduce_chan_level4(inp_dec_level4)
        out_dec_level4 = self.decoder_level4(inp_dec_level4)

        inp_dec_level3 = self.up4_3(out_dec_level4) # dim
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1) # dim*2
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3) # dim*2
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2) # dim*3
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) # 48 channels

        return out_dec_level1
    


class DenoiseDecoder(nn.Module):
    def __init__(self,
        inp_channels=3, 
        out_channels=3, 
        dim = 24,  # 48
        num_blocks = [1, 1, 1, 1], 
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',  ## Other option 'BiasFree'
    ):
        super(DenoiseDecoder, self).__init__()

        # noise removal decoder
        self.latent = nn.Sequential(*[NAFBlock(c=int(dim)) for i in range(num_blocks[1])])

        self.up3_2_denoising = Upsample_Only(int(dim)) ## From Level 3 to Level 2
        self.decoder_level2_denoising = nn.Sequential(*[NAFBlock(c=int(dim)) for i in range(num_blocks[1])])
        
        self.up2_1_denoising = Upsample_Only(int(dim))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1_denoising = nn.Sequential(*[NAFBlock(c=int(dim)) for i in range(num_blocks[0])])

    
    def forward(self, latent_gt_dn):
        # noise removal decoder
        latent_gt_dn = self.latent(latent_gt_dn) # 48 channels

        inp_dec_level2_denoising = self.up3_2_denoising(latent_gt_dn)
        inp_dec_level2_denoising = self.decoder_level2_denoising(inp_dec_level2_denoising)

        inp_dec_level1_denoising = self.up2_1_denoising(inp_dec_level2_denoising)
        out_dec_level1_denoising = self.decoder_level1_denoising(inp_dec_level1_denoising) # 24 channels

        return out_dec_level1_denoising


import time
##########################################################################
##---------- Restormer -----------------------
class SSHNet(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,  # 48
        num_blocks = [2, 4, 4, 6, 6],
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = True,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        refine_layers = 6,
    ):
        super(SSHNet, self).__init__()

        self.share_encoder = SharedEncoder(inp_channels, out_channels, dim, num_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.watermark_decoder = WatermarkDecoder(inp_channels, out_channels, dim, num_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type, refine_layers)
        self.denoise_decoder = DenoiseDecoder(inp_channels, out_channels, dim, num_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type)
         
        self.feature_combiner1 = GatedFeatureCombination(in_channels=dim)
        self.refinement = NAFBlock(c=dim)

        self.output_denoising_img = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1)
        self.output_watermark_img = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, inp_img):
        latent, latent_dn, [out_enc_level4, out_enc_level3, out_enc_level2, out_enc_level1] = self.share_encoder(inp_img)

        # noise removal decoder
        out_dec_level1_denoising_feature = self.denoise_decoder(latent_dn)
        denoised_output = self.output_denoising_img(out_dec_level1_denoising_feature)

        # watermarking decoder
        out_dec_level1_watermark_feature = self.watermark_decoder(latent, [out_enc_level4, out_enc_level3, out_enc_level2, out_enc_level1])
        watermark_removed_output = self.output_watermark_img(out_dec_level1_watermark_feature)

        # 融合去噪和去水印结果
        final_output = self.feature_combiner1(out_dec_level1_denoising_feature, out_dec_level1_watermark_feature) 
        final_output = self.output(self.refinement(final_output)) + inp_img

        return final_output, denoised_output, watermark_removed_output


def measure_runtime_and_memory(model, input_tensor, repeat=100):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timings = []

    with torch.no_grad():
        # warm-up
        for _ in range(10):
            _ = model(input_tensor)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(repeat):
            starter.record()
            _ = model(input_tensor)
            ender.record()
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender)  # milliseconds
            timings.append(elapsed_time)

        avg_time = sum(timings) / len(timings)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    return avg_time, peak_memory

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    img = torch.randn((1, 3, 256, 256)).cuda()
    model = SSHNet().cuda()
    
    from thop import profile
    flops, params = profile(model, inputs=(img, ))
    print(f"{flops / 1e9:.2f} GFLOPs, {params / 1e6:.2f} MParams")

    avg_time_ms, peak_mem = measure_runtime_and_memory(model, img, repeat=400)
    print(f"Average Runtime: {avg_time_ms:.2f} ms")
    print(f"Peak GPU Memory Usage: {peak_mem:.2f} MB")