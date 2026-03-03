## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import time
import math
from functools import partial
from typing import Optional, Callable

import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
#from mamba_ssm.modules.mamba2 import Mamba2

import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from pdb import set_trace as stx
import numbers
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from window import WindowmambaBlock
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from einops import rearrange, repeat
import math
import copy

class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])


from einops import rearrange, repeat



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class PyramidPooling:
    def __init__(self, levels=3):
        self.levels = levels
        self.pooled_features = []

    def pyramid_pooling(self, image):
        # Initialize a list to store pooled features at different levels
        self.pooled_features = []

        # Append the original image as the base level feature map
        self.pooled_features.append(image.clone())

        # Apply pyramid pooling layer by layer
        for i in range(1, self.levels):
            # Reduce the spatial dimensions using average pooling
            reduced_image = F.avg_pool2d(image, kernel_size=2**i, stride=2**i)
            self.pooled_features.append(reduced_image.clone())

        l1 = self.pooled_features[0]
        l2 = self.pooled_features[1]
        l3 = self.pooled_features[2]

        # Return the pooled features for each level
        return l1, l2, l3


def extract_color_map(image):
    # Convert image to LAB color space
    lab_image = kornia.color.rgb_to_lab(image.float())

    # Split LAB channels
    l_channel = lab_image[:, 0:1, :, :]  # L channel, shape (1, 1, 128, 128)
    ab_channels = lab_image[:, 1:3, :, :]  # ab channels, shape (1, 2, 128, 128)

    # Create maps for AB channels
    ab_channels_map = ab_channels

    # Create map for L channel
    L_channels_map = l_channel

    return L_channels_map, ab_channels_map

def extract_hsv_color_map(image):
    # Convert image to HSV color space
    hsv_image = kornia.color.rgb_to_hsv(image.float())

    # Split HSV channels
    s_channel = hsv_image[:, 1:2, :, :]  # H channel, shape (1, 1, 128, 128)
    # Create maps for each HSV channel
    s_channel_map = s_channel
    return s_channel_map


def extract_gradient_map(image):
    # Ensure the image is in the right format (float tensor)
    image = kornia.color.rgb_to_grayscale(image.float())

    # Compute gradients in x and y directions using Sobel operator
    grad_x = kornia.filters.sobel(image)  # Gradient along x-axis
    grad_y = kornia.filters.sobel(image.transpose(2, 3))  # Gradient along y-axis

    # You can return both gradients as a map or compute gradient magnitude
    grad_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # Gradient magnitude

    return grad_map




def calculate_max_color_difference(image):
    # Assuming input image has shape (B, C, H, W) where B is batch size, C is channels, H is height, W is width
    # Extract BGR channels
    b_channel = image[:, 0, :, :]  # Blue channel
    g_channel = image[:, 1, :, :]  # Green channel
    r_channel = image[:, 2, :, :]  # Red channel

    # Convert channels to torch tensors
    b_tensor = torch.tensor(b_channel, dtype=torch.float)
    g_tensor = torch.tensor(g_channel, dtype=torch.float)
    r_tensor = torch.tensor(r_channel, dtype=torch.float)

    # Calculate maximum color difference
    max_color_diff = torch.maximum(torch.abs(g_tensor - r_tensor), torch.abs(b_tensor - r_tensor))

    # Convert max_color_diff to numpy array for further processing if needed
    max_color_diff_image = max_color_diff.unsqueeze(0)

    return max_color_diff_image

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

class UNetEncoder(nn.Module):
    def __init__(self, dim, n_channels=3):
        super(UNetEncoder, self).__init__()
        self.inc = nn.Conv2d(1, dim, kernel_size=1)
        self.down1_2 = Downsample(dim)
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.down3_4 = Downsample(int(dim * 2 ** 2))

    def forward(self, x):
        # print("xxx =", x.shape)
        x1 = self.inc(x)
        x2 = self.down1_2(x1)
        x3 = self.down2_3(x2)
        x4 = self.down3_4(x3)
        return x1, x2, x3, x4


class FeatureExtraction(nn.Module):
    """
    Feature Extraction (FE)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of Conv2d. Default: 3
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FeatureExtraction, self).__init__()

        # FE
        self.fe = nn.Sequential(
            # Conv + LReLU + IN
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x):  # B 3 H W
        # FE
        x = self.fe(x)  # B C H W
        return x


class SCBlock(nn.Module):
    """
    MultiScale Feature Extraction (MFE)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self,dim):
        super(SCBlock, self).__init__()

        # MultiScale (3*3 + 5*5 + 7*7)
        self.scale_3 = FeatureExtraction(dim, 8, kernel_size=3)
        self.scale_5 = FeatureExtraction(dim, 8, kernel_size=5)
        self.scale_7 = FeatureExtraction(dim, 8, kernel_size=7)

    def forward(self, x):  # B 48 H W
        # MultiScale (3*3 + 5*5 + 7*7)
        x1 = self.scale_3(x)  # B C H W
        x2 = self.scale_5(x)  # B C H W
        x3 = self.scale_7(x)  # B C H W
        # Concat
        x = torch.cat([x1,x2,x3] ,dim=1) # B 48 H W
        return x


class VSSBlock(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            headdim: int = 12,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.self_attention = Mamba2(d_model=hidden_dim, d_state=d_state, expand=expand, headdim=headdim)
        self.drop_path = DropPath(drop_path)
        self.norm = nn.LayerNorm(hidden_dim)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv = nn.Conv2d(dim * 2, dim,kernel_size=1)
      #  self.pc = image_to_patches_and_sum(patchsize=4)

    def forward(self, input, t):
        # x [B, C, H, W]
        B, C, H, W = input.shape
        x = input.reshape(B, H, W, C).contiguous()  # [B,H,W,C]
        B1, C1, H1, W1 = t.shape
        input_l = t.reshape(B1, H1, W1, C1).contiguous()  # [B,H,W,C]
       # print(input_l.shape)
        x = self.ln_1(x)
        input_l = self.ln_1(input_l)
        x = self.drop_path(self.self_attention(x, input_l))
      #  input_l = self.pc(input_l)
        x1 = self.drop_path(self.self_attention(input_l, x))
        # x = x.reshape(B, C, H, W)
        x = self.norm(x)
        x1 = self.norm(x1)
        x = x.reshape(B, C, H, W)
        x1 = x1.reshape(B, C, H, W)
        x = torch.cat([x, x1], dim=1)
        x = self.conv(x)
        x = x + input

        return x, t

class CrossMamba(nn.Module):
    def __init__(self, dim,d_state, expand, hidden_dim,headdim, drop_path, attn_drop_rate):
        super(CrossMamba, self).__init__()
        self.cross_mamba = VSSBlock(dim=dim, d_state=d_state, expand=expand, hidden_dim=dim, drop_path=drop_path, headdim=headdim, attn_drop_rate=attn_drop_rate)
        self.norm = LayerNorm(dim, 'with_bias')
      #  self.unet = UNetEncoder(dim=dim, n_channels=3)

    def forward(self, x):
        x, t = x
       # print("L_channel_maps =", x.shape)
       # L_channel_maps = self.unet(L_channel_maps)
        global_f, global_t = self.cross_mamba(x, t)

        return global_f, global_t

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class RCACM(nn.Module):
    def __init__(self, in_channels, hidden_dim, w, h, window_size, shift_size):
        super(RCACM, self).__init__()
        self.ca = ResidualBlock(in_channels, hidden_dim)
        self.cm = CCOSS(in_channels, w, h, d_state=16, expand=1, d_conv=4, mam_block=2)
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.LocalMamba = WindowmambaBlock(dim=in_channels, h=h, w=w, window_size=window_size, shift_size=shift_size)
        self.skff2 = SKFF2(in_channels, 2)

    def forward(self, x):  # B C H W
       # a,m = x.chunk(2, dim=-1)c
        a = self.cm(x)
      #  out = self.LocalMamba(m)
        m = self.cm(x)
      #  out = torch.cat([a, m], dim=1)
      #  out = self.conv(out)
        out = self.skff2([a, m])
        return out

class RSACM(nn.Module):
    def __init__(self, dim, d_state, expand, hidden_dim, headdim, drop_path, h, w, window_size, shift_size):
        super(RSACM, self).__init__()
        self.sa = ResidualBlock1(dim, hidden_dim)
        self.sm = VssBlock(dim=dim, d_state=d_state, expand=expand, hidden_dim=dim, drop_path=drop_path, headdim=headdim, d_model=hidden_dim)
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.LocalMamba = WindowmambaBlock(dim=dim, h=h, w=w, window_size=window_size, shift_size=shift_size)
        self.skff3 = SKFF2(dim, 2)

    def forward(self, x):  # B C H W
       # a,m = x.chunk(2, dim=-1)
        a = self.sm(x)
     #   out = self.LocalMamba(m)
        m = self.sm(x)
     #   out = torch.cat([a, m], dim=1)
     #   out = self.conv(out)
        out = self.skff3([a, m])
        return out

class VssBlock(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 1.,
            headdim: int = 12,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.self_attention = SS2D(dim=dim, d_state=d_state, expand=expand, hidden_dim=dim, drop_path=drop_path, headdim=headdim, d_model=hidden_dim)
      #  self.self_attention = CCOSS(h=h,w=w, d_state=d_state, expand=1, d_conv=4,channel=dim)
        self.drop_path = DropPath(drop_path)
      #  self.sa = SpatialAttention()

    def forward(self, input):
        # x [B, C, H, W]
        input_0 =input
        B, C, H, W = input.shape
        input = input.reshape(B, H , W, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
     #   x = x.reshape(B, C ,1, H)
       # x = x.view(B,H,W,1,C)
        x = self.drop_path(self.self_attention(x))
        x = self.ln_1(x)
        x = x.view(B, C, H, W)
      #  x = self.sa(x)
        x = x + input_0

        return x





##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class ChannelAttention(nn.Module):
    """
    Channel Attention (CA)
    Args:
        in_channels (int): Number of input channels.
        ratio (int): Ratio of MLP. Default: 8
    """

    def __init__(self, in_channels, ratio=6):
        super(ChannelAttention, self).__init__()

        # Avg Pool & Max pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # MLP
        self.mlp = nn.Sequential(
            # Conv + LReLU + Conv (Replace Linear + LReLU + Linear)
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        # Sigmoid -> Weight(C*1*1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # B C H W
        # CA -> Weight
        w = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        w = self.sigmoid(w)  # B C 1 1
        # Weight * X
        x = w * x  # B C H W
        return x

class SpatialAttention(nn.Module):
    """
    Spatial Attention (SA)
    Args:
        kernel_size (int): Kernel size of Conv2d. Default: 7
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # Conv (B 2 H W -> B 1 H W)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        # Sigmoid -> Weight(1*H*W)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # B C H W
        # SA -> Weight
        avg_out = torch.mean(x, dim=1, keepdim=True)  # B 1 H W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # B 1 H W
        w = self.conv(torch.cat([avg_out, max_out], dim=1))  # B 2 H W -> B 1 H W
        w = self.sigmoid(w)  # B 1 H W
        # Weight * X
        x = w * x  # B C H W
        return x


class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
      #  print(inp_feats[1].shape)
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        inp = inp_feats[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
       # print(inp_feats[0].shape)
        feats = feats_V + inp

        return feats

class SKFF2(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF2, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
      #  print(inp_feats[1].shape)
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
       # print(inp_feats[0].shape)
        feats = feats_V

        return feats

class ResidualBlock(nn.Module):
    def __init__(self,  dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, hidden_dim // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim // 4, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.ca = ChannelAttention(dim, ratio=6)

    def forward(self, x):
       # out = self.ca(x)
        out = self.conv1(x)
      #  print(out.shape)
        out = self.bn1(out)
       # print(out.shape)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)

        out = out + x
        out = self.relu(out)

        return out

class ResidualBlock1(nn.Module):
    def __init__(self,  dim, hidden_dim):
        super(ResidualBlock1, self).__init__()
        self.conv1 = nn.Conv2d(dim, hidden_dim // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim // 4, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.sa = SpatialAttention()

    def forward(self, x):
       # out = self.ca(x)
        out = self.conv1(x)
      #  print(out.shape)
        out = self.bn1(out)
       # print(out.shape)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sa(out)

        out = out + x
        out = self.relu(out)

        return out

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self,
        inp_channels=3,
        dim = 24,
        embed_dim=192,
        embed_dim1=384,
        res = 128,
        infer_mode=False,
        hidden_dim = 0,
        drop_path = 0.1,
        attn_drop_rate = 0,
        mlp_ratio = 2.0,
        post_norm=True,
        layer_scale=None,
        headdim = 12,
        dark_kernel_size =1,
        drop = 0.1,
        d_state = 16,
        expand = 2,
        reduction_ratio = 1,
        nums = [2,2,2,2],
        num_outputs = 1,
        num_blocks = [2,2,2,2],
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.dark_channel = DarkChannel(dark_kernel_size)
        self.unet = UNetEncoder(dim=dim, n_channels=3)
       # self.vss = VssBlock(dim=dim, d_state=d_state, expand=1, hidden_dim=dim, drop_path=drop_path, headdim=headdim)
        self.sc = SCBlock(dim=1)
        self.sc1  = SCBlock(dim=2)
        self.skff1 = SKFF(24, 3)
        self.skff2 = SKFF(48, 3)
        self.skff3 = SKFF(96, 3)
        self.skff4 = SKFF(192, 3)
        self.sk1 = SKFF2(24, 2)
        self.sk2 = SKFF2(48,2)
        self.sk3 = SKFF2(96, 2)
        self.sk4 = SKFF2(192,2)
        self.res = ResidualBlock(dim, hidden_dim)


       # self.mcd = MaxColorDifferenceCalculator()
       # self.pp = PyramidPooling(levels=3)

        self.encoder_level1 = nn.Sequential(*[
            CrossMamba(dim=dim, d_state=d_state, expand=expand, hidden_dim=dim, drop_path=drop_path, attn_drop_rate=attn_drop_rate, headdim=headdim) for i in range(num_blocks[0])])
        self.vss1 = nn.Sequential(*[RSACM(dim=dim, d_state=d_state, expand=expand, hidden_dim=dim, drop_path=drop_path, headdim=headdim, h=224, w=224, window_size=28, shift_size=3) for i in range(nums[0])])
        # self.encoder_level1 = CrossMamba(dim=dim, d_state=d_state, expand=expand, hidden_dim=dim, drop_path=drop_path, attn_drop_rate=attn_drop_rate)
        self.ccm1 = nn.Sequential(*[RCACM(in_channels=dim, hidden_dim=dim, w=224, h=224, window_size=28, shift_size=3) for i in range (nums[0])])
        self.rsb1 = nn.Sequential(*[ResidualBlock(dim=dim, hidden_dim=dim) for i in range(nums[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            CrossMamba(dim=int(dim * 2 ** 1), d_state=d_state, expand=expand, hidden_dim=int(dim * 2 ** 1), drop_path=drop_path, attn_drop_rate=attn_drop_rate,  headdim=headdim) for i in range(num_blocks[1])])
        self.vss2 = nn.Sequential(*[RSACM(dim=int(dim * 2 ** 1), d_state=d_state,expand=expand, hidden_dim=int(dim * 2 ** 1), drop_path=drop_path,  headdim=headdim, h=112, w=112, window_size=14, shift_size=3) for i in range(nums[1])])
        self.ccm2 = nn.Sequential(*[RCACM(in_channels=int(dim * 2 ** 1), hidden_dim=int(dim *2 ** 1), w=112, h=112, window_size=14, shift_size=3) for i in range(nums[1])])
        self.rsb2 = nn.Sequential(
            *[ResidualBlock(dim=int(dim * 2 ** 1), hidden_dim=int(dim * 2 ** 1)) for i in range(nums[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            CrossMamba(dim=int(dim * 2 ** 2),d_state=d_state, expand=expand, hidden_dim=int(dim *2 ** 2), drop_path=drop_path, attn_drop_rate=attn_drop_rate,  headdim=headdim) for i in range(num_blocks[2])])
        self.vss3 = nn.Sequential(
            *[RSACM(dim=int(dim * 2 ** 2),d_state=d_state, expand=expand, hidden_dim=int(dim *2 ** 2), drop_path=drop_path,  headdim=headdim, h=56, w=56, window_size=7, shift_size=2) for i in range(nums[2])])
        self.ccm3 = nn.Sequential(
            *[RCACM(in_channels=int(dim * 2 ** 2), hidden_dim=int(dim * 2 ** 2), w=56, h=56, window_size=7, shift_size=2) for i in range(nums[2])])
        self.rsb3 = nn.Sequential(
            *[ResidualBlock(dim=int(dim * 2 ** 2), hidden_dim=int(dim * 2 ** 2)) for i in range(nums[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            CrossMamba(dim=int(dim * 2 ** 3), d_state=d_state, expand=expand, hidden_dim=int(dim * 2 ** 3), drop_path=drop_path, attn_drop_rate=attn_drop_rate,  headdim=headdim) for i in range(num_blocks[3])])
        self.vss4 = nn.Sequential(
            *[RSACM(dim=int(dim * 2 ** 3), d_state=d_state, expand=expand, hidden_dim=int(dim * 2 ** 3), drop_path=drop_path,   headdim=headdim, h=28, w =28, window_size=4, shift_size=1) for i in range(nums[3])])
        self.ccm4 = nn.Sequential(
            *[RCACM(in_channels=int(dim * 2 ** 3), hidden_dim=int(dim * 2 ** 3), w=28, h=28, window_size=4, shift_size=1) for i in range(nums[3])])
        self.rsb4 = nn.Sequential(
            *[ResidualBlock(dim=int(dim * 2 ** 3), hidden_dim=int(dim * 2 ** 3)) for i in range(nums[3])])

        self.embed_dim = embed_dim
      #  self.embed_dim1 = embed_dim1
      #  self.wight = WeightedSumModel()
        self.conv = nn.Conv2d(2, dim, kernel_size=1, bias=bias)
        self.conv1 = nn.Conv2d(1, dim, kernel_size=1, bias=bias)
        self.drop = drop
        self.relu = nn.ReLU()
      #  self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.weight = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 4, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 4, num_outputs),
            nn.Sigmoid()
        )


    def forward(self, inp_img):
        inp1 = self.patch_embed(inp_img)
        dark_map = self.dark_channel(inp_img)
       # x1, x2, x3, x4 = self.unet(dark_map)
        mcd = calculate_max_color_difference(inp_img)
      #  p1, p2, p3, p4 = self.unet(mcd)
        L_channels_map, ab_channels_map = extract_color_map(inp_img)
        s_channels_map = extract_hsv_color_map(inp_img)

        td = extract_gradient_map(inp_img)



       # x1, x2, x3, x4 = self.unet(s_channels_map)
       # color_map = torch.cat([ab_channels_map, hs_channels_map],dim=1)
      #  dl = torch.cat([dark_map, L_channels_map],dim=1)
       # L_channels_map = self.conv(L_channels_map)
        d11, d22, d33, d44 = self.unet(dark_map)
       # print("x11, x22, x33, x44 =", x11.shape, x22.shape, x33.shape, x44.shape)
        inp_enc_level1 = self.conv1(L_channels_map)
       # inp_enc_level1 = inp1 * self.mlp1(x1)
        out_enc_level1 = self.encoder_level1([inp_enc_level1, d11])
        inp_enc_level2 = self.down1_2(out_enc_level1[0])
      #  inp_enc_level2 = inp_enc_level2 * (self.weight2 * x2)
       # inp_enc_level2 = inp_enc_level2 * self.mlp2(x2)
        out_enc_level2 = self.encoder_level2([inp_enc_level2,d22])
        inp_enc_level3 = self.down2_3(out_enc_level2[0])
      #  inp_enc_level3 = inp_enc_level3 * (self.weight3 * x3)
      #  inp_enc_level3 = inp_enc_level3  * self.mlp3(x3)
        out_enc_level3 = self.encoder_level3([inp_enc_level3,d33])
        inp_enc_level4 = self.down3_4(out_enc_level3[0])
      #  inp_enc_level4 = inp_enc_level4 * (self.weight4 * x4)
      #  inp_enc_level4 = inp_enc_level4 * self.mlp4(x4)
        latent = self.latent([inp_enc_level4,d44])
       # print(latent.shape)
        latent = latent[0]
       # print("latent=", latent.shape)
        mcd = self.conv1(mcd)
        B, C, H, W = mcd.shape
        mcd = mcd.view(B, H, W, C)
        mcd = self.weight(mcd)
      #  print(mcd.shape)
        inp = self.sc1(ab_channels_map)
        B1, C1, H1, W1 = inp.shape
        inp = inp.view(B1,H1,W1,C1)
      #  print(inp.shape)
        inp = inp * mcd + inp
        inp = inp.view(B1, C1, H1, W1)
      #  inp = inp.view(0,1,2,3) + inp
        x11 = self.ccm1(inp)
        x22 = self.down1_2(x11)
        x33 = self.ccm2(x22)
        x44 = self.down2_3(x33)
        x55 = self.ccm3(x44)
        x66 = self.down3_4(x55)
        x77 = self.ccm4(x66)
      #  print("x77=", x77.shape)


      #  print(latent)
      #  print(latent.shape)c

       # latent = nn.array(latent)
       # print(latent.shape)
       # mcd = self.conv(mcd)
        x0 = self.sc(td)
       # x = self.conv1(color_map)
        x1 = self.vss1(x0)
        x2 = self.down1_2(x1)
        x3 = self.vss2(x2)
        x4 = self.down2_3(x3)
        x5 = self.vss3(x4)
        x6 = self.down3_4(x5)
        x7 = self.vss4(x6)
      #  print(x7.shape)

       # latent = self.skff([latent, x])

      #  x = self.sc(x)

       # x = mcd + ab_channels_map
      #  x = self.sc(self.vss(x))
      #  x = self.down3_4(self.down2_3(self.down1_2(x)))
       # print(x.shape)
      #  latent = torch.cat([latent, x],dim=1)
     #   latent = latent + x
       # latent = self.avg_pool(latent)
       # x = torch.cat([latent, x77, x7], dim=1)
      #  x = self.skff([x77, latent, x7])
        x = self.rsb1(inp1)
        l = self.skff1([out_enc_level1[0], x11, x1])
        x = self.sk1([x, l])
        x = self.down1_2(x)
        x = self.rsb2(x)
        l1 = self.skff2([out_enc_level2[0], x33, x3])
        x = self.sk2([x, l1])
        x = self.down2_3(x)
        x = self.rsb3(x)
        l2 = self.skff3([out_enc_level3[0], x55, x5])
        x = self.sk3([x, l2])
        x = self.down3_4(x)
        x = self.rsb4(x)
        l3 = self.skff4([latent, x77, x7])
        x = self.sk4([l3, x])
        b, c, h, w = x.shape
        x = x.view(b, h, w, c)

        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score

from thop import profile
if __name__ == '__main__':
    data = torch.randn([1, 3, 224, 224]).cuda()
    # model = VMBlock(32)#.cuda()
    model = Restormer().cuda()
    out = model(data)
    print("out =", out.shape)
    flops, params = profile(model, (data, ))
    print("flops: ", flops / 1e9, "params: ", params / 1e6)

