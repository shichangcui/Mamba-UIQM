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


class DarkChannel(nn.Module):
    def __init__(self, kernel_size=1):
        super(DarkChannel, self).__init__()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)
      #  self.omega = omega

    def forward(self, x):
        # x : (B, 3, H, W), in [-1, 1]
        # x = (x + 1.0) / 2.0
        H, W = x.size()[2], x.size()[3]

        # maximum among three channels
        x, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        x = self.unfold(x)  # (B, k*k, H*W)
        x = x.unsqueeze(1)  # (B, 1, k*k, H*W)

        # maximum in (k, k) patch
        dark_map, _ = x.min(dim=2, keepdim=False)  # (B, 1, H*W)
        x = dark_map.view(-1, 1, H, W)
      #  print(x)
       # x = 1 - self.omega * x
#        print(x.shape)
        return 1-x

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


class Mamba2(nn.Module):
    def __init__(
            self,
            d_model,
            d_conv=3,  # default to 3 for 2D
            conv_init=None,
            expand=2,
            headdim=12,  # default to 64
            ngroups=1,
            A_init_range=(1, 16),
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=1e-4,
            dt_limit=(0.0, float("inf")),
            learnable_init_states=False,
            activation="silu",  # default to silu
            bias=False,
            conv_bias=True,
            # Fused kernel and sharding options
            chunk_size=256,
            use_mem_eff_path=False,  # default to False, for custom implementation
            layer_idx=None,  # Absorb kwarg for general module
            device=None,
            dtype=None,
            linear_attn_duality=False,
            bidirection=True,
            d_state=16,
            **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim  # equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        # convert chunk_size to triton.language.int32
        self.chunk_size = chunk_size  # torch.tensor(chunk_size,dtype=torch.int32)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.bidirection = bidirection
        self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True)  # default to False, ablation for linear attn duality
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs)  #
        t_in_proj = 2 * self.d_inner +2* self.ngroups * self.d_state + self.nheads
        self.t_in_proj = nn.Linear(self.d_model, int(t_in_proj), bias=bias, **factory_kwargs)
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # modified from RMSNormGated to layer norm
        # assert RMSNormGated is not None
        # self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # linear attention duality
        self.linear_attn_duality = linear_attn_duality
        self.kwargs = kwargs

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        '''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        '''

        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3)  # (B, H, L, D)
        dt = dt.permute(0, 2, 1)  # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        if self.ssd_positve_dA: dA = -dA

        V_scaled = V * dA
        K = B.view(batch, 1, seqlen, dstate)  # (B, 1, L, D)
        if getattr(self, "__DEBUG__", False):
            A_mat = dA.cpu().detach().numpy()
            A_mat = A_mat.reshape(batch, -1, H, W)
            setattr(self, "__data__", dict(
                dA=A_mat, H=H, W=W, V=V, ))

        if self.ngroups == 1:
            ## get kv via transpose K and V
            KV = K.transpose(-2, -1) @ V_scaled  # (B, H, dstate, D)
            Q = C.view(batch, 1, seqlen, dstate)  # .repeat(1, head, 1, 1)
            x = Q @ KV  # (B, H, L, D)
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4)  # (B, 1, g, L, dstate)
            V_scaled = V_scaled.view(batch, head // self.ngroups, self.ngroups, seqlen, dim)  # (B, H//g, g, L, D)
            Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4)  # (B, 1, g, L, dstate)

            KV = K.transpose(-2, -1) @ V_scaled  # (B, H//g, g, dstate, D)
            x = Q @ KV  # (B, H//g, g, L, D)
            V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head // self.ngroups,
                                                                                self.ngroups, seqlen,
                                                                                dim)  # (B, H//g, g, L, D)
            x = x + V_skip  # (B, H//g, g, L, D)
            x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim)  # (B, L, H, D)
            x = x.contiguous()

        return x

    def forward(self, u, t,seq_idx=None):
        """
        u: (B,C,H,W)
        Returns: same shape as u
        """
        #  print(u.shape)
        B0, H, W, C0 = u.shape
        u = u.view(B0, H * W, C0)
        batch, seqlen, dim = u.shape
        #   print(u.shape)
        B2, H2, W2, C2 = t.shape
        t = t.view(B2, H2 * W2, C2)
        batch2, seqlen2, dim2 = t.shape

        zxbcdt = self.in_proj(u) # (B, L, d_in_proj)
        z1x1b1c1dt1 = self.t_in_proj(t)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        z1, x1B1C1, dt1 = torch.split(
            z1x1b1c1dt1, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        #  print(z.shape)
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]

        # 2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H * W, -1).contiguous()

        x1B1C1 = x1B1C1.view(batch2, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        x1B1C1 = self.act(self.conv2d(x1B1C1))
        x1B1C1 = x1B1C1.permute(0, 2, 3, 1).view(batch2, H2 * W2, -1).contiguous()
        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C= torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x1, B1, C1 = torch.split(x1B1C1, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        C = C1
        x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
        if self.linear_attn_duality:
            y = self.non_casual_linear_attn(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt, A, B, C, self.D, H, W
            )
        else:
            if self.bidirection:
                # if self.kwargs.get('bidirection', False):
                # assert self.ngroups == 2 #only support bidirectional with 2 groups
                x = to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)).chunk(2, dim=-2)
                B = to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                C = to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                dt = dt.chunk(2, dim=-1)  # (B, L, nheads) -> (B, L, nheads//2)*2
                A, D = A.chunk(2, dim=-1), self.D.chunk(2, dim=-1)  # (nheads) -> (nheads//2)*2
                y_forward = mamba_chunk_scan_combined(
                    x[0], dt[0], A[0], B[0], C[0], chunk_size=self.chunk_size, D=D[0], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y_backward = mamba_chunk_scan_combined(
                    x[0].flip(1), dt[0].flip(1), A[1], B[0].flip(1), C[0].flip(1), chunk_size=self.chunk_size, D=D[1],
                    z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y = torch.cat([y_forward, y_backward.flip(1)], dim=-2)
            else:
                y = mamba_chunk_scan_combined(
                    to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)),
                    to_ttensor(dt),
                    to_ttensor(A),
                    to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)),
                    to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)),
                    chunk_size=self.chunk_size,
                    D=to_ttensor(self.D),
                    z=None,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
        y = rearrange(y, "b l h p -> b l (h p)")

        # # Multiply "gate" branch and apply extra normalization layer
        # y = self.norm(y, z)
        y = self.norm(y)
        #  print(y.shape)
        y = y * z
        out = self.out_proj(y)
       # print(out.shape)
        out = out.view(B0, H, W, C0)
        # print(out.shape)
        return out



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

class ChannelMamba(nn.Module):
    def __init__(
        self,
        d_model,
        dim=None,
        d_state=16,
        d_conv=4,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        bimamba_type="v2",
        if_devide_out=False
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.if_devide_out = if_devide_out
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.bimamba_type = bimamba_type
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(normalized_shape=dim)
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.conv2d = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            bias=conv_bias,
            kernel_size=3,
            groups=dim,
            padding=1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

    def forward(self, u):
        """
        u: (B, H, W, C)
        Returns: same shape as hidden_states
        """
        b, d, h, w = u.shape
        l = h * w
        u = rearrange(u, "b d h w-> b (h w) d").contiguous()

        conv_state, ssm_state = None, None

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(u, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=l,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        x, z = xz.chunk(2, dim=1)
        x = rearrange(self.conv2d(rearrange(x, "b l d -> b d 1 l")), "b d 1 l -> b l d")

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=l)
        B = rearrange(B, "(b l) d -> b d l", l=l).contiguous()
        C = rearrange(C, "(b l) d -> b d l", l=l).contiguous()

        x_dbl_b = self.x_proj_b(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b = self.dt_proj_b.weight @ dt_b.t()
        dt_b = rearrange(dt_b, "d (b l) -> b d l", l=l)
        B_b = rearrange(B_b, "(b l) d -> b d l", l=l).contiguous()
        C_b = rearrange(C_b, "(b l) d -> b d l", l=l).contiguous()
        if self.bimamba_type == "v1":
            A_b = -torch.exp(self.A_b_log.float())
            out = selective_scan_fn(
                x,
                dt,
                A_b,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
        elif self.bimamba_type == "v2":
            A_b = -torch.exp(self.A_b_log.float())
            out = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            out_b = selective_scan_fn(
                x.flip([-1]),
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            out = self.ln(out) * self.act(z)
            out_b = self.ln1(out_b) * self.act(z)
            if not self.if_devide_out:
                out = rearrange(out + out_b.flip([-1]), "b l (h w) -> b l h w", h=h, w=w)
            else:
                out = rearrange(out + out_b.flip([-1]), "b l (h w) -> b l h w", h=h, w=w) / 2

        return out

class CCOSS(nn.Module):
    def __init__(self,channel, w, h, d_state=16, expand=1, d_conv=4, mam_block=2):
        super().__init__()

        self.H_CSSM = nn.Sequential(*[
            ChannelMamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=h,
                dim=channel,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
            ) for i in
            range(mam_block)])

        self.W_CSSM = nn.Sequential(*[
            ChannelMamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=w,
                dim=channel,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
            ) for i in
            range(mam_block)])

        self.channel = channel
        self.ln = nn.LayerNorm(normalized_shape=channel)
        self.softmax = nn.Softmax(1)

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,
                                  bias=False)

        self.dwconv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, groups=channel,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel)

        self.silu_h = nn.SiLU()
        self.silu_w = nn.SiLU()

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
       # print(x.shape)
        x_s = x.contiguous()
        b, c, w, h = x.shape
        x = rearrange(x, "b c h w-> b (h w) c ")
      #  x = (self.ln(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
       # print((x.shape))
        x_in = x
        x_shotcut = self.softmax(self.dwconv(x))
        x_h = torch.mean(x_in, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x_in, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        x_h = x_cat_conv_split_h.permute(0, 3, 2, 1)#input=[b, h, 1, c]
        x_h= self.H_CSSM(x_h).permute(0, 3, 2, 1)
        x_w = x_cat_conv_split_w.permute(0, 3, 2, 1)
        x_w = self.W_CSSM(x_w).permute(0, 3, 2, 1)  #input=[b, w, 1, c]
        s_h = self.sigmoid_h(x_h.permute(0, 1, 3, 2))
        s_w = self.sigmoid_w(x_w)
        out = s_h.expand_as(x) * s_w.expand_as(x) * x_shotcut

        return out + x_s

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


class SS2DChannel(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()

        # 修改为通道上的投影
        self.x_proj = (
            nn.Linear(self.d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        # x: [B, C]
        B, C = x.shape
        K = 2  # 只保留两个方向（正向和反向）

        # 投影到时间步特征
        x_dbl = torch.einsum("b c, k c d -> b k d", x, self.x_proj_weight)  # [B, K, d]
        dts, Bs = torch.split(x_dbl, [self.dt_rank, self.d_state], dim=1)
        dts = torch.einsum("b k r, k d r -> b k d", dts, self.dt_projs_weight)  # [B, K, d]
        Bs = Bs.view(B, K, -1)  # [B, K, D]

        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # [D,]
        Ds = self.Ds.float().view(-1)  # [D,]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # [K * d]

        # 使用 selective_scan
        out_y = self.selective_scan(
            x.unsqueeze(1), dts, As, Bs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1)  # [B, K, d]

        # 合并正向和反向的结果
        y = out_y[:, 0] + out_y[:, 1]  # [B, d]

        # 输出投影
        y = self.out_norm(y)
        y = self.out_proj(y)  # [B, C]

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        # x: [B, C]
        B, C = x.shape

        # 输入投影
        xz = self.in_proj(x)  # [B, 2 * C]
        x, z = xz.chunk(2, dim=-1)

        # 核心逻辑
        y = self.forward_core(x)  # [B, C]

        # 输出投影和 dropout
        out = y * F.silu(z)  # 激活函数
        if self.dropout is not None:
            out = self.dropout(out)

        return out

class VssBlockChannel(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: callable = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 1.,
            headdim: int = 12,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(dim)  # 对通道维度进行归一化
        self.self_attention = SS2DChannel(
            d_model=dim,
            d_state=d_state,
            expand=expand,
            hidden_dim=dim,
            drop_path=drop_path,
            headdim=headdim,
            d_model=dim,
            is_light_sr=is_light_sr,
            **kwargs
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input):
        # x [B, C, H, W]
        input_0 = input  # 保存输入，用于残差连接
        B, C, H, W = input.shape

        # 将空间维度压缩为通道特征
        x = input.mean(dim=(2, 3))  # [B, C]

        # 自注意力操作
        x = self.drop_path(self.self_attention(x))  # [B, C]

        # 残差连接
        x = x + input_0.mean(dim=(2, 3))  # 保持形状一致

        return x




from thop import profile
if __name__ == '__main__':
    data = torch.randn([1, 3, 224, 224]).cuda()
    # model = VMBlock(32)#.cuda()
    model = Restormer().cuda()
    out = model(data)
    print("out =", out.shape)
    flops, params = profile(model, (data, ))
    print("flops: ", flops / 1e9, "params: ", params / 1e6)

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
        K = 4  # 假设 K 是扫描的方向数

        # 将通道维度 C 作为扫描的主要维度
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(B, H * W, C)  # (B, L, C)，其中 L = H * W

        # 对通道维度进行投影
        x_dbl = torch.einsum("b l c, k c d -> b k l d", x, self.x_proj_weight)  # (B, K, L, dt_rank + 2 * d_state)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # 对 dt 进行投影
        dts = torch.einsum("b k l r, k d r -> b k l d", dts, self.dt_projs_weight)  # (B, K, L, d_inner)

        # 将张量转换为适合扫描的形状
        xs = x.float().view(B, -1, C)  # (B, L, C)
        dts = dts.contiguous().float().view(B, K, -1, C)  # (B, K, L, C)
        Bs = Bs.float().view(B, K, -1, C)  # (B, K, d_state, C)
        Cs = Cs.float().view(B, K, -1, C)  # (B, K, d_state, C)

        # 初始化 A 和 D
        Ds = self.Ds.float().view(-1)  # (K * d_inner)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (K * d_inner, d_state)

        # 在通道维度上进行选择性扫描
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),  # (K * d_inner)
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, C)  # (B, K, L, C)

        # 恢复空间维度
        out_y = out_y.view(B, K, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, K, H, W)

        return out_y[:, :, 0], out_y[:, :, 1], out_y[:, :, 2], out_y[:, :, 3]


def forward(self, x: torch.Tensor, **kwargs):
    B, H, W, C = x.shape

    # 投影输入
    xz = self.in_proj(x)
    x, z = xz.chunk(2, dim=-1)

    # 调整形状并应用卷积
    x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
    x = self.act(self.conv2d(x))

    # 在通道维度上进行选择性扫描
    y1, y2, y3, y4 = self.forward_core(x)

    # 合并结果
    y = y1 + y2 + y3 + y4  # (B, C, H, W)
    y = y.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

    # 归一化和输出投影
    y = self.out_norm(y)
    y = y * F.silu(z)
    out = self.out_proj(y)

    if self.dropout is not None:
        out = self.dropout(out)

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