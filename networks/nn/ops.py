import os
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
__all__ = [
    "ConvLayer",
    "UpSampleLayer",
    "ConvPixelUnshuffleDownSampleLayer",
    "PixelUnshuffleChannelAveragingDownSampleLayer",
    "ConvPixelShuffleUpSampleLayer",
    "ChannelDuplicatingPixelUnshuffleUpSampleLayer",
    "LinearLayer",
    "IdentityLayer",
    'InceptionBlockWithResidual',
    "DSConv",
    "MBConv",
    "FusedMBConv",
    "ResBlock",
    "LiteMLA",
    "EfficientViTBlock",
    "ResidualBlock",
    "DAGBlock",
    "OpSequential",
]


from networks.nn.norm import build_norm
from networks.nn.act import build_act
from networks.utils import get_same_padding, resize, val2list, val2tuple, list_sum


#################################################################################
#                             Basic Layers                                      #
#################################################################################


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: Optional[Union[int, tuple[int, int], list[int]]] = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.factor)
        return x


class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x = self.conv(x)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################
class InceptionBlockWithResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_ratio: float = 0.25,
        norm="bn2d",
        act_func="relu"
    ):
        super(InceptionBlockWithResidual, self).__init__()

        mid_channels = int(out_channels * bottleneck_ratio)

        # 分支 1: 1x1
        self.branch1 = ConvLayer(
            in_channels, mid_channels,
            kernel_size=1,
            norm=norm,
            act_func=act_func
        )

        # 分支 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            ConvLayer(in_channels, mid_channels, kernel_size=1, norm=norm, act_func=act_func),
            ConvLayer(mid_channels, mid_channels, kernel_size=3, norm=norm, act_func=act_func)
        )

        # 分支 3: 1x1 -> 5x5 (使用两个3x3代替)
        self.branch3 = nn.Sequential(
            ConvLayer(in_channels, mid_channels, kernel_size=1, norm=norm, act_func=act_func),
            ConvLayer(mid_channels, mid_channels, kernel_size=3, norm=norm, act_func=act_func),
            ConvLayer(mid_channels, mid_channels, kernel_size=3, norm=norm, act_func=act_func)
        )

        # 分支 4: AvgPool -> 1x1
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels, mid_channels, kernel_size=1, norm=norm, act_func=act_func)
        )

        # 输出融合卷积（降维）
        self.output_conv = ConvLayer(
            in_channels=4 * mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm,
            act_func=act_func
        )

        # 残差连接（如果 in/out 不一致，进行对齐）
        self.residual_conv = (
            ConvLayer(in_channels, out_channels, kernel_size=1, norm=norm, act_func=None)
            if in_channels != out_channels else nn.Identity()
        )

        self.act = build_act(act_func)

    def forward(self, x):
        residual = self.residual_conv(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.output_conv(out)

        out = out + residual
        if self.act:
            out = self.act(out)
        return out



class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=(None, None, "ln2d"),
        act_func=("silu", "silu", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.glu_act = build_act(act_func[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            stride=stride,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    vis_counter = 0
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)



        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def visualize_tensor(self, tensor: torch.Tensor, name: str, max_samples=1):
        """
        可视化张量的统计信息和前几个样本
        tensor: 待可视化张量
        name: 张量名称
        max_samples: 可视化前几个 batch
        """
        print(
            f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}, min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}")
        # 如果是二维或三维可以画图（只取第一个样本）
        if tensor.dim() >= 3:
            sample = tensor[0]
            if sample.dim() == 2:  # [C, L]
                plt.imshow(sample.detach().cpu(), aspect='auto')
                plt.title(f"{name}")
                plt.colorbar()
                plt.show()
            elif sample.dim() == 3:  # [C, H, W]
                c = min(3, sample.size(0))
                for i in range(c):
                    plt.imshow(sample[i].detach().cpu(), cmap='viridis')
                    plt.title(f"{name}[channel {i}]")
                    plt.colorbar()
                    plt.show()

    def vis_spatial_map(
            self,
            x: torch.Tensor,
            H: int,
            W: int,
            name: str,
            channel_reduce: str = "mean",
            head_reduce: str = "mean",
    ):
        """
        x: [B, N, C, HW] or [B, N, C, H, W]
        """
        assert x.dim() in (4, 5)

        # 统一到 [B, N, C, HW]
        if x.dim() == 5:
            x = x.view(x.size(0), x.size(1), x.size(2), -1)

        # 通道维度聚合
        if channel_reduce == "mean":
            x = x.mean(dim=2)
        elif channel_reduce == "sum":
            x = x.sum(dim=2)
        elif channel_reduce == "l2":
            x = torch.norm(x, dim=2)
        else:
            raise ValueError(channel_reduce)

        # head / group 聚合
        if head_reduce == "mean":
            x = x.mean(dim=1)
        elif head_reduce == "sum":
            x = x.sum(dim=1)

        # 取第一个 batch
        x = x[0].view(H, W)

        # 归一化（便于可视化）
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)

        plt.figure(figsize=(4, 4))
        plt.imshow(x.detach().cpu(), cmap="jet")
        plt.title(name)
        plt.colorbar()
        plt.axis("off")
        plt.show()

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        # self.vis_spatial_map(qkv, H, W, "Q (channel-merged)")
        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        # ===== 可视化 Q K V（通道聚合）=====
        q_vis = q.reshape(B, -1, H, W)
        k_vis = k.reshape(B, -1, H, W)
        v_vis = v.reshape(B, -1, H, W)

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        qk_vis = q.reshape(B, -1, H, W)
        kk_vis = k.reshape(B, -1, H, W)


        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)

        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))
        vis_spatial_map(q_vis, "Q", reduce="norm")
        vis_spatial_map(k_vis, "K", reduce="norm")
        vis_spatial_map(v_vis, "V", reduce="norm")
        vis_spatial_map(qk_vis, "Q", reduce="norm")
        vis_spatial_map(kk_vis, "K", reduce="norm")
        vis_spatial_map(out, "Attention Map", reduce="norm")
        # Q / K / V（通道聚合后）
        save_attn_map_cv2(q_vis, f"q_before_kernel{LiteMLA.vis_counter}.png", reduce="norm")
        save_attn_map_cv2(k_vis, f"k_before_kernel{LiteMLA.vis_counter}.png", reduce="norm")
        save_attn_map_cv2(v_vis, f"v{LiteMLA.vis_counter}.png", reduce="norm")

        # kernel 后
        save_attn_map_cv2(qk_vis, f"q_after_kernel{LiteMLA.vis_counter}.png", reduce="norm")
        save_attn_map_cv2(kk_vis, f"k_after_kernel{LiteMLA.vis_counter}.png", reduce="norm")

        # 输出
        save_attn_map_cv2(out, f"attn_out{LiteMLA.vis_counter}.png", reduce="norm")
        LiteMLA.vis_counter +=1
        return out





    @torch.autocast(device_type="cuda", enabled=False)
    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        att_map = att_map.to(original_dtype)

        out = torch.matmul(v, att_map)  # b h d n

        out = torch.reshape(out, (B, -1, H, W))

        return out

    import torch


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)  # 1
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)

        return out


def vis_spatial_map(
        x: torch.Tensor,
        title: str,
        batch_idx: int = 0,
        reduce: str = "mean",
        cmap: str = "jet",
        save_path: str = None,
):
    """
    x: (B, C, H, W)
    """
    assert x.dim() == 4

    x = x.detach().cpu()

    if reduce == "mean":
        attn = x[batch_idx].mean(dim=0)
    elif reduce == "sum":
        attn = x[batch_idx].sum(dim=0)
    elif reduce == "norm":
        attn = torch.norm(x[batch_idx], p=2, dim=0)
    else:
        raise ValueError(f"Unknown reduce type: {reduce}")

    attn = attn.numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(attn, cmap=cmap)

    # ===== 关闭所有坐标轴 =====
    ax.axis("off")

    # ===== 标题放在下方 =====
    ax.text(
        0.5,
        -0.08,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontname="Times New Roman",
    )

    # ===== 可选：保留 colorbar（论文中常见）=====
    # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=9)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()





def save_attn_map_cv2(
    x: torch.Tensor,
    save_path: str,
    batch_idx: int = 0,
    reduce: str = "mean",
    colormap: int = cv2.COLORMAP_JET,
):
    """
    x: (B, C, H, W)
    """
    assert x.dim() == 4

    # ===== tensor -> numpy =====
    x = x.detach().cpu()

    if reduce == "mean":
        attn = x[batch_idx].mean(dim=0)
    elif reduce == "sum":
        attn = x[batch_idx].sum(dim=0)
    elif reduce == "norm":
        attn = torch.norm(x[batch_idx], p=2, dim=0)
    else:
        raise ValueError(f"Unknown reduce type: {reduce}")

    attn = attn.numpy()

    # ===== 归一化到 [0, 255] =====
    attn = attn - attn.min()
    attn = attn / (attn.max() + 1e-8)
    attn = (attn * 255).astype(np.uint8)

    # ===== 伪彩色映射（BGR）=====
    attn_color = cv2.applyColorMap(attn, colormap)

    # ===== 保存 =====
    cv2.imwrite(save_path, attn_color)






class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
        context_module: str = "LiteMLA",
        local_module: str = "MBConv",
    ):
        super(EfficientViTBlock, self).__init__()
        if context_module == "LiteMLA":
            self.context_module = ResidualBlock(
                LiteMLA(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    heads_ratio=heads_ratio,
                    dim=dim,
                    norm=(None, norm),
                    scales=scales,
                ),
                IdentityLayer(),
            )
        else:
            raise ValueError(f"context_module {context_module} is not supported")
        if local_module == "MBConv":
            self.local_module = ResidualBlock(
                MBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        elif local_module == "GLUMBConv":
            self.local_module = ResidualBlock(
                GLUMBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        else:
            raise NotImplementedError(f"local_module {local_module} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: Optional[nn.Module],
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: list[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
