from typing import Optional, Any, Union, List

import torch
import torch.nn as nn
from torch import Tensor

from networks.nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
)
from networks.utils import build_kwargs_from_config

__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
    "efficientvit_backbone_l3",
    'trsa_s',
    'trsa_b',
    'trsa_l',
]


class EfficientViTBackbone(nn.Module):
    def __init__(
            self,
            width_list: list[int],
            depth_list: list[int],
            in_channels=3,
            dim=32,
            expand_ratio=4,
            norm="bn2d",
            act_func="hswish",
    ) -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: float,
            norm: str,
            act_func: str,
            fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def init_weights(self, pretrained=None):
        """加载预训练权重（兼容 MMDetection / MMSeg）
        Args:
            pretrained (str or None): 预训练模型路径，支持 .pth/.pt 文件。
        """
        if pretrained is None:
            print(f"[Info] No pretrained weights specified for {self.__class__.__name__}, training from scratch.")
            return
        print(f"[Info] Loading pretrained weights from {pretrained} for {self.__class__.__name__}...")
        state_dict = torch.load(pretrained, map_location='cpu')

        # 一些预训练模型会嵌套在 state_dict / model 字段里
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']['backbone']

        # 如果是用 DDP/DDPWrapper 训练的可能带有 "module." 前缀
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if
                      k.startswith('backbone.')}

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

        print(
            f"[Info] Loaded pretrained weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")
        if missing_keys:
            print(f" - Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f" - Unexpected keys: {unexpected_keys}")

    def forward(self, x: torch.Tensor):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        fusion_out = []
        for stage_id in range(5):
            fusion_out.append(output_dict["stage%d" % stage_id])
        return fusion_out


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def trsa_s(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def trsa_b(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def trsa_l(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 4, 6, 6, 9],
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


class EfficientViTLargeBackbone(nn.Module):
    def __init__(
            self,
            width_list: list[int],
            depth_list: list[int],
            block_list: Optional[list[str]] = None,
            expand_list: Optional[list[float]] = None,
            fewer_norm_list: Optional[list[bool]] = None,
            in_channels=3,
            qkv_dim=32,
            norm="bn2d",
            act_func="gelu",
    ) -> None:
        super().__init__()
        block_list = ["res", "fmb", "fmb", "mb", "att"] if block_list is None else block_list
        expand_list = [1, 4, 4, 4, 6] if expand_list is None else expand_list
        fewer_norm_list = [False, False, False, True, True] if fewer_norm_list is None else fewer_norm_list

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(
                        EfficientViTBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    block = ResidualBlock(block, IdentityLayer())
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    def init_weights(self, pretrained=None):
        """加载预训练权重（兼容 MMDetection / MMSeg）
        Args:
            pretrained (str or None): 预训练模型路径，支持 .pth/.pt 文件。
        """
        if pretrained is None:
            print(f"[Info] No pretrained weights specified for {self.__class__.__name__}, training from scratch.")
            return
        print(f"[Info] Loading pretrained weights from {pretrained} for {self.__class__.__name__}...")
        state_dict = torch.load(pretrained, map_location='cpu')

        # 一些预训练模型会嵌套在 state_dict / model 字段里
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']['backbone']

        # 如果是用 DDP/DDPWrapper 训练的可能带有 "module." 前缀
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if
                      k.startswith('backbone.')}

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

        print(
            f"[Info] Loaded pretrained weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")
        if missing_keys:
            print(f" - Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f" - Unexpected keys: {unexpected_keys}")

    @staticmethod
    def build_local_block(
            block: str,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: float,
            norm: str,
            act_func: str,
            fewer_norm: bool = False,
    ) -> nn.Module:
        if block == "res":
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "fmb":
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "mb":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        else:
            raise ValueError(block)
        return block

    def forward(self, x: torch.Tensor):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)

        x = (x - mean) / std
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        fusion_out = []
        for stage_id in range(len(self.stages)):
            fusion_out.append(output_dict["stage%d" % stage_id])
        return fusion_out


def efficientvit_backbone_l0(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l1(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )

    return backbone


def efficientvit_backbone_l2(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l3(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone
