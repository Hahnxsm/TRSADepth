import numpy as np
import torch
from torch import nn

from layers import ConvBlock, fSEModule, Conv1x1, Conv3x3
from .ms_sam import MSSAM
from .sru import SRU



class DepthDecoder(nn.Module):
    def __init__(self,
                 ch_enc = [64, 128, 256, 512, 1024],
                 num_ch_enc = [64, 64, 128, 256, 512 ],
                 num_output_channels=1):
        super(DepthDecoder, self).__init__()
        self.ch_enc = ch_enc # 对应的中间处理后的编码器特征通道
        self.num_ch_enc = num_ch_enc  # 编码器通道配置（实际输入特征的通道数）
        self.num_output_channels = num_output_channels     # 输出通道数（默认为深度图1通道）
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])  # 解码阶段的每层通道配置（从浅到深）
        # decoder
        self.convs = nn.ModuleDict()  # 用于存储各层卷积的模块容器

        # --------------- 特征融合部分（使用注意力模块） ---------------
        self.convs['f4'] = MSSAM(self.ch_enc[4],self.num_ch_enc[4])
        self.convs["f3"] = MSSAM(self.ch_enc[3], self.num_ch_enc[3])
        self.convs["f2"] = MSSAM(self.ch_enc[2], self.num_ch_enc[2])
        self.convs["f1"] = MSSAM(self.ch_enc[1], self.num_ch_enc[1])
        self.convs["f0"] = MSSAM(self.ch_enc[0], self.num_ch_enc[0])

        # ------------------- 解码器结构连接索引 -----------------------
        # 解码器中所有的连接位置，采用 X_ij 形式（第i层，第j列）
        # 每个字符串 "ij" 表示特征图的来源于：
        # 第 i 层（row，从下往上，0 表示最浅层、4 表示最深层）
        # 第 j 列（col，表示解码的阶段或第几次融合）
        self.all_position = ["01", "11", "21", "31",  # 第一列  # 512
                             "02", "12", "22",  # 第二列  # 256
                             "03", "13", # 第三列  # 128
                             "04"] # 第四列 # 64

        # ------------------- 构建所有中间层的卷积 ----------------------
        for j in range(5):  # 列 0，1，2，3，4
            for i in range(5 - j):  # 行
                # 构造X_ij中的Conv_0(降通道+上采样)
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:  # 01 02 03 04
                    num_ch_in /= 2
                num_ch_out = num_ch_in // 2
                self.convs[f'X_{i}{j}_Conv_0'] = SRU(int(num_ch_in), int(num_ch_out))

                # 只对X_04再添加额外的Conv_1层（输出最终解码特征）
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs[f'X_{i}{j}_Conv_1'] = ConvBlock(int(num_ch_in), int(num_ch_out))

        # 需要使用注意力增强的连接位置
        self.attention_position = ["31",
                                   "22",
                                   "13",
                                   "04"]  # 在每层的最后使用注意力进行特征融合
        # ------------------- 构建注意力和常规融合模块 -------------------
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])

            # 注意力模块使用fSEModule
            self.convs[f'X_{index}_attention'] = fSEModule(
                num_ch_enc[row + 1] // 2,
                self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1)
            )

        # 使用标准卷积堆叠的连接位置
        self.non_attention_position = ["01", "11", "21",  # 其他位置使用标准卷积进行融合
                                       "02", "12",
                                       "03"]
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            # 非注意力模块使用标准卷积或1*1降维后再卷积
            if col == 1:
                self.convs[f'X_{row + 1}{col - 1}_Conv_1'] = ConvBlock(
                    num_ch_enc[row + 1] // 2 + self.num_ch_enc[row],
                    self.num_ch_dec[row + 1],
                )
            else:
                self.convs[f'X_{index}_downsample'] = Conv1x1(
                    num_ch_enc[row + 1] // 2 + self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1),
                    self.num_ch_dec[row + 1] * 2,
                )
                self.convs[f'X_{row + 1}{col - 1}_Conv_1'] = ConvBlock(
                    self.num_ch_dec[row + 1] * 2,
                    self.num_ch_dec[row + 1],
                )

        # ------------------- 最终生成深度图的卷积 -------------------
        for i in range(4):
            self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        # 组织所有子模块
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nextConv(self,conv,high_feature,low_features,freqFusion=None):
        """
        嵌套卷积模块，输入高层特征与多个低层特征，融合后生成新特征
        """
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features,list)
        if freqFusion is None:
            high_features = [conv_0(high_feature)] # 上采样高层特征
            for feature in low_features:
                high_features.append(feature) # 拼接低层特征
            high_features = torch.cat(high_features, 1) # 通道维拼接
        else:
            pass
        if len(conv) == 3:
            high_features = conv[2](high_features)  # 下采样融合特征
        return conv_1(high_features)  # 最后进行卷积融合

    def forward(self, input_features, ):
        """
        :param input_features: 编码层特征[f0, f1, f2, f3, f4]  # (从浅到深)
        :return:
        """
        # 第一步：Attention 模块对特征增强
        feat ={}
        # feat[4] = self.convs["f4"](self.convs['f4_mscam'](input_features[4]))
        feat[4] = self.convs["f4"](input_features[4])
        feat[3] = self.convs["f3"](input_features[3])
        feat[2] = self.convs["f2"](input_features[2])
        feat[1] = self.convs["f1"](input_features[1])
        feat[0] = self.convs["f0"](input_features[0])
        # feat[0] = input_features[0]
        # 初始化 X_i0 特征 ->解码器每层的输入特征
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = feat[i]

        # 整个编码网格结构
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])
            # 获取当前位置需要融合的低层特征
            low_features = [features[f"X_{row}{i}"] for i in range(col)]  # 当前层前边的输出

            # 使用注意力路径
            if index in self.attention_position:
                features[f'X_{index}'] = self.convs[f'X_{index}_attention'](
                    self.convs[f'X_{row+1}{col-1}_Conv_0'](features[f'X_{row+1}{col-1}']),
                    low_features,
                )
            elif index in self.non_attention_position:
                conv = [
                    self.convs[f'X_{row+1}{col-1}_Conv_0'],
                    self.convs[f'X_{row+1}{col-1}_Conv_1'],
                ]
                if col != 1:
                    conv.append(self.convs[f'X_{index}_downsample'])
                features[f'X_{index}'] = self.nextConv(conv,features[f'X_{row+1}{col-1}'],low_features)

        x = features[f'X_04']
        x = self.convs['X_04_Conv_0'](x)
        x = self.convs['X_04_Conv_1'](x)

        # 多尺度视差输出
        outputs = {("disp", 0): self.sigmoid(self.convs["dispconv0"](x)),
                   ("disp", 1): self.sigmoid(self.convs["dispconv1"](features["X_04"])),
                   ("disp", 2): self.sigmoid(self.convs["dispconv2"](features["X_13"])),
                   ("disp", 3): self.sigmoid(self.convs["dispconv3"](features["X_22"]))}
        return outputs
