import numpy as np
import torch.nn as nn
from torch import Tensor
import torch
from transformer_layers import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward


class Conv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.LeakyReLU(inplace=True)
        )
        # self.cov1 = nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, padding=1)
        # self.layer_norm1 = nn.LayerNorm(output_channel)
        # self.relu1 = nn.LeakyReLU(inplace=True)
        # self.cov2 = nn.Conv1d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, padding=1)
        # self.layer_norm2 = nn.LayerNorm(output_channel)
        # self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # x = self.cov1(x)
        # x = x.permute(0, 2, 1)
        # x = self.layer_norm1(x)
        # x = x.permute(0, 2, 1)
        # x = self.relu1(x)
        # x = self.cov2(x)
        # x = x.permute(0, 2, 1)
        # x = self.layer_norm2(x)
        # x = x.permute(0, 2, 1)
        # x = self.relu2(x)

        return self.layer(x)


class DownSample(nn.Module):
    """下采样层
        1. 可选择：
        ->>model="conv"卷积的方式采样；用卷积将保留更多特征
        ->>model="maxPool"最大池化的方式进行采样。若采用该方法，将不用输入通道数目
        2. 默认使用卷积的方式进行下采样。
        3. 数据形状：
        ->> 输入: (batch, in_channel, image_h, image_w)
        ->> 输出: (batch, in_channel, image_h/2, image_w/2)
        4. 作用：将图像大小缩小一半"""

    def __init__(self, channel=None, model="conv"):
        super(DownSample, self).__init__()
        if model == "conv":
            self.layer = nn.Sequential(
                nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2, stride=2, bias=False),
                nn.LeakyReLU(inplace=True)
            )
        if model == "maxPool":
            self.layer = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    """上采样层"""

    def __init__(self, scale=2):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=scale)

    def forward(self, x):
        return self.up(x)


class Attention(nn.Module):
    """自注意力层"""

    def __init__(self, num_heads, size, dropout):
        super(Attention, self).__init__()
        # self.pe = PositionalEncoding(size, mask_count=True)
        self.att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=1024)

    def forward(self, x):
        # x = self.pe(x)
        return self.feed_forward(self.att(x, x, x) + x)


class Unet(nn.Module):
    def __init__(self, hidden_dim):
        super(Unet, self).__init__()
        self.d_c0 = Conv(1, 2)
        """"输入：(batch, 1, 512) -> 输出：(batch, 2, 512)"""
        self.d_s0 = DownSample(2)
        """输入：(batch, 2, 512) -> 输出：(batch, 2, 256)"""
        self.d_c1 = Conv(2, 4)
        """输入：(batch, 2, 256) -> 输出：(batch, 4, 256)"""
        self.d_s1 = DownSample(4)
        """输入：(batch, 4, 256) -> 输出：(batch, 4, 128)"""
        self.d_c2 = Conv(4, 8)
        """输入：(batch, 4, 128) -> 输出：(batch, 8, 128)"""
        self.middle = Conv(8, 4)
        """输入：(batch, 1024, 128) -> 输出：(batch, 512, 128)"""
        """输入：(batch, 8, 128) -> 输出：(batch, 4, 128)"""
        self.middle_up = UpSample()
        """输入：(batch, 512, 128) -> 输出：(batch, 512, 256)"""
        """输入：(batch, 8, 128) -> 输出：(batch, 4, 256)"""
        # -------------------------------上采样阶段-----------------------------------
        # 上采样阶段将图片还原
        self.u_c2 = Conv(8, 2)
        """输入：(batch, 8, 512) -> 输出：(batch, 2, 512)"""
        self.u_s2 = UpSample()
        """输入：(batch, 8, 512) -> 输出：(batch, 256, 256)"""

        self.u_c1 = Conv(4, 1)
        """输入：(batch, 4, 256) -> 输出：(batch, 1, 256)"""
        # ---------------------------------attention阶段---------------------------------------------
        self.att1 = Attention(8, 256, 0)
        self.att2 = Attention(8, 512, 0)
        self.att_mid = Attention(8, 128, 0)
        # 底层加权
        self.Up0 = UpSample()
        self.Up1 = UpSample()
        self.crop0 = nn.Sequential(
            Conv(10, 8),
            DownSample(8),
            Conv(8, 4),
            DownSample(4)
        )
        # 中层加权
        self.Up2 = UpSample()
        self.crop1 = nn.Sequential(
            Conv(6, 4),
            DownSample(4)
        )
        self.linear1 = nn.Linear(in_features=hidden_dim + 150 + 3, out_features=512)

        self.layer_norm = nn.LayerNorm(512, eps=1e-6)
        self.linear2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 150),
            # nn.Dropout(0.1),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 512))

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        time_emb = time_emb.repeat(1, x.shape[1], 1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        x = torch.cat([x, ctx_emb], dim=2)
        # ctx_emb = torch.cat([time_emb, context], dim=-1)
        x = self.linear1(x)
        x += self.pos_embedding
        """该处的输入是(batch, 1, 512)"""
        # x = torch.cat([x, time_emb], dim=2)
        # """在此处将输入改成(batch, 512, 1)"""
        # x = x.transpose(1, 2)
        d_c0_output = self.d_c0(x)
        """"输入：(batch, 1, 512) -> 输出：(batch, 2, 512)"""
        d_c1_output = self.d_c1(self.d_s0(d_c0_output))
        """"输入：(batch, 2, 512) -> 中间输出：(batch, 2, 256) -> 输出：(batch, 4, 256)"""
        d_c2_output = self.d_c2(self.d_s1(d_c1_output))
        """"输入：(batch, 4, 256) -> 中间输出：(batch, 4, 128) -> 输出：(batch, 8, 128)"""
        middle = self.middle(d_c2_output)
        """(batch, 4, 128)"""
        middle = self.att_mid(middle)
        d_c1_output = self.att1(d_c1_output)
        d_c0_output = self.att2(d_c0_output)

        # 底层加权
        x = torch.cat([d_c0_output, self.Up1(torch.cat([d_c1_output, self.Up0(middle)], dim=1))], dim=1)
        """(batch, 10, 512)"""
        middle = self.crop0(x)
        # 中层加权
        y = torch.cat([d_c0_output, self.Up2(d_c1_output)], dim=1)
        """(batch, 6, 512)"""
        d_c1_output = self.crop1(y)
        """(batch, 4, 256)"""

        middle_output = self.middle_up(middle)
        """"输入：(batch, 4, 128) -> 输出：(batch, 4, 256)"""

        # d_c1_output = self.att1(d_c1_output, middle_output)
        u_s2_output = self.u_s2(self.u_c2(self.cat(d_c1_output, middle_output)))
        """"输入：(batch, 8, 256)-> 中间输出：(batch, 2, 256) -> 输出：(batch, 2, 512)"""

        # d_c0_output = self.att2(d_c0_output, u_s2_output)
        output = self.u_c1(self.cat(d_c0_output, u_s2_output))
        output = self.linear2(self.layer_norm(output))
        """"输入：(batch, 2, 512) -> 中间输出：(batch, 4, 512) -> 输出：(batch, 1, 512)"""
        return output

    def cat(self, x1, x2):
        """在通道维度上组合"""
        return torch.cat([x1, x2], dim=1)
