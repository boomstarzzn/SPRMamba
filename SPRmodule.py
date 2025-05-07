import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SF(torch.nn.Module):
    def __init__(self, in_ch):
        super(SF, self).__init__()
        # self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)  # generate k by conv
    def forward(self, x):
        _, _, h, w = x.size()
        q = x.mean(dim=[2, 3], keepdim=True)
        k = x
        square = (k - q).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + np.finfo(np.float32).eps) + 0.5
        att_weight = nn.Sigmoid()(att_score)

        return x * att_weight

class SPR(nn.Module):
    def __init__(self, dim, hidden_dim, embed_dim, memory_size=4):
        super(SSBWithCMM, self).__init__()
        self.hidden_dim = hidden_dim
        self.cmm = CMM(dim, embed_dim, memory_size)
        self.conv_gates = nn.Conv2d(2 * dim, 2 * hidden_dim, kernel_size=1)
        self.alpha_param = nn.Parameter(torch.tensor(0.5))
        self.beta_param = nn.Parameter(torch.tensor(0.5))
        self.fusion_activation = nn.GELU()
        self.spfilter = SF(dim)
    def forward(self, x, y):

        diff_fea = torch.abs(x - y)
        diff_sp = self.spfilter(diff_fea) ## B C H W
        significant_weight, nonsignificant_weight = self.cmm(diff_fea)

        fused_fea = torch.cat([x, y], dim=1)  # [B, 2 * C, H, W]
        fuse_conv = self.conv_gates(fused_fea)  # [B, 2 * hidden_dim, H, W]
        salien_fea, nonsalien_fea = torch.split(fuse_conv, self.hidden_dim, dim=1)

        salien_fea = salien_fea * significant_weight
        nonsalien_fea = nonsalien_fea * nonsignificant_weight

        alpha, beta = torch.softmax(torch.cat([self.alpha_param.unsqueeze(0), self.beta_param.unsqueeze(0)], dim=0),
                                    dim=0)
        final_fuse = alpha * salien_fea + beta * nonsalien_fea

        return self.fusion_activation(final_fuse)

class CMM(nn.Module):
    def __init__(self, in_channels, embed_dim, memory_size=4):
        super(CMM, self).__init__()
        self.memory_units = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1) for _ in range(memory_size)
        ])
        #self.predictor = Predictor(embed_dim=embed_dim)  
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )
    def forward(self, x):

        if len(x.shape) == 4:
            B, C, H, W = x.size()
            x_rs = x.reshape(B, C, -1).permute(0, 2, 1)
        else:
            B, N, C = x.size()
            H = int(N**0.5)
            x_rs = x
        x_rs = self.in_conv(x_rs)
        B, N, C = x_rs.size()

        window_scale = int(H//2)
        local_x = x_rs[:, :, :C // 2]
        global_x = x_rs[:, :, C // 2:].view(B, H, -1, C // 2).permute(0, 3, 1, 2)
        global_x_avg = F.adaptive_avg_pool2d(global_x,  (2, 2)) # [b, c, 2, 2]
        global_x_avg_concat = F.interpolate(global_x_avg, scale_factor=window_scale)
        global_x_avg_concat = global_x_avg_concat.view(B, C // 2, -1).permute(0, 2, 1).contiguous()

        x_rs = torch.cat([local_x, global_x_avg_concat], dim=-1)

        x_score = self.out_conv(x_rs)
        mapping = x_score.permute(0, 2, 1).reshape(B, 2, H, -1)

        significant_weight = mapping[:, 0:1, :, :]
        nonsignificant_weight = mapping[:, 1:2, :, :]


        return significant_weight, nonsignificant_weight
