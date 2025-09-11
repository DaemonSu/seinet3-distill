# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_raw = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_abs = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

        self.conv_fft = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # shpae of x : [batch,channel,axis1]
        x_raw = x[:, 0:x.shape[1]//3, :]
        x_abs = x[:, x.shape[1]//3:x.shape[1]*2//3, :]
        x_fft = x[:, x.shape[1]*2//3:x.shape[1], :]
        raw = self.conv_raw(x_raw)
        fft = self.conv_fft(x_fft)
        abs= self.conv_abs(x_abs)
        output = torch.cat((raw, abs,fft), dim=1)
        return output
