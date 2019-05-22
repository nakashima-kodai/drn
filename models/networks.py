import torch
import torch.nn as nn
from .blocks import *


class DRNC26(nn.Module):
    def __init__(self, input_nc, norm, activation, pad_type):
        super(DRNC26, self).__init__()

        # level 1
        model = [Conv2dBlock(input_nc, 16, 7, 1, 3, norm, activation, pad_type)]
        model += [ResBlock(input_nc=16, output_nc=16, norm=norm)]
        # level 2
        model += [downResBlock(input_nc=16, output_nc=32, norm=norm)]
        # level 3
        model += [downResBlock(input_nc=32, output_nc=64, norm=norm)]
        model += [ResBlock(input_nc=64, output_nc=64, norm=norm)]
        # level 4
        model += [downResBlock(input_nc=64, output_nc=128, norm=norm)]
        model += [ResBlock(input_nc=128, output_nc=128, norm=norm)]
        # level 5
        model += [ResBlock2(input_nc=128, output_nc=256, dilation=2, norm=norm)]
        model += [ResBlock(input_nc=256, output_nc=256, dilation=2, norm=norm)]
        # level 6
        model += [ResBlock2(input_nc=256, output_nc=512, dilation=4, norm=norm)]
        model += [ResBlock(input_nc=512, output_nc=512, dilation=4, norm=norm)]
        # level 7
        model += [Conv2dBlock(input_nc=512, output_nc=512, kernel_size=3, padding=1, norm=norm, dilation=2)]
        model += [Conv2dBlock(input_nc=512, output_nc=512, kernel_size=3, padding=1, norm=norm, activation='none', dilation=2)]
        # level 8
        model += [Conv2dBlock(input_nc=512, output_nc=512, kernel_size=3, padding=1, norm=norm)]
        model += [Conv2dBlock(input_nc=512, output_nc=512, kernel_size=3, padding=1, norm=norm, activation='none')]

        self.model = nn.Sequential(*model)
        self.output_nc = 512

    def forward(self, x):
        return self.model(x)
