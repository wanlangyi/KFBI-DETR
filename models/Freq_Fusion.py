import torch
import torch.nn as nn
from models.kan import channel_shuffle, _init_weights
from timm.models.helpers import named_apply
from functools import partial


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        # self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
        #                                   kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.conv_layer = Partial_conv3(dim=in_channels*2, n_div=4, forward='split_cat')
        self.bn = torch.nn.BatchNorm2d(out_channels*2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class Freq_Fusion(nn.Module):
    def __init__(self, dim):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.FFC = FourierUnit(self.dim, self.dim)
        self.bn = torch.nn.BatchNorm2d(self.dim)
        self.relu = torch.nn.ReLU(inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.FFC(x) + x
        x = self.relu(self.bn(x))
        return x


class Fourier_Fusion_Mixer(nn.Module):
    def __init__(self, dim):
        super(Fourier_Fusion_Mixer, self).__init__()
        self.dim = dim

        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )

        self.fusion_mixer = Freq_Fusion(dim=dim*2)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x, g):
        x_dw = self.dw_conv_1(x)
        g_dw = self.dw_conv_2(g)
        x_cat = torch.cat([x_dw, g_dw], dim=1)
        x_cat = channel_shuffle(x_cat, 64)
        x_fusion = self.fusion_mixer(x_cat)
        x_out = self.ca_conv(x_fusion)
        x_out = self.ca(x_out) * x_out

        return x_out


if __name__ == '__main__':
    input_dim = 256
    output_dim = 256
    model = Fourier_Fusion_Mixer(dim=256)
    dummy_input1 = torch.randn(2, input_dim, 64, 64)
    dummy_input2 = torch.randn(2, input_dim, 64, 64)

    output = model(dummy_input1, dummy_input2)
    print("features shape:", output.size())
