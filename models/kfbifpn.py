import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from mmcv.cnn.bricks import Swish
from typing import Tuple, Union
from timm.models.helpers import named_apply
from models.kan import KANConv2DLayer, act_layer, _init_weights, channel_shuffle
from models.Freq_Fusion import Fourier_Fusion_Mixer


def resize_feature_map(feature_map, target_height, target_width):
    """
    Resize the feature map to the target height and width by padding or cropping.

    Args:
    - feature_map (torch.Tensor): The input feature map with shape (batch_size, channels, height, width).
    - target_height (int): The target height.
    - target_width (int): The target width.

    Returns:
    - torch.Tensor: The resized feature map with the same number of channels.
    """
    batch_size, channels, height, width = feature_map.size()

    # Calculate padding or cropping needed
    pad_height = max(target_height - height, 0)
    pad_width = max(target_width - width, 0)
    crop_height = max(height - target_height, 0)
    crop_width = max(width - target_width, 0)

    # Padding
    if pad_height > 0 or pad_width > 0:
        padding = (pad_width // 2, pad_width - pad_width // 2,  # width padding
                   pad_height // 2, pad_height - pad_height // 2)  # height padding
        feature_map = F.pad(feature_map, padding, mode='constant', value=0)  # Pad with 0s

    # Cropping
    if crop_height > 0 or crop_width > 0:
        feature_map = feature_map[:, :, crop_height // 2:height - (crop_height - crop_height // 2),
                      crop_width // 2:width - (crop_width - crop_width // 2)]

    return feature_map


class DCUB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(DCUB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


class MaxPool2dSamePadding(nn.Module):

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 2,
                 **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) -
                   1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) -
                   1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])
        x = self.pool(x)

        return x


class BiFPNStage(nn.Module):
    """
        in_channels: List[int], input dim for P3, P4, P5
        out_channels: int, output dim for P2 - P7
        first_time: int, whether is the first bifpnstage
        num_outs: int, BiFPN need feature maps num
        use_swish: whether use MemoryEfficientSwish
        norm_cfg: (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer.
        epsilon: float, hyperparameter in fusion features
    """

    def __init__(self, in_channels=[256, 256, 256, 256, 256], out_channels=256, conv_bn_act_pattern=False, epsilon=1e-4):
        super(BiFPNStage, self).__init__()

        assert isinstance(in_channels, list)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.epsilon = epsilon
        self.eucb_ks = 3

        # up to bottom: feature map up_sample module
        self.f3_up_sample = DCUB(in_channels=in_channels[4], out_channels=out_channels, kernel_size=self.eucb_ks,
                                 stride=self.eucb_ks // 2)
        self.f2_up_sample = DCUB(in_channels=in_channels[3], out_channels=out_channels, kernel_size=self.eucb_ks,
                                 stride=self.eucb_ks // 2)
        self.f1_up_sample = DCUB(in_channels=in_channels[2], out_channels=out_channels, kernel_size=self.eucb_ks,
                                 stride=self.eucb_ks // 2)
        self.f0_up_sample = DCUB(in_channels=in_channels[1], out_channels=out_channels, kernel_size=self.eucb_ks,
                                 stride=self.eucb_ks // 2)

        # bottom to up: feature map down_sample module
        self.f4_down_sample = MaxPool2dSamePadding(3, 2)
        self.f3_down_sample = MaxPool2dSamePadding(3, 2)
        self.f2_down_sample = MaxPool2dSamePadding(3, 2)
        self.f1_down_sample = MaxPool2dSamePadding(3, 2)

        self.mscb0 = KANConv2DLayer(in_channels[0], out_channels, kernel_size=3, spline_order=3, groups=1, padding=1,
                                    stride=1, dilation=1)
        self.mscb1 = KANConv2DLayer(in_channels[1], out_channels, kernel_size=3, spline_order=3, groups=1, padding=1,
                                    stride=1, dilation=1)
        self.mscb2 = KANConv2DLayer(in_channels[2], out_channels, kernel_size=3, spline_order=3, groups=1, padding=1,
                                    stride=1, dilation=1)
        self.mscb3 = KANConv2DLayer(in_channels[3], out_channels, kernel_size=3, spline_order=3, groups=1, padding=1,
                                    stride=1, dilation=1)
        self.mscb4 = KANConv2DLayer(in_channels[4], out_channels, kernel_size=3, spline_order=3, groups=1, padding=1,
                                    stride=1, dilation=1)

        self.ffm0 = Fourier_Fusion_Mixer(dim=in_channels[0])
        self.ffm1 = Fourier_Fusion_Mixer(dim=in_channels[1])
        self.ffm2 = Fourier_Fusion_Mixer(dim=in_channels[2])
        self.ffm3 = Fourier_Fusion_Mixer(dim=in_channels[3])
        self.ffm4 = Fourier_Fusion_Mixer(dim=in_channels[4])

        # up to bottom weights
        self.f0_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.f0_w1_relu = nn.ReLU()

        self.f1_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.f1_w1_relu = nn.ReLU()

        self.f2_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.f2_w1_relu = nn.ReLU()

        self.f3_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.f3_w1_relu = nn.ReLU()

        # bottom to up weights
        self.f1_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.f1_w2_relu = nn.ReLU()

        self.f2_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.f2_w2_relu = nn.ReLU()

        self.f3_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.f3_w2_relu = nn.ReLU()

        self.f4_w2 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.f4_w2_relu = nn.ReLU()

        # self.swish = MemoryEfficientSwish() if use_meswish else Swish()
        self.swish = Swish()

    def combine(self, x):
        if not self.conv_bn_act_pattern:
            x = self.swish(x)

        return x

    def forward(self, x):
        f0_in, f1_in, f2_in, f3_in, f4_in = x

        # Weights for f3_0 and f4_0 to f3_1
        f3_w1 = self.f3_w1_relu(self.f3_w1)
        weight = f3_w1 / (torch.sum(f3_w1, dim=0) + self.epsilon)
        # Connections for f1_0 and f0_0 to f1_1 respectively
        # f3_up = self.mscb3(
        #     self.combine(weight[0] * f3_in +
        #                  weight[1] * resize_feature_map(self.f3_up_sample(f4_in), f3_in.size()[2], f3_in.size()[3])))  # (1,64,8,8)
        f3_us = resize_feature_map(self.f3_up_sample(f4_in), f3_in.size()[2], f3_in.size()[3])
        f3_in_ffm = self.ffm3(f3_in, f3_us)
        f3_up = self.mscb3(self.combine(weight[0] * f3_in_ffm + weight[1] * f3_us))  # (1,64,8,8)

        # Weights for P5_0 and P6_1 to P5_1
        f2_w1 = self.f2_w1_relu(self.f2_w1)
        weight = f2_w1 / (torch.sum(f2_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        # f2_up = self.mscb2(
        #     self.combine(weight[0] * f2_in +
        #                  weight[1] * resize_feature_map(self.f2_up_sample(f3_up), f2_in.size()[2], f2_in.size()[3])))  # (1,64,16,16)
        f2_us = resize_feature_map(self.f2_up_sample(f3_up), f2_in.size()[2], f2_in.size()[3])
        f2_in_ffm = self.ffm2(f2_in, f2_us)
        f2_up = self.mscb2(self.combine(weight[0] * f2_in_ffm + weight[1] * f2_us))  # (1,64,16,16)

        # Weights for P4_0 and P5_1 to P4_1
        f1_w1 = self.f1_w1_relu(self.f1_w1)
        weight = f1_w1 / (torch.sum(f1_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        # f1_up = self.mscb1(
        #     self.combine(weight[0] * f1_in +
        #                  weight[1] * resize_feature_map(self.f1_up_sample(f2_up), f1_in.size()[2], f1_in.size()[3])))  # (1,64,32,32)
        f1_us = resize_feature_map(self.f1_up_sample(f2_up), f1_in.size()[2], f1_in.size()[3])
        f1_in_ffm = self.ffm1(f1_in, f1_us)
        f1_up = self.mscb1(self.combine(weight[0] * f1_in_ffm + weight[1] * f1_us))  # (1,64,32,32)

        # Weights for P3_0 and P4_1 to P3_2
        f0_w1 = self.f0_w1_relu(self.f0_w1)
        weight = f0_w1 / (torch.sum(f0_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        # f0_out = self.mscb0(
        #     self.combine(weight[0] * f0_in +
        #                  weight[1] * resize_feature_map(self.f0_up_sample(f1_up), f0_in.size()[2], f0_in.size()[3])))  # (1,64,64,64)
        f0_us = resize_feature_map(self.f0_up_sample(f1_up), f0_in.size()[2], f0_in.size()[3])
        f0_in_ffm = self.ffm0(f0_in, f0_us)
        f0_out = self.mscb0(self.combine(weight[0] * f0_in_ffm + weight[1] * f0_us))  # (1,64,64,64)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        f1_w2 = self.f1_w2_relu(self.f1_w2)
        weight = f1_w2 / (torch.sum(f1_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        # f1_out = self.mscb1(
        #     self.combine(weight[0] * f1_in + weight[1] * f1_up +
        #                  weight[2] * resize_feature_map(self.f1_down_sample(f0_out), f1_in.size()[2], f1_in.size()[3])))  # (1,64,32,32)
        f1_ds = resize_feature_map(self.f1_down_sample(f0_out), f1_in.size()[2], f1_in.size()[3])
        f1_up_ffm = self.ffm1(f1_up, f1_ds)
        f1_out = self.mscb1(self.combine(weight[0] * f1_in + weight[1] * f1_up_ffm + weight[2] * f1_ds))  # (1,64,32,32)

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        f2_w2 = self.f2_w2_relu(self.f2_w2)
        weight = f2_w2 / (torch.sum(f2_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        # f2_out = self.mscb2(
        #     self.combine(weight[0] * f2_in + weight[1] * f2_up +
        #                  weight[2] * resize_feature_map(self.f2_down_sample(f1_out), f2_in.size()[2], f2_in.size()[3])))  # (1,64,16,16)
        f2_ds = resize_feature_map(self.f2_down_sample(f1_out), f2_in.size()[2], f2_in.size()[3])
        f2_up_ffm = self.ffm2(f2_up, f2_ds)
        f2_out = self.mscb2(self.combine(weight[0] * f2_in + weight[1] * f2_up_ffm + weight[2] * f2_ds))  # (1,64,16,16)

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        f3_w2 = self.f3_w2_relu(self.f3_w2)
        weight = f3_w2 / (torch.sum(f3_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        # f3_out = self.mscb3(
        #     self.combine(weight[0] * f3_in + weight[1] * f3_up +
        #                  weight[2] * resize_feature_map(self.f3_down_sample(f2_out), f3_in.size()[2], f3_in.size()[3])))  # (1,64,8,8)
        f2_ds = resize_feature_map(self.f3_down_sample(f2_out), f3_in.size()[2], f3_in.size()[3])
        f3_up_ffm = self.ffm3(f3_up, f2_ds)
        f3_out = self.mscb3(self.combine(weight[0] * f3_in + weight[1] * f3_up_ffm + weight[2] * f2_ds))  # (1,64,8,8)

        # Weights for P7_0 and P6_2 to P7_2
        f4_w2 = self.f4_w2_relu(self.f4_w2)
        weight = f4_w2 / (torch.sum(f4_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        # f4_out = self.mscb4(
        #     self.combine(weight[0] * f4_in +
        #                  weight[1] * resize_feature_map(self.f4_down_sample(f3_out), f4_in.size()[2], f4_in.size()[3])))  # (1,64,4,4)
        f4_ds = resize_feature_map(self.f4_down_sample(f3_out), f4_in.size()[2], f4_in.size()[3])
        f4_in_ffm = self.ffm4(f4_in, f4_ds)
        f4_out = self.mscb4(self.combine(weight[0] * f4_in_ffm + weight[1] * f4_ds))  # (1,64,4,4)

        return f0_out+f0_in, f1_out+f1_in, f2_out+f2_in, f3_out+f3_in, f4_out+f4_in


class BiFPN(nn.Module):
    """
        num_stages: int, bifpn number of repeats
        in_channels: List[int], input dim for P3, P4, P5
        out_channels: int, output dim for P2 - P7
        start_level: int, Index of input features in backbone
        epsilon: float, hyperparameter in fusion features
        apply_bn_for_resampling: bool, whether use bn after resampling
        conv_bn_act_pattern: bool, whether use conv_bn_act_pattern
        use_swish: whether use MemoryEfficientSwish
        norm_cfg: (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer.
        init_cfg: MultiConfig: init method
    """

    def __init__(self, num_stages=1, in_channels=[256, 256, 256, 256, 256], out_channels=256,
                 epsilon=1e-4, conv_bn_act_pattern=False):
        super(BiFPN, self).__init__()
        self.conv_gn = nn.Sequential(
            nn.Conv2d(in_channels[-1], out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels)
        )
        self.bifpn = nn.Sequential(*[
            BiFPNStage(in_channels=in_channels, out_channels=out_channels, conv_bn_act_pattern=conv_bn_act_pattern,
                       epsilon=epsilon)
            for _ in range(num_stages)
        ])

    def forward(self, x):
        x0, x1, x2, x3 = x
        x4 = self.conv_gn(x3)
        x_scale5 = [x0, x1, x2, x3, x4]
        x_out = self.bifpn(x_scale5)
        x0_out, x1_out, x2_out, x3_out, x4_out = x_out
        x_add = [x0+x0_out, x1+x1_out, x2+x2_out, x3+x3_out]
        return x_add


if __name__ == '__main__':
    input_dim = 256
    output_dim = 256
    model = BiFPN()
    dummy_input1 = torch.randn(2, input_dim, 64, 64)
    dummy_input2 = torch.randn(2, input_dim, 64, 64)
    dummy_input3 = torch.randn(2, input_dim, 64, 64)
    dummy_input4 = torch.randn(2, input_dim, 64, 64)
    dummy_input5 = torch.randn(2, input_dim, 64, 64)

    outputs = model([dummy_input1, dummy_input2, dummy_input3, dummy_input4, dummy_input5])

    for output in outputs:
        print("features shape:", outputs.size())
