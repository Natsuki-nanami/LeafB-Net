import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt
from timm.layers import SqueezeExcite
from torch.nn import init

class DualECAAttention(nn.Module):  #
    def __init__(self, kernel_size=3):  #
        super().__init__()  #
        self.gap = nn.AdaptiveAvgPool2d(1)  #
        self.gmp = nn.AdaptiveMaxPool2d(1)  #
        self.conv_gap = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)  #
        self.conv_gmp = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)  #
        self.sigmoid = nn.Sigmoid()  #
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, C, H, W) #
        y_gap = self.gap(x)  #
        y_gap = y_gap.squeeze(-1).permute(0, 2, 1)  # (B, 1, C) #
        y_gap = self.conv_gap(y_gap)  #

        y_gmp = self.gmp(x)  #
        y_gmp = y_gmp.squeeze(-1).permute(0, 2, 1)  # (B, 1, C) #
        y_gmp = self.conv_gmp(y_gmp)  #

        y_fused = y_gap + y_gmp  #
        y_fused = self.sigmoid(y_fused)  # (B, 1, C) #
        y_fused = y_fused.permute(0, 2, 1).unsqueeze(-1)  # (B, C, 1, 1) #
        return x * y_fused.expand_as(x)  #


class BasicConv(nn.Module):  #
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):  #
        super(BasicConv, self).__init__()  #
        self.out_channels = out_planes  #
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  #
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None  #
        self.relu = nn.ReLU() if relu else None  #

    def forward(self, x):  #
        x = self.conv(x)  #
        if self.bn is not None:  #
            x = self.bn(x)  #
        if self.relu is not None:  #
            x = self.relu(x)  #
        return x  #


class ZPool(nn.Module):  #
    def forward(self, x):  #
        a = torch.max(x, 1)[0].unsqueeze(1)  #
        b = torch.mean(x, 1).unsqueeze(1)  #
        c = torch.cat((a, b), dim=1)  #
        return c  #


class AttentionGate(nn.Module):  #
    def __init__(self):  #
        super(AttentionGate, self).__init__()  #
        kernel_size = 7  #
        self.compress = ZPool()  #
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)  #

    def forward(self, x):  #
        x_compress = self.compress(x)  #
        x_out = self.conv(x_compress)  #
        scale = torch.sigmoid_(x_out)  #
        return x * scale  #


class TripletAttention(nn.Module):  #
    def __init__(self, no_spatial=False):  #
        super(TripletAttention, self).__init__()  #
        self.cw = AttentionGate()  #
        self.hc = AttentionGate()  #
        self.no_spatial = no_spatial  #
        if not no_spatial:  #
            self.hw = AttentionGate()  #

    def forward(self, x):  #
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()  #
        x_out1 = self.cw(x_perm1)  #
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()  #

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()  #
        x_out2 = self.hc(x_perm2)  #
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()  #

        if not self.no_spatial:  #
            x_out_hw = self.hw(x)  # Note: Renamed to x_out_hw to avoid confusion with x_out before 1/3 calc #
            x_out = 1 / 3 * (x_out_hw + x_out11 + x_out21)  #
        else:  #
            x_out = 1 / 2 * (x_out11 + x_out21)  #
        return x_out  #


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):  #
    w = pywt.Wavelet(wave)  #
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)  #
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)  #
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  #
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  #
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  #
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)  #

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)  #

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])  #
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])  #
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),  #
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),  #
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),  #
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)  #

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)  #

    return dec_filters, rec_filters  #


def wavelet_transform(x, filters):  #
    b, c, h, w = x.shape  #
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)  #
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)  #
    x = x.reshape(b, c, 4, h // 2, w // 2)  #
    return x  #


def inverse_wavelet_transform(x, filters):  #
    b, c, _, h_half, w_half = x.shape  #
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)  #
    x = x.reshape(b, c * 4, h_half, w_half)  #
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)  #
    return x  #


class _ScaleModule(nn.Module):  #
    def __init__(self, dims, init_scale=1.0, init_bias=0):  #
        super(_ScaleModule, self).__init__()  #
        self.dims = dims  #
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)  #
        self.bias = None  #

    def forward(self, x):  #
        return torch.mul(self.weight, x)  #


class DWConv2d_BN_ReLU(nn.Sequential):  #
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):  #
        super().__init__()  #
        self.add_module('dwconv3x3',  #
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                  #
                                  groups=in_channels, bias=False))  #
        self.add_module('bn1', nn.BatchNorm2d(in_channels))  #
        self.add_module('relu', nn.ReLU(inplace=True))  #
        self.add_module('pwconv1x1',  #
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))  #
        self.add_module('bn2', nn.BatchNorm2d(out_channels))  #

        nn.init.constant_(self.bn1.weight, bn_weight_init)  #
        nn.init.constant_(self.bn1.bias, 0)  #
        nn.init.constant_(self.bn2.weight, bn_weight_init)  #
        nn.init.constant_(self.bn2.bias, 0)  #


class Conv2d_BN(torch.nn.Sequential):  #
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,  #
                 groups=1, bn_weight_init=1, ):  #
        super().__init__()  #
        self.add_module('c', torch.nn.Conv2d(  #
            a, b, ks, stride, pad, dilation, groups, bias=False))  #
        self.add_module('bn', torch.nn.BatchNorm2d(b))  #
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)  #
        torch.nn.init.constant_(self.bn.bias, 0)  #


def nearest_multiple_of_N(n, N):  #
    if n <= 0: return 0
    if N <= 0: raise ValueError("N must be positive")
    if n % N == 0:  #
        return n  #
    else:  #
        lower_multiple = (n // N) * N  #
        upper_multiple = lower_multiple + N  #
        if lower_multiple == 0 : return N #
        if (n - lower_multiple) < (upper_multiple - n):  #
            return lower_multiple  #
        else:  #
            return upper_multiple  #


class WTModule(nn.Module):  #
    def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1'):  #
        super().__init__()  #
        self.in_channels = in_channels  #
        self.wt_levels = wt_levels  #

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)  #
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)  #
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)  #

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)  #
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)  #

        self.wavelet_convs = nn.ModuleList(  #
            [DWConv2d_BN_ReLU(in_channels * 4, in_channels * 4, kernel_size=kernel_size)  #
             for _ in range(self.wt_levels)]  #
        )  #

        self.wavelet_scale = nn.ModuleList(  #
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]  #
        )  #

    def forward(self, x):  #
        x_ll_in_levels = []  #
        x_h_in_levels = []  #
        shapes_in_levels = []  #

        curr_x_ll = x  #

        for i in range(self.wt_levels):  #
            curr_shape = curr_x_ll.shape  #
            shapes_in_levels.append(curr_shape)  #
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):  #
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)  #
                curr_x_ll = F.pad(curr_x_ll, curr_pads)  #

            curr_x_subbands = self.wt_function(curr_x_ll)  #
            curr_x_ll = curr_x_subbands[:, :, 0, :, :]  #

            shape_x = curr_x_subbands.shape  # B, C, 4, H/2, W/2 #
            curr_x_tag_intermediate = curr_x_subbands.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])  #

            curr_x_tag_intermediate = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag_intermediate))  #

            curr_x_tag_intermediate = curr_x_tag_intermediate.reshape(shape_x)  #

            x_ll_in_levels.append(curr_x_tag_intermediate[:, :, 0, :, :])  #
            x_h_in_levels.append(curr_x_tag_intermediate[:, :, 1:4, :, :])  #

        next_x_ll = 0  #

        for i in range(self.wt_levels - 1, -1, -1):  #
            curr_x_ll_from_list = x_ll_in_levels.pop()  #
            curr_x_h = x_h_in_levels.pop()  #
            curr_shape = shapes_in_levels.pop()  #

            curr_x_ll_from_list = curr_x_ll_from_list + next_x_ll  #

            curr_x_combined = torch.cat([curr_x_ll_from_list.unsqueeze(2), curr_x_h], dim=2)  #
            next_x_ll = self.iwt_function(curr_x_combined)  #

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]  #

        return next_x_ll  #


class LBModule(nn.Module):
    def __init__(self, dim,
                 identity_ratio=0.25, conv_ratio=0.5, wt_enh_ratio=0.25,
                 conv_kernel_size=3, wt_kernel_size=5,
                 se_reduction=0.25, #
                 eca_kernel_size=3, #
                 wt_levels=1, wt_type='db1'):
        super().__init__()
        self.dim = dim
        N = 4

        self.identity_channels = nearest_multiple_of_N(int(identity_ratio * dim), N)
        self.conv_channels = nearest_multiple_of_N(int(conv_ratio * dim), N)
        #
        temp_sum = self.identity_channels + self.conv_channels
        self.wt_enh_channels = nearest_multiple_of_N(dim - temp_sum, N)

        current_total = self.identity_channels + self.conv_channels + self.wt_enh_channels
        if current_total != dim:
            diff = dim - current_total
            #
            if self.conv_channels > 0 : self.conv_channels += diff
            elif self.wt_enh_channels > 0 : self.wt_enh_channels += diff
            elif self.identity_channels > 0 : self.identity_channels += diff
            else: #
                if dim > 0: #
                    self.conv_channels = dim # Or some other strategy

        self.identity_channels = max(0, self.identity_channels)
        self.conv_channels = max(0, self.conv_channels)
        self.wt_enh_channels = max(0, self.wt_enh_channels)
        self.identity_op = nn.Identity()

        if self.conv_channels > 0:
            self.conv_op = DWConv2d_BN_ReLU(self.conv_channels, self.conv_channels, kernel_size=conv_kernel_size)
            self.se_conv = DualECAAttention(kernel_size=eca_kernel_size)
        else:
            self.conv_op = nn.Identity()
            self.se_conv = nn.Identity()

        if self.wt_enh_channels > 0:
            self.wt_enh_op = WTModule(self.wt_enh_channels,
                                      kernel_size=wt_kernel_size,
                                      wt_levels=wt_levels, wt_type=wt_type)
            self.se_wt_enh = DualECAAttention(kernel_size=eca_kernel_size)
        else:
            self.wt_enh_op = nn.Identity()
            self.se_wt_enh = nn.Identity()

        self.proj = nn.Sequential(
            nn.ReLU(inplace=True),
            Conv2d_BN(dim, dim, bn_weight_init=0)
        )

    def forward(self, x):
        current_offset = 0
        x_identity_in = x.narrow(1, current_offset, self.identity_channels) if self.identity_channels > 0 else torch.empty(x.shape[0], 0, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        current_offset += self.identity_channels

        x_conv_in = x.narrow(1, current_offset, self.conv_channels) if self.conv_channels > 0 else torch.empty(x.shape[0], 0, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        current_offset += self.conv_channels

        actual_wt_enh_channels = x.size(1) - (self.identity_channels + self.conv_channels)
        x_wt_enh_in = x.narrow(1, current_offset, actual_wt_enh_channels) if actual_wt_enh_channels > 0 else torch.empty(x.shape[0], 0, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)


        x_identity = self.identity_op(x_identity_in)
        x_conv = self.conv_op(x_conv_in)
        x_wt_enh = self.wt_enh_op(x_wt_enh_in)
        if self.conv_channels > 0: x_conv = self.se_conv(x_conv)
        if actual_wt_enh_channels > 0: x_wt_enh = self.se_wt_enh(x_wt_enh)


        outputs_to_cat = []
        if self.identity_channels > 0: outputs_to_cat.append(x_identity)
        if self.conv_channels > 0: outputs_to_cat.append(x_conv)
        if actual_wt_enh_channels > 0: outputs_to_cat.append(x_wt_enh)


        if not outputs_to_cat:
            if x.shape[1] == 0 : return torch.empty_like(x)
            return self.proj(torch.zeros(x.shape[0], self.dim, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype))


        x_combined = torch.cat(outputs_to_cat, dim=1)
        x_out = self.proj(x_combined)
        return x_out

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, bn_weight_init=0):
        super().__init__()
        out_features = out_features or in_features
        self.pw1 = Conv2d_BN(in_features, hidden_features)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = Conv2d_BN(hidden_features, out_features, bn_weight_init=bn_weight_init)

    def forward(self, x):
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class DW3x3_Unit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.dw_conv(x)))


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        print("RepVGGDW fuse() method needs a .fuse() in Conv2d_BN or revised logic.")
        return self


class Stem(nn.Module):
    def __init__(self, in_channels, stem_inter_channels, out_channels):
        super().__init__()
        self.conv1 = BasicConv(in_channels, stem_inter_channels, kernel_size=3, stride=2, padding=1, relu=True, bn=True)
        self.conv2 = BasicConv(stem_inter_channels, stem_inter_channels, kernel_size=3, stride=2, padding=1, relu=True,
                               bn=True)
        self.conv3 = BasicConv(stem_inter_channels, out_channels, kernel_size=3, stride=2, padding=1, relu=False,
                               bn=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DownsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.dw_conv(x)))
        x = self.relu2(self.bn2(self.pw_conv(x)))
        return x


class StageBlock(nn.Module):
    def __init__(self, dim, core_module_class,
                 ffn_expansion_factor=4,
                 core_module_kwargs=None,
                 se_rd_ratio=0.0625,
                 core_module_eca_kernel_size=3):
        super().__init__()
        self.dim = dim

        self.dw3x3_unit = DW3x3_Unit(dim)
        self.se_layer = SqueezeExcite(dim, rd_ratio=se_rd_ratio)

        ffn_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn1 = FFN(dim, ffn_hidden_dim, out_features=dim)

        _core_module_kwargs = core_module_kwargs.copy() if core_module_kwargs is not None else {}


        if core_module_class == LBModule:
            lb_defaults = {
                'identity_ratio': 0.25, 'conv_ratio': 0.5,
                'conv_kernel_size': 3, 'wt_kernel_size': 3,
                'se_reduction': se_rd_ratio,
                'eca_kernel_size': core_module_eca_kernel_size,
                'wt_levels': 1, 'wt_type': 'db1'
            }
            final_lb_kwargs = {**lb_defaults, **_core_module_kwargs}
            self.core_module = core_module_class(dim=dim, **final_lb_kwargs)

        elif core_module_class == TripletAttention:
            ta_defaults = {'no_spatial': False}
            final_ta_kwargs = {**ta_defaults, **_core_module_kwargs}
            self.core_module = core_module_class(**final_ta_kwargs)
        else:
            raise ValueError(f"Unsupported core_module_class: {core_module_class}")

        self.ffn2 = FFN(dim, ffn_hidden_dim, out_features=dim)

    def forward(self, x):
        identity_pre_dw = x
        x_dw_processed = self.dw3x3_unit(x)
        x_after_dw_res = identity_pre_dw + x_dw_processed
        x_after_se = self.se_layer(x_after_dw_res)
        identity_pre_ffn1 = x_after_se
        x_ffn1_processed = self.ffn1(x_after_se)
        x_after_ffn1_res = identity_pre_ffn1 + x_ffn1_processed
        identity_pre_core = x_after_ffn1_res
        x_core_processed = self.core_module(x_after_ffn1_res)
        x_after_core_res = identity_pre_core + x_core_processed
        identity_pre_ffn2 = x_after_core_res
        x_ffn2_processed = self.ffn2(x_after_core_res)
        x_after_ffn2_res = identity_pre_ffn2 + x_ffn2_processed
        return x_after_ffn2_res


class AblatedStageBlock(nn.Module):
    def __init__(self, dim, core_module_class,
                 ffn_expansion_factor=4,
                 core_module_kwargs=None,
                 core_module_internal_se_rd_ratio=0.0625,
                 core_module_eca_kernel_size=3):
        super().__init__()
        self.dim = dim

        ffn_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn1 = FFN(dim, ffn_hidden_dim, out_features=dim)

        _core_module_kwargs = core_module_kwargs.copy() if core_module_kwargs is not None else {}

        if core_module_class == LBModule:
            lb_defaults = {
                'identity_ratio': 0.25, 'conv_ratio': 0.5,
                'conv_kernel_size': 3, 'wt_kernel_size': 3,
                'se_reduction': core_module_internal_se_rd_ratio,
                'eca_kernel_size': core_module_eca_kernel_size,
                'wt_levels': 1, 'wt_type': 'db1'
            }
            final_lb_kwargs = {**lb_defaults, **_core_module_kwargs}
            self.core_module = core_module_class(dim=dim, **final_lb_kwargs)
        elif core_module_class == TripletAttention:
            ta_defaults = {'no_spatial': False}
            final_ta_kwargs = {**ta_defaults, **_core_module_kwargs}
            self.core_module = core_module_class(**final_ta_kwargs)
        else:
            raise ValueError(f"Unsupported core_module_class: {core_module_class}")

        self.ffn2 = FFN(dim, ffn_hidden_dim, out_features=dim)

    def forward(self, x):
        identity_pre_ffn1 = x
        x_ffn1_processed = self.ffn1(x)
        x_after_ffn1_res = identity_pre_ffn1 + x_ffn1_processed
        identity_pre_core = x_after_ffn1_res
        x_core_processed = self.core_module(x_after_ffn1_res)
        x_after_core_res = identity_pre_core + x_core_processed
        identity_pre_ffn2 = x_after_core_res
        x_ffn2_processed = self.ffn2(x_after_core_res)
        x_after_ffn2_res = identity_pre_ffn2 + x_ffn2_processed
        return x_after_ffn2_res


class LBNet_xs(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=(64, 128, 256, 384),
                 depths=(0, 2, 8, 10),
                 ffn_expansion_factor=4,
                 stage_block_se_rd_ratio=0.0625,
                 lb_module_eca_kernel_size=3,
                 lb_module_kwargs=None,
                 triplet_attention_kwargs=None,
                 use_ablated_block=False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.depths = depths

        CurrentStageBlock = AblatedStageBlock if use_ablated_block else StageBlock

        current_stage_block_extra_kwargs = {}
        if use_ablated_block:
            current_stage_block_extra_kwargs['core_module_internal_se_rd_ratio'] = stage_block_se_rd_ratio
            current_stage_block_extra_kwargs['core_module_eca_kernel_size'] = lb_module_eca_kernel_size
        else: # Original StageBlock
            current_stage_block_extra_kwargs['se_rd_ratio'] = stage_block_se_rd_ratio
            current_stage_block_extra_kwargs['core_module_eca_kernel_size'] = lb_module_eca_kernel_size


        _lb_module_kwargs_base = lb_module_kwargs.copy() if lb_module_kwargs is not None else {}
        _lb_module_kwargs_base.setdefault('eca_kernel_size', lb_module_eca_kernel_size)
        _lb_module_kwargs_base.setdefault('se_reduction', stage_block_se_rd_ratio)


        _triplet_attention_kwargs_base = triplet_attention_kwargs.copy() if triplet_attention_kwargs is not None else {}

        # Stem
        stem_inter_channels = embed_dims[0] // 2 if embed_dims[0] >= 32 else embed_dims[0]
        self.stem_convs = Stem(in_chans, stem_inter_channels, embed_dims[0])

        # Stage 1
        stage1_params = {
            'dim': embed_dims[0],
            'core_module_class': LBModule,
            'ffn_expansion_factor': ffn_expansion_factor,
            'core_module_kwargs': _lb_module_kwargs_base,
            **current_stage_block_extra_kwargs
        }
        self.stage1_block = CurrentStageBlock(**stage1_params)
        self.stage1_extra_blocks = nn.Sequential(*[
            CurrentStageBlock(**stage1_params)
            for _ in range(depths[0])
        ])

        # Stage 2
        self.downsample2 = DownsampleModule(embed_dims[0], embed_dims[1])
        stage2_params = {
            'dim': embed_dims[1],
            'core_module_class': LBModule,
            'ffn_expansion_factor': ffn_expansion_factor,
            'core_module_kwargs': _lb_module_kwargs_base,
            **current_stage_block_extra_kwargs
        }
        self.stage2_blocks = nn.Sequential(*[
            CurrentStageBlock(**stage2_params)
            for _ in range(depths[1])
        ])

        # Stage 3
        self.downsample3 = DownsampleModule(embed_dims[1], embed_dims[2])
        stage3_params = {
            'dim': embed_dims[2],
            'core_module_class': LBModule,
            'ffn_expansion_factor': ffn_expansion_factor,
            'core_module_kwargs': _lb_module_kwargs_base,
            **current_stage_block_extra_kwargs
        }
        self.stage3_blocks = nn.Sequential(*[
            CurrentStageBlock(**stage3_params)
            for _ in range(depths[2])
        ])

        # Stage 4 (TA Block uses TripletAttention as core_module_class)
        self.downsample4 = DownsampleModule(embed_dims[2], embed_dims[3])
        stage4_params = {
            'dim': embed_dims[3],
            'core_module_class': TripletAttention,
            'ffn_expansion_factor': ffn_expansion_factor,
            'core_module_kwargs': _triplet_attention_kwargs_base,
            **current_stage_block_extra_kwargs
        }

        if use_ablated_block and 'core_module_eca_kernel_size' in stage4_params:
             if stage4_params['core_module_class'] == TripletAttention:
                del stage4_params['core_module_eca_kernel_size']
        elif not use_ablated_block and 'core_module_eca_kernel_size' in stage4_params :
             if stage4_params['core_module_class'] == TripletAttention:
                del stage4_params['core_module_eca_kernel_size']


        self.stage4_blocks = nn.Sequential(*[
            CurrentStageBlock(**stage4_params)
            for _ in range(depths[3])
        ])

        # Head
        self.norm = nn.BatchNorm2d(embed_dims[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embed_dims[3], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if hasattr(m, 'groups') and m.groups == m.in_channels and m.groups != 1:
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else:
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem_convs(x)
        x = self.stage1_block(x)
        if self.depths[0] > 0:
            x = self.stage1_extra_blocks(x)
        x = self.downsample2(x)
        x = self.stage2_blocks(x)
        x = self.downsample3(x)
        x = self.stage3_blocks(x)
        x = self.downsample4(x)
        x = self.stage4_blocks(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def count_dual_eca_attention(m, x_tuple, y):
    x_input = x_tuple[0]
    C = x_input.shape[1]
    m.total_ops += 2 * x_input.nelement()
    kernel_size = m.conv_gap.kernel_size[0] # Assuming conv_gmp has the same kernel size
    m.total_ops += 2 * (C * kernel_size)
    m.total_ops += C
    m.total_ops += C
    m.total_ops += x_input.nelement()


if __name__ == '__main__':
    from thop import profile, clever_format

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    thop_input_tensor = torch.randn(1, 3, 224, 224).to(device)

    # --- NEW "SLIGHTLY LIGHTER" CONFIGURATION ---
    config_embed_dims = [40, 80, 160, 240]
    config_depths = [1, 2, 4, 3]
    config_ffn_expansion_factor = 4.0
    num_classes = 1000

    print("--- Initializing LBNet_test3 with SLIGHTLY LIGHTER COMPLEXITY ---")

    lb_module_args_for_run = {
        'identity_ratio': 0.25,
        'conv_ratio': 0.5,
        'conv_kernel_size': 3,
        'wt_kernel_size': 5,
        'wt_levels': 1,
        'wt_type': 'db1',
    }
    triplet_attention_args_for_run = {'no_spatial': False}

    config_stage_block_se_rd_ratio_for_run = 0.0625
    config_lb_module_eca_kernel_size = 3

    model = LBNet_xs(
        img_size=224,
        in_chans=3,
        num_classes=num_classes,
        embed_dims=config_embed_dims,
        depths=config_depths,
        ffn_expansion_factor=config_ffn_expansion_factor,
        stage_block_se_rd_ratio=config_stage_block_se_rd_ratio_for_run,
        lb_module_eca_kernel_size=config_lb_module_eca_kernel_size,
        lb_module_kwargs=lb_module_args_for_run,
        triplet_attention_kwargs=triplet_attention_args_for_run,
        use_ablated_block=True
    )
    model.to(device)
    print(model)

    dummy_input_for_test = torch.randn(1, 3, 224, 224).to(device)

    print("\n--- Performing a forward pass on slightly lighter LBNet_test3 ---")
    try:
        output = model(dummy_input_for_test)
        print(f"Input shape: {dummy_input_for_test.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (dummy_input_for_test.shape[0], num_classes)
        print("Forward pass successful and output shape is correct.")
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Slightly Lighter LBNet_test3 Summary ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    model.eval()
    macs, params = profile(model, inputs=(thop_input_tensor,),
                           custom_ops={DualECAAttention: count_dual_eca_attention, SqueezeExcite: profile})
    macs, params = clever_format([macs, params], "%.3f")

    print('Flops:  ', macs)
    print('Params: ', params)

    throughput_dummy_input = torch.randn(32, 3, 224, 224).to(device)

    for _ in range(10):
        model(throughput_dummy_input)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(throughput_dummy_input)
    torch.cuda.synchronize()
    end = time.time()

    batch_size = 64
    iterations = 100
    total_samples = batch_size * iterations
    total_time = end - start
    throughput = total_samples / total_time
    print(f"吞吐量: {throughput:.2f} samples/second")