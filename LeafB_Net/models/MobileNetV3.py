from torch import nn

from block import _make_divisible, MobileNetV3Bottleneck, h_swish


class MobileNetV3_Small(nn.Module):
    def __init__(self, mode='small', num_classes=1000, width_mult=1.0):
        super().__init__()
        self.mode = mode
        setting = self._get_network_config(mode, width_mult)

        init_conv_out = _make_divisible(16 * width_mult)
        self.conv1 = nn.Conv2d(3, init_conv_out, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_conv_out)
        self.act1 = h_swish()
        layers = []
        for (k, exp, out, use_se, act, s) in setting:
            out_channels = _make_divisible(out * width_mult)
            exp_channels = _make_divisible(exp * width_mult)
            layers.append(MobileNetV3Bottleneck(
                in_channels=init_conv_out,
                out_channels=out_channels,
                kernel_size=k,
                exp_size=exp_channels,
                stride=s,
                use_se=use_se,
                activation=act
            ))
            init_conv_out = out_channels

        self.blocks = nn.Sequential(*layers)
        last_conv_in = _make_divisible(960 * width_mult) if mode == 'large' else _make_divisible(576 * width_mult)
        last_conv_out = _make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = nn.Conv2d(init_conv_out, last_conv_in, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(last_conv_in)
        self.act2 = h_swish()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(last_conv_in, last_conv_out, 1)
        self.act3 = h_swish()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(last_conv_out, num_classes)

        self._initialize_weights()

    def _get_network_config(self, mode, width_mult):
        return [
            [3, 16, 16, True, 'relu', 2],
            [3, 72, 24, False, 'relu', 2],
            [3, 88, 24, False, 'relu', 1],
            [5, 96, 40, True, 'hswish', 2],
            [5, 240, 40, True, 'hswish', 1],
            [5, 240, 40, True, 'hswish', 1],
            [5, 120, 48, True, 'hswish', 1],
            [5, 144, 48, True, 'hswish', 1],
            [5, 288, 96, True, 'hswish', 2],
            [5, 576, 96, True, 'hswish', 1],
            [5, 576, 96, True, 'hswish', 1]
            ]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.act3(self.conv3(x))
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
