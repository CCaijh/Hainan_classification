import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """max+avg → 7×7 Conv → Sigmoid"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]      # [B,1,H,W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)        # [B,1,H,W]
        attn     = torch.sigmoid(self.bn(self.conv(
                        torch.cat([max_pool, avg_pool], dim=1))))
        return x * attn
# -------------------------------------------------


# ----------------- BasicBlock / Bottleneck -----------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Note: In the original paper, on the main branch of the dotted residual structure, the step distance of the first 1x1 convolutional layer is 2, and that of the second 3x3 convolutional layer is 1.

    However, in the official implementation process of pytorch, the step distance of the first 1x1 convolutional layer is 1, and that of the second 3x3 convolutional layer is 2.

    The advantage of doing this is that it can increase the accuracy rate by approximately 0.5% on the top1.

    Refer to Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

# --------------------------------------------------------------------------


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super().__init__()
        self.include_top   = include_top
        self.in_channel0   = 32       # conv0 output
        self.in_channel    = 64       # conv1 output
        self.groups        = groups
        self.width_per_group = width_per_group

        # ---------- conv0（把 448→224） ----------
        self.conv0 = nn.Conv2d(3, self.in_channel0, 7, 2, 3, bias=False)
        self.bn0   = nn.BatchNorm2d(self.in_channel0)
        self.relu  = nn.ReLU(inplace=True)

        # ---------- conv1 ----------
        self.conv1 = nn.Conv2d(self.in_channel0, self.in_channel, 7, 2, 3, bias=False)
        self.bn1   = nn.BatchNorm2d(self.in_channel)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # ---------- 四个 stage ----------
        self.layer1 = self._make_layer(block,  64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)


        self.sa1 = nn.Identity()
        self.sa2 = nn.Identity()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()

        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # ----------  stage ----------
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = [block(self.in_channel, channel, stride=stride,
                        downsample=downsample,
                        groups=self.groups, width_per_group=self.width_per_group)]
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,
                                groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    # ---------- Forward ----------
    def forward(self, x):
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.sa1(self.layer1(x))
        x = self.sa2(self.layer2(x))
        x = self.sa3(self.layer3(x))
        x = self.sa4(self.layer4(x))

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x



def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top)

