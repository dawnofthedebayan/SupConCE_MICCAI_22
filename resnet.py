import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,
                 include_fc=False):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.include_fc = include_fc

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if self.include_fc: self.fc = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion, n_classes), nn.ReLU(), nn.Linear(n_classes, n_classes))
        #self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.include_fc: x = self.fc(x)

        return x



class ResNetDecoder_Inpainting(nn.Module):
    def __init__(self, autoencoding=False, num_classes= 1,base_n_filter = 8):

        super(ResNetDecoder_Inpainting, self).__init__()

        
        self.base_n_filter = base_n_filter
        self.autoencoding = autoencoding
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.upscale_4x = nn.Upsample(scale_factor=4, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.BatchNorm3d(self.base_n_filter*8)
        if autoencoding: 
            # Level 1 localization pathway
            self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(512, 512)
            self.conv3d_l1 = nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(256, 256)

            # Level 2 localization pathway
            self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(256, 256)
            self.conv3d_l2 = nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(128, 128)

            # Level 3 localization pathway
            self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(128, 128)
            self.conv3d_l3 = nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(64, 64)

            # Level 4 localization pathway
            self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(64, 64)
            self.conv3d_l4 = nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l4 = self.norm_lrelu_upscale_conv_norm_lrelu(32, 32)

            # Level 5 localization pathway
            self.conv_norm_lrelu_l5 = self.conv_norm_lrelu(32, 32)
            self.conv3d_l5 = nn.Conv3d(32, 16, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l5 = self.norm_lrelu_upscale_conv_norm_lrelu(16, 16)

            # Level 6 localization pathway
            self.conv_norm_lrelu_l6 = self.conv_norm_lrelu(16, 16)
            self.conv3d_l6 = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)

            self.ds2_1x1_conv3d = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.ds3_1x1_conv3d = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0, bias=False)

        else:

            # Level 1 localization pathway
            self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(1024, 1024)
            self.conv3d_l1 = nn.Conv3d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(512, 512)

            # Level 2 localization pathway
            self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(1024, 1024)
            self.conv3d_l2 = nn.Conv3d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(512, 512)

            # Level 3 localization pathway
            self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(1024, 1024)
            self.conv3d_l3 = nn.Conv3d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(512, 256)

            # Level 4 localization pathway
            self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(512, 512)
            self.conv3d_l4 = nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l4 = self.norm_lrelu_upscale_conv_norm_lrelu(256, 128)

            # Level 5 localization pathway
            self.conv_norm_lrelu_l5 = self.conv_norm_lrelu(256, 256)
            self.conv3d_l5 = nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm_lrelu_upscale_conv_norm_lrelu_l5 = self.norm_lrelu_upscale_conv_norm_lrelu(128, 64)

            # Level 4 localization pathway
            self.conv_norm_lrelu_l6 = self.conv_norm_lrelu(128, 128)
            self.conv3d_l6 = nn.Conv3d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)

            self.ds2_1x1_conv3d = nn.Conv3d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.ds3_1x1_conv3d = nn.Conv3d(256, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.tanh = nn.Tanh()


    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.BatchNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.BatchNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x,skips):
        
        out = x
        # Level 1 localization pathway
        if skips is not None: out = torch.cat([x, skips[0]], dim=1)

        out = torch.unsqueeze(out,axis=2)
        out = torch.unsqueeze(out,axis=2)
        out = torch.unsqueeze(out,axis=2)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        if skips is not None: out = torch.cat([out, skips[1]], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        if skips is not None: out = torch.cat([out, skips[2]], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        if skips is not None: out = torch.cat([out, skips[3]], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        ds2 = out
        out = self.conv3d_l4(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l4(out)

        # Level 3 localization pathway
        if skips is not None: out = torch.cat([out, skips[4]], dim=1)
        out = self.conv_norm_lrelu_l5(out)
        ds3 = out
        out = self.conv3d_l5(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l5(out)

        # Level 4 localization pathway
        if skips is not None: out = torch.cat([out, skips[5]], dim=1)
        out = self.conv_norm_lrelu_l6(out)
        out_pred = self.conv3d_l6(out)


        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv

        
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)
        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale

        out = self.tanh(out)
        #seg_layer = out
        #out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
        #out = out.view(-1, self.n_classes)
        #out = self.softmax(out)
        return out

class ResNetDecoder(nn.Module): 

    def __init__(self,in_channels=512,num_classes=2):

        super(ResNetDecoder, self).__init__()


        

        self.model_reconstr = ResNetDecoder_Inpainting(autoencoding=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x,skips):
        

        x_recon = self.model_reconstr(x,skips=None)
        return x_recon


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model