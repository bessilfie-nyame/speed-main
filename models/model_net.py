# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.utils.model_zoo as model_zoo
from torch import nn

from models.non_local import NONLocalBlock3D
import json
import os

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

with open(os.path.join(os.getcwd(), 'config.json'), "r") as config_file:
    config = json.load(config_file)

model_path = os.path.join(os.getcwd(), config['model_path'])

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)


class I3DBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, non_local=0, use_3d_conv=0):
        super(I3DBottleneck, self).__init__()

        if use_3d_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                                   padding=(1, 0, 0), bias=False)
        else:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm3d(planes, momentum=0.01)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                               padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=0.01)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.non_local = non_local
        if self.non_local == 1:
            self.NL = NONLocalBlock3D(in_channels=planes * 4, sub_sample=False)
            #  self.NL = nn.Sequential(NONLocalBlock3D(in_channels=planes*4, sub_sample=False),)

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

        if self.non_local:
            out = self.NL(out)

        return out


class I3DResNet(nn.Module):
    def __init__(self, block, layers, num_classes, non_local_set, use_3d_conv_set):
        self.inplanes = 64
        super(I3DResNet, self).__init__()

        self.conv_1a = nn.Conv3d(5, 16, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                               bias=False)
        self.bn_1a = nn.BatchNorm3d(16, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.conv_1b = nn.Conv3d(5, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn_1b = nn.BatchNorm3d(16, momentum=0.01)

        self.conv_1c = nn.Conv3d(5, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.bn_1c = nn.BatchNorm3d(16, momentum=0.01)
       

        self.conv_2 = nn.Conv3d(48, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2),
                                       bias=False)
        self.bn_2 = nn.BatchNorm3d(64, momentum=0.01)
        self.maxpool = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=0)

        self.layer1 = self._make_layer(block, 64, layers[0],
                                       non_local=non_local_set[0],
                                       use_3d_conv=use_3d_conv_set[0])
        
        #  non-local add pooling after res2
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       non_local=non_local_set[1],
                                       use_3d_conv=use_3d_conv_set[1])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       non_local=non_local_set[2],
                                       use_3d_conv=use_3d_conv_set[2])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       non_local=non_local_set[3],
                                       use_3d_conv=use_3d_conv_set[3])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # last_duration = math.ceil(args.train_seq_len / 2)
        # last_h = math.ceil(args.height / 32)
        # last_w = math.ceil(args.width / 32)

        # self.avgpool = nn.AvgPool3d(kernel_size=(last_duration, last_h, last_w), stride=1)

        # self.dropout = nn.Dropout(args.dropout)
        # self.dropout = nn.Dropout(p=0.2)


        # self.fc = nn.Conv3d(512*block.expansion, num_classes, kernel_size=1, stride=1)
        self.conv1x1_1 = nn.Sequential(
            nn.Conv3d(2048, num_classes, (1, 1, 1))
        )

    def _make_layer(self, block, planes, blocks, stride=1, non_local=[], use_3d_conv=[]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion, momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            non_local=non_local[0], use_3d_conv=use_3d_conv[0]))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                non_local=non_local[i],
                                use_3d_conv=use_3d_conv[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv_1(x)
        # x = self.bn_1(x)
        # x = self.relu(x)
        a = self.conv_1a(x)
        a = self.bn_1a(a)
        a = self.relu(a)

        b = self.conv_1b(x)
        b = self.bn_1b(b)
        b = self.relu(b)

        c = self.conv_1c(x)
        c = self.bn_1c(c)
        c = self.relu(c)

        x = torch.cat((a, b, c), 1)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
       
        # non-local add pooling after res2

        x = self.layer2(x)
        
        x = self.maxpool2(x)
        x = self.layer3(x)
      

        x = self.layer4(x)

        x = self.avgpool(x)  # (bs, 2048, 1, 1, 1)
        
        # x = self.dropout(x)      
        x = self.conv1x1_1(x)
        x = x.squeeze(2).squeeze(2).squeeze(2)
        return x

def i3dinit(model, strct):
    if strct == 'reid':
        pre_dict = torch.load('/DATA/pytorch-ckpt/pretrain_all_data/model_best.pth.tar')['state_dict']
        new_pre_dict = dict()
        for k, v in pre_dict.items():
            if 'base' in k:
                new_k = '.'.join(k.split('.')[1:])
                new_pre_dict[new_k] = v
            else:
                new_pre_dict[k] = v
        pre_dict = new_pre_dict
    else:
        pre_dict = model_zoo.load_url(model_urls[strct], model_path)

    own_state = model.state_dict()
    own_state.pop('conv_1a.weight')
    own_state.pop('bn_1a.weight')
    own_state.pop('bn_1a.bias')
    own_state.pop('bn_1a.running_mean')
    own_state.pop('bn_1a.running_var')

    own_state.pop('conv_1b.weight')
    own_state.pop('bn_1b.weight')
    own_state.pop('bn_1b.bias')
    own_state.pop('bn_1b.running_mean')
    own_state.pop('bn_1b.running_var')

    own_state.pop('conv_1c.weight')
    own_state.pop('bn_1c.weight')
    own_state.pop('bn_1c.bias')
    own_state.pop('bn_1c.running_mean')
    own_state.pop('bn_1c.running_var')

    own_state.pop('conv_2.weight')
    own_state.pop('bn_2.weight')
    own_state.pop('bn_2.bias')
    own_state.pop('bn_2.running_mean')
    own_state.pop('bn_2.running_var')

    init_params = {}
    uninit_params = {}

    for k, v in pre_dict.items():
        init_params[k] = 1
        if 'fc' in k or 'bottleneck' in k or 'classifier' in k:
            # torch.nn.init.normal_(own_state[k], std=0.01)
            continue

        if isinstance(v, torch.nn.parameter.Parameter):
            v = v.data

        if k == 'conv1.weight' or k == 'bn1.running_mean' or k == 'bn1.running_var' or k == 'bn1.weight' or k == 'bn1.bias':
            continue
        if v.dim() == own_state[k].dim():
            own_state[k].copy_(v)
        else:
            assert v.dim() == 4 and own_state[k].dim() == 5, 'conv layer only'
            r = own_state[k].shape[2]
            view_shape = v.shape[:2] + (1,) + v.shape[2:]
            own_state[k].copy_(v.view(view_shape).repeat(1, 1, r, 1, 1) / r)

    for k in own_state.keys():
        if k not in init_params:
            uninit_params[k] = 1

    model.conv1x1_1.apply(weights_init_kaiming)
    # print('inited oprs\n', init_params.keys())
    # print('uninited oprs\n', uninit_params.keys())


def I3DR50(num_classes, init_model='kinetics'):
    """Constructs a ResNet-50 model.
    """

    use_temp_convs_2 = [1, 1, 1]
    non_local_2 = [0, 0, 0]

    use_temp_convs_3 = [1, 0, 1, 0]
    non_local_3 = [0, 1, 0, 1]

    use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
    non_local_4 = [0, 1, 0, 1, 0, 1]

    use_temp_convs_5 = [0, 1, 0]
    non_local_5 = [0, 0, 0]
   
    #non_local_3 = [0, 0, 0, 0]
    #non_local_4 = [0, 0, 0, 0, 0, 0]
    
    non_local_set = [non_local_2, non_local_3, non_local_4, non_local_5]
    use_3d_conv_set = [use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]

    model = I3DResNet(I3DBottleneck, [3, 4, 6, 3], num_classes,
                      non_local_set=non_local_set, use_3d_conv_set=use_3d_conv_set)
    if init_model == 'resnet':
        i3dinit(model, 'resnet50')
        print('initialize i3D with resnet')
    else:
        print('*' * 20)
    return model

