# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network import Network
from model.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, Iterable
from torch.autograd import Variable
import math
import torchvision.models as models

class vgg16(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._net_conv_channels = 512
    self._fc7_channels = 4096

  def _init_head_tail(self):
    self.vgg = models.vgg16()
    # Remove fc8
    self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.vgg.features[layer].parameters(): p.requires_grad = False

    # not using the last maxpool layer
    self._layers['head'] = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])

  # v1.0
  def _image_to_head(self):
    s1 = nn.Sequential(OrderedDict(list(self._layers['head']._modules.items())[0:16]))
    s2 = nn.Sequential(OrderedDict(list(self._layers['head']._modules.items())[16:23]))
    s3 = nn.Sequential(OrderedDict(list(self._layers['head']._modules.items())[23:30]))
    temp10 = s1(self._image)
    temp11 = self.branch11(temp10)
    net_conv1 = self.branch12(temp11)
    temp20 = s2(temp10)
    net_conv2 = self.branch2(temp20)
    temp30 = s3(temp20)
    net_conv3 = self.branch3(temp30)

    self._act_summaries['conv'] = temp30
    
    return net_conv1, net_conv2, net_conv3, temp30
  # v1.0

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.vgg.classifier(pool5_flat)

    return fc7

  def load_pretrained_cnn(self, state_dict):
    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})