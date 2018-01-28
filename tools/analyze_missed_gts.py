from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import os, cv2
import pickle
from nets.vgg16 import vgg16
import torch

if __name__ == '__main__':
    # load model
    saved_model = '/home/zhbli/Project/faster-rcnn/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.pth'
    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))
    net = vgg16()
    net.create_architecture(21,
                            tag='default', anchor_scales=[8, 16, 32])
    net.load_state_dict(torch.load(saved_model))
    net.eval()
    net.cuda()
    #

    # load missed_gts
    missed_gt_file = open('backup/missed_gt_pottedplant.pkl', 'rb')
    missed_gt = pickle.load(missed_gt_file)
    missed_gt_file.close()
    #

    # test every img
    im_names = missed_gt.keys()
    for image_name in im_names:
        ## Load the demo image
        print('detect img {}'.format(image_name))
        im_file = '/data/zhbli/VOCdevkit/VOC2007/JPEGImages/{}.jpg'.format(image_name)
        im = cv2.imread(im_file)
        assert im is not None, 'no img: {}'.format(im_file)
        scores, boxes = im_detect(net, im)
        exit()
        ##
    #