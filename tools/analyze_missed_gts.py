from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import torch
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import os, cv2
from nets.vgg16 import vgg16
import pickle

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def get_IoU(BBGT, bb):
    # compute overlaps
    ## intersection
    BBGT = BBGT.reshape(-1, 4)
    ixmin = np.maximum(BBGT[:, 0], bb[:,0])
    iymin = np.maximum(BBGT[:, 1], bb[:,1])
    ixmax = np.minimum(BBGT[:, 2], bb[:,2])
    iymax = np.minimum(BBGT[:, 3], bb[:,3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    ##
    ## union
    uni = ((bb[:, 2] - bb[:, 0] + 1.) * (bb[:, 3] - bb[:, 1] + 1.) +
           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
    ##
    ## get IoU
    overlaps = inters / uni
    ##
    #

    return overlaps

if __name__ == '__main__':
    # declare vars
    num_IoU_greater_than_half = 0
    num_IoU_less_than_half = 0
    #

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
    cls_name = 'pottedplant'
    cls_id = CLASSES.index(cls_name)
    missed_gt_file = open('backup/missed_gt_{}.pkl'.format(cls_name), 'rb')
    missed_gt = pickle.load(missed_gt_file)
    missed_gt_file.close()
    #

    # test every img
    im_names = missed_gt.keys()
    for image_name in im_names:
        ## load missed_gt in current img
        current_missed_gt = missed_gt[image_name]
        ##
        ## get detection result
        im_file = '/data/zhbli/VOCdevkit/VOC2007/JPEGImages/{}.jpg'.format(image_name)
        im = cv2.imread(im_file)
        assert im is not None, 'no img: {}'.format(im_file)
        scores, rois, boxes = im_detect(net, im)
        det_boxes = boxes[:, 4 * cls_id:4 * (cls_id + 1)]
        det_scores = scores[:, cls_id]
        ##
        ## calculate IoU between 300 rois from RPN and missed_gts
        for i in range(current_missed_gt.shape[0]):
            IoU = get_IoU(current_missed_gt[i, :], rois)
            if max(IoU) > 0.5:
                num_IoU_greater_than_half = num_IoU_greater_than_half + 1
                print('max IoU = {}, score = {}'.format(max(IoU), det_scores[np.argmax(IoU)]))
            else:
                num_IoU_less_than_half = num_IoU_less_than_half + 1
        ##
        ## calculate IoU between final detect result and missed_gts
        # for i in range(current_missed_gt.shape[0]):
        #     IoU = get_IoU(current_missed_gt[i, :], det_boxes)
        #     if max(IoU) > 0.5:
        #         num_IoU_greater_than_half = num_IoU_greater_than_half + 1
        #         print('max IoU = {}, score = {}'.format(max(IoU), det_scores[np.argmax(IoU)]))
        #     else:
        #         num_IoU_less_than_half = num_IoU_less_than_half + 1
        ##
    print('IoU > 0.5: {}, IoU < 0.5: {}'.format(num_IoU_greater_than_half, num_IoU_less_than_half))
    #