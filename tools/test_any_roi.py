# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import _init_paths
from model.test import im_detect
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from model.config import cfg
import torch
from torch.autograd import Variable
from model.nms_wrapper import nms
import matplotlib.pyplot as plt

global img
global point1, point2, roi

def on_mouse(event, x, y, flags, param):
    global img, point1, point2, roi
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP: # if left button is up
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5)
        cv2.imshow('image', img2)
        x1 = min(point1[0],point2[0])
        y1 = min(point1[1],point2[1])
        x2 = max(point1[0], point2[0])
        y2 = max(point1[1], point2[1])
        roi = np.asarray([x1, y1, x2, y2])
        """resize roi"""
        im_shape = img.shape
        im_size_min = min(im_shape[0:2])
        im_size_max = max(im_shape[0:2])
        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        global_roi = np.asarray([0, x1, y1, x2, y2])
        global_roi = global_roi * im_scale
        global_roi = Variable(torch.from_numpy(global_roi).type(torch.cuda.FloatTensor).unsqueeze(0))
        cfg.global_roi = global_roi
        """resize roi"""

def main():
    # define vars
    global img
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor',
               'aeroplane_truncated', 'bicycle_truncated', 'bird_truncated', 'boat_truncated',
               'bottle_truncated', 'bus_truncated', 'car_truncated', 'cat_truncated', 'chair_truncated',
               'cow_truncated', 'diningtable_truncated', 'dog_truncated', 'horse_truncated',
               'motorbike_truncated', 'person_truncated', 'pottedplant_truncated',
               'sheep_truncated', 'sofa_truncated', 'train_truncated', 'tvmonitor_truncated'
               )
    #

    # create net
    net = vgg16()
    net.create_architecture(21, tag='default', anchor_scales=[8, 16, 32])
    saved_model = '/data/zhbli/backup/vgg16_faster_rcnn_iter_70000.pth'
    net.load_state_dict(torch.load(saved_model))
    net.eval()
    net.cuda()
    print('Loaded network {:s}'.format(saved_model))
    #

    """loop"""
    while 1:
        # get RoI need to test
        img = cv2.imread('/data/zhbli/VOCdevkit/VOC2007/JPEGImages/000698.jpg')
        assert img is not None, "fail to load img"
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', img)
        cv2.waitKey(-1)
        print('got rectangle')
        cv2.destroyAllWindows()
        #

        # get det result
        scores, boxes = im_detect(net, img)
        idx = np.argmax(scores, 1).squeeze()
        box = boxes[:, 4 * idx:4 * (idx + 1)][0]
        cls = CLASSES[idx]
        #

        # display det result
        im = img[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        bbox = box
        score = np.max(scores)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(cls, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.show()
        #

if __name__ == '__main__':
    main()
