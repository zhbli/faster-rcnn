# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
import time
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.bbox import bbox_overlaps


import torch
from torch.autograd import Variable

def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  if cfg.TRAIN.USE_GT:
    zeros = rpn_rois.data.new(gt_boxes.shape[0], 1)
    all_rois = torch.cat(
      (all_rois, torch.cat((zeros, gt_boxes[:, :-1]), 1))
    , 0)
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = torch.cat((all_scores, zeros), 0)

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))

  # Sample rois with classification labels and bounding box regression
  # targets
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes)

  rois = rois.view(-1, 5)
  roi_scores = roi_scores.view(-1)
  labels = labels.view(-1, 1)
  bbox_targets = bbox_targets.view(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
  bbox_outside_weights = (bbox_inside_weights > 0).float()

  return rois, roi_scores, labels, Variable(bbox_targets), Variable(bbox_inside_weights), Variable(bbox_outside_weights)


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """
  # Inputs are tensor

  clss = bbox_target_data[:, 0]
  bbox_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
  bbox_inside_weights = clss.new(bbox_targets.shape).zero_()
  inds = (clss > 0).nonzero().view(-1)
  if inds.numel() > 0:
    clss = clss[inds].contiguous().view(-1,1)
    dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
    dim2_inds = torch.cat([4*clss, 4*clss+1, 4*clss+2, 4*clss+3], 1).long()
    bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
    bbox_inside_weights[dim1_inds, dim2_inds] = bbox_targets.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, 4).expand_as(dim1_inds)

  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""
  # Inputs are tensor

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return torch.cat(
    [labels.unsqueeze(1), targets], 1)

def get_bg_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image):
    overlaps = bbox_overlaps(
        all_rois[:, 1:5].data,
        gt_boxes[:, :4].data)
    max_overlaps, _ = overlaps.max(1)
    bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(
        -1)
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    if bg_inds.numel() > 0:
        bg_inds = bg_inds[torch.from_numpy(
            np.random.choice(np.arange(0, bg_inds.numel()), size=int(bg_rois_per_image), replace=True)).long().cuda()]
        bg_rois = all_rois[bg_inds].contiguous()
    else:
        print('No bg_rois in this img.')
        bg_rois = torch.from_numpy(np.array([0, 10,10,20,20]*bg_rois_per_image))\
            .type(torch.cuda.FloatTensor)
    return bg_rois

def get_IoU(BBGT, bb):
    # compute overlaps
    # intersection
    BBGT = BBGT.reshape(-1, 4)
    ixmin = np.maximum(BBGT[:, 0], bb[:,0])
    iymin = np.maximum(BBGT[:, 1], bb[:,1])
    ixmax = np.minimum(BBGT[:, 2], bb[:,2])
    iymax = np.minimum(BBGT[:, 3], bb[:,3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[:, 2] - bb[:, 0] + 1.) * (bb[:, 3] - bb[:, 1] + 1.) +
           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def filter_fg_rois_by_attention(gt, gt_selected, pre_fg_rois_np, fg_rois_per_image, gt_choice_idx_pre, scale):
    """
    :param gt: ndarray, shape=[?, 4]
    :param gt_selected: ndarray, shape=[256, 4]
    :param pre_fg_rois_np: ndarray, shape=[256, 4]
    :param fg_rois_per_image: int, =64
    :param gt_choice_idx_pre: ndarray, shape=256, the corresponding gt_index of every roi.
    :return fg_rois_np: shape=[64, 4]
    """

    # flip and rescale attention maps
    current_img_attention = cfg.attention[cfg.current_img_name]
    for i in range(len(current_img_attention)):
        ## flip attention maps
        if cfg.current_img_is_flipped:
            current_img_attention[i] = cv2.flip(current_img_attention[i], 1)
        ##
        ## rescale attention maps
        current_img_attention[i] = cv2.resize(current_img_attention[i], (gt[i][2] - gt[i][0], gt[i][3] - gt[i][1]))
        ##
        ## set value<0.5 to 0
        current_img_attention[i][current_img_attention[i] < 0.5] = 0
        ##
    #

    # get intersection region of every roi
    ixmin = np.maximum(gt_selected[:, 0], pre_fg_rois_np[:, 0])
    iymin = np.maximum(gt_selected[:, 1], pre_fg_rois_np[:, 1])
    ixmax = np.minimum(gt_selected[:, 2], pre_fg_rois_np[:, 2])
    iymax = np.minimum(gt_selected[:, 3], pre_fg_rois_np[:, 3])
    #

    # shift the coordinate of intersection region from whole_img_based to gt_based
    ixmin = (ixmin - gt_selected[:, 0]).astype(int)
    iymin = (iymin - gt_selected[:, 1]).astype(int)
    ixmax = (ixmax - gt_selected[:, 0]).astype(int)
    iymax = (iymax - gt_selected[:, 1]).astype(int)
    #

    # for every roi, get attention score.
    attention_score = -np.ones(len(pre_fg_rois_np))
    for i in range(len(pre_fg_rois_np)):
        intersection_attention = current_img_attention[gt_choice_idx_pre[i]][iymin[i]:iymax[i],ixmin[i]:ixmax[i]]
        attention_score[i] = np.sum(intersection_attention)
    #

    # get the gt_inds of high_scored_rois
    idx_of_selected_rois = np.argsort(-attention_score)[0:fg_rois_per_image]
    gt_choice_idx_after = gt_choice_idx_pre[idx_of_selected_rois]  # len: 64
    #

    # get the selected rois
    fg_rois_np_after = pre_fg_rois_np[idx_of_selected_rois,:]
    #

    return gt_choice_idx_after, fg_rois_np_after



def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  """Args:
    all_rois: Variable, cuda, FloatTensor, [?, 5]. First col are all 0's, [0, x1, y1, x2, y2]
    gt_boxes: Variable, cuda, FloatTensor, [?, 5]. [x1, y1, x2, y2, cls]
  """

  time_start = time.time()
  """variable declaration"""
  im_info = cfg.current_im_info
  im_width, im_height = im_info[1], im_info[0]
  pre_fg_roi_num = 256  # First, generate 256 fg_rois, then filter 64 fg_rois by attention_scores.
  """end: variable declaration"""

  """end: generate fg_rois"""
  gt_boxes_np = gt_boxes.data.cpu().numpy()
  gt = gt_boxes_np[:, :-1]
  gt_label = gt_boxes_np[:, -1]
  # r: randomly sampled IoU of every RoI.
  r = np.random.rand(pre_fg_roi_num) * (1 - cfg.TRAIN.FG_THRESH) + cfg.TRAIN.FG_THRESH
  num_gt = gt.shape[0]
  gt_choice_idx_pre = np.random.choice(num_gt, pre_fg_roi_num)
  gt_selected = gt[gt_choice_idx_pre]
  x1, y1, x2, y2 = gt_selected[:, 0], gt_selected[:, 1], gt_selected[:, 2], gt_selected[:, 3]
  area_gt = (y2 - y1) * (x2 - x1)
  bg_rois = get_bg_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image)
  """STEP 1: s1a"""
  s1a = x2 - (x2 - x1) / r
  s1a = np.maximum(0, s1a)
  """STEP 2: s1b"""
  s1b = x2 - r * (x2 - x1)
  # Get s1
  s1 = np.random.rand(pre_fg_roi_num) * (s1b - s1a) + s1a

  """STEP 3: s2a"""
  s2a = r * (x2 - np.minimum(x1, s1)) + np.maximum(x1, s1)
  """STEP 4: s2b"""
  s2b = np.minimum(x1, s1) + (x2 - np.maximum(x1, s1)) / r
  s2b = np.minimum(im_width, s2b)
  # Get s2
  s2 = np.random.rand(pre_fg_roi_num) * (s2b - s2a) + s2a

  intersection_width = np.minimum(x2, s2) - np.maximum(x1, s1)
  """STEP 5: t1a"""
  intersection = (y2 - y1) * intersection_width
  t1a = y2 - (intersection / r - area_gt + intersection) / (s2 - s1)
  t1a = np.maximum(0, t1a)
  """STEP 6: t1b"""
  t1b = y2 - area_gt / (intersection_width / r - s2 + s1 + intersection_width)
  # Get t1
  t1 = np.random.rand(pre_fg_roi_num) * (t1b - t1a) + t1a

  """STEP 7: t2a"""
  t2a = (np.maximum(t1,
                    y1) * intersection_width - r * s2 * t1 + r * s1 * t1 + r * area_gt + r * intersection_width * np.maximum(
    t1, y1)) / (intersection_width - r * s2 + r * s1 + r * intersection_width)
  """STEP 8: t2b"""
  intersection = (y2 - np.maximum(t1, y1)) * intersection_width
  t2b = (intersection / r - area_gt + intersection) / (s2 - s1) + t1
  t2b = np.minimum(im_height, t2b)
  # Get t2
  t2 = np.stack((t2a, t2b))
  t2_choice_idx = np.random.choice(2, pre_fg_roi_num)
  t2 = t2[t2_choice_idx, range(0, pre_fg_roi_num)]

  pre_fg_rois_np = np.stack((s1, t1, s2, t2), 1)
  """end: generate fg_rois"""
  # IoU = get_IoU(gt_selected, fg_rois_np)
  # if min(IoU) < 0.4:
  #   print('IoU too small: {:f}'.format(min(IoU)))

  """select 64 fg_rois by attention_score"""
  gt_choice_idx_after, fg_rois_np = filter_fg_rois_by_attention(gt, gt_selected, pre_fg_rois_np, fg_rois_per_image, gt_choice_idx_pre, im_info[2])
  """end: select 64 fg_rois by attention_score"""

  """generate variables need to return"""
  """get labels"""
  labels = np.zeros(int(rois_per_image))
  gt_selected_label = gt_label[gt_choice_idx_after]
  labels[0:len(gt_selected_label)] = gt_selected_label
  labels = torch.from_numpy(labels).type(torch.cuda.FloatTensor)
  labels = Variable(labels, requires_grad=False)
  """end: get labels"""

  """get return value: rois"""
  fg_rois_np0 = np.c_[[0] * fg_rois_np.shape[0], fg_rois_np]
  fg_rois = torch.from_numpy(fg_rois_np0).type(torch.cuda.FloatTensor)
  fg_rois = Variable(fg_rois, requires_grad=True)
  rois = torch.cat([fg_rois, bg_rois], 0)
  """end: get return value: rois"""

  """get return value: roi_scores"""
  roi_scores = all_scores[0:int(rois_per_image)].contiguous()  # No use
  """end: get return value: roi_scores"""

  gt_assignment_np = np.zeros(int(rois_per_image))
  gt_assignment_np[0:gt_choice_idx_after.shape[0]] = gt_choice_idx_after  # gt_idx of every roi
  gt_assignment = torch.from_numpy(gt_assignment_np).type(torch.cuda.LongTensor)
  bbox_target_data = _compute_targets(
    rois[:, 1:5].data, gt_boxes[gt_assignment][:, :4].data, labels.data)
  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)
  """end: generate variables need to return"""

  time_end = time.time()
  print('_sample_rois() cost {:f} s.'.format(time_end - time_start))

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
