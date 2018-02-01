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
import os
import matplotlib.pyplot as plt
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

def visualize_rois(rois, labels, gt_boxes, hard_inds, fg_inds):
    """
    :param rois:        ndarray, size = [256, 4]
    :param labels:      ndarray, size = [256]
    :param gt_boxes:    ndarray, size = [?, 4]
    :param hard_inds:   ndarray or None, size = [?]
    :param fg_inds:     LongTensor
    :return: None
    """

    # define vars
    is_flipped = cfg.current_img_flipped  # bool
    img_name = cfg.current_img_name
    width = cfg.current_img_width_after_scaled
    scale = cfg.current_img_scale
    save_path = 'output/visualize_rois'
    #

    # delete old imgs
    os.system('rm -rf ' + save_path)
    #

    # Handle flip and scale
    if is_flipped:
        oldx1 = rois[:, 0].copy()
        oldx2 = rois[:, 2].copy()
        rois[:, 0] = width - oldx2 - 1  # self._im_info[1] is width of img after scaled
        rois[:, 2] = width - oldx1 - 1
        oldx1 = gt_boxes[:, 0].copy()
        oldx2 = gt_boxes[:, 2].copy()
        gt_boxes[:, 0] = width - oldx2 - 1  # self._im_info[1] is width of img after scaled
        gt_boxes[:, 2] = width - oldx1 - 1
    rois = rois / scale
    gt_boxes = gt_boxes / scale
    #cfg.current_img_scale

    # display rois
    im = cv2.imread(img_name)
    for j in range(gt_boxes.shape[0]):
        gt = gt_boxes[j].astype(np.int)
        pt1 = (gt[0], gt[1])
        pt2 = (gt[2], gt[3])
        cv2.rectangle(im, pt1, pt2, color=(0, 255, 0))
    if cfg.current_img_name[-10:-4] in cfg.hard_negative.keys():
        hard_negative = cfg.hard_negative[cfg.current_img_name[-10:-4]].astype(np.int)  # ndarray, [4]
        pt1 = (hard_negative[0], hard_negative[1])
        pt2 = (hard_negative[2], hard_negative[3])
        cv2.rectangle(im, pt1, pt2, color=(0, 0, 255))
        if hard_inds is None:
            print('Hard negative exists, but no hard roi is selected.')
    else:
        print('no hard negative')
    if len(fg_inds) == 0:
        print('No fg_rois.')
    for i in range(rois.shape[0]):
        im1 = im.copy()
        roi = rois[i].astype(np.int)
        # label = labels[i]
        pt1 = (roi[0], roi[1])
        pt2 = (roi[2], roi[3])
        cv2.rectangle(im1, pt1, pt2, color=(255, 0, 0))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, '{}_{}.jpg'.format(cfg.current_img_name[-10:-4], i)), im1)
    #

def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # get hard negatives
  hard_inds = None
  if cfg.current_img_name[-10:-4] in cfg.hard_negative.keys():
      hard_negative = cfg.hard_negative[cfg.current_img_name[-10:-4]].astype(np.float)  # ndarray, [4]
      ## flip
      if cfg.current_img_flipped:
          oldx1 = hard_negative[0].copy()
          oldx2 = hard_negative[2].copy()
          hard_negative[0] = cfg.current_img_width_origin - oldx2 - 1
          hard_negative[2] = cfg.current_img_width_origin - oldx1 - 1
          assert (hard_negative[2] >= hard_negative[0]).all()
      ##
      ## rescale
      hard_negative = hard_negative * cfg.current_img_scale
      ##
      ## get IoU between all_rois and hard_engative
      overlaps_hard_negative = bbox_overlaps(
          all_rois[:, 1:5].data.cpu().numpy(),
          hard_negative.reshape(-1, 4))
      hard_inds, _ = np.where(overlaps_hard_negative > 0.5)# inds whose IoU is greater than 0.5 with hard_negative
      if hard_inds.shape[0] != 0:
          hard_inds = np.random.choice(hard_inds, 32)
          hard_inds = torch.from_numpy(hard_inds).type(torch.cuda.LongTensor)
      else:
          hard_inds = None
      ##
  #

  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    all_rois[:, 1:5].data,
    gt_boxes[:, :4].data)
  max_overlaps, gt_assignment = overlaps.max(1)
  labels = gt_boxes[gt_assignment, [4]]

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1)

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.numel() > 0 and bg_inds.numel() > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(fg_rois_per_image), replace=False)).long().cuda()]
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.numel() < bg_rois_per_image
    bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(bg_rois_per_image), replace=to_replace)).long().cuda()]
    if hard_inds is not None:
        bg_inds[-len(hard_inds):] = hard_inds
  elif fg_inds.numel() > 0:
    to_replace = fg_inds.numel() < rois_per_image
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = rois_per_image
  elif bg_inds.numel() > 0:
    to_replace = bg_inds.numel() < rois_per_image
    bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  keep_inds = torch.cat([fg_inds, bg_inds], 0)
  # Select sampled values from various arrays:
  labels = labels[keep_inds].contiguous()
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  rois = all_rois[keep_inds].contiguous()
  roi_scores = all_scores[keep_inds].contiguous()

  bbox_target_data = _compute_targets(
    rois[:, 1:5].data, gt_boxes[gt_assignment[keep_inds]][:, :4].data, labels.data)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  if cfg.current_iter % 2000 == 0:
    print('Visualize_rois.')
    visualize_rois(rois[:, 1:].data.cpu().numpy(), labels.data.cpu().numpy(), gt_boxes[:, :4].data.cpu().numpy(), hard_inds, fg_inds)

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
