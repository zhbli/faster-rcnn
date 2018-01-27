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
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.bbox import bbox_overlaps
import time


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


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """

  """Args:
      all_rois: Variable, FloatTensor, [roi_num, 5], [0, x1, y1, x2, y2]
      gt_boxes: Variable, [gt_num, 5], [x1, y1, x2, y2, class_id]
      fg_rois_per_image: int, 64
      rois_per_image: float, 256.0
      num_classes: int, 2 (or 3)
  """

  # declare variables
  time_start = time.time()
  img_width = float(cfg.current_im_info[1])
  img_height = float(cfg.current_im_info[0])
  #

  # convert Variable to numpy
  gt_boxes_np0 = gt_boxes.data.cpu().numpy()
  gt_boxes_np = gt_boxes_np0[:,:-1]
  #

  # select 64 gts whose cls_name is pottedplant
  selected_pottedplant_gt_idx = np.random.choice(np.where(gt_boxes_np0[:,-1] == 1)[0], fg_rois_per_image)
  selected_pottedplant_gt = gt_boxes_np[selected_pottedplant_gt_idx]
  #

  #get width and height of every selected gt_box
  width_selected_pottedplant_gt = (selected_pottedplant_gt[:, 2] - selected_pottedplant_gt[:, 0]).reshape(-1, 1)  # x2-x1
  height_selected_pottedplant_gt = (selected_pottedplant_gt[:, 3] - selected_pottedplant_gt[:, 1]).reshape(-1, 1)
  #

  # get the width and height delta. delta is used to generate rois
  delta = np.random.rand(fg_rois_per_image, 4) * 0.2  # [0, 0.2)
  delta = delta * np.concatenate((-width_selected_pottedplant_gt, -height_selected_pottedplant_gt, width_selected_pottedplant_gt, height_selected_pottedplant_gt), axis = 1)
  #

  # generate 64 pottedplant rois
  pottedplant_rois = selected_pottedplant_gt + delta
  #

  # manage the boundary
  pottedplant_rois[:, 0] = np.maximum(0, pottedplant_rois[:, 0])  # x1
  pottedplant_rois[:, 1] = np.maximum(0, pottedplant_rois[:, 1])  # y1
  pottedplant_rois[:, 2] = np.minimum(img_width, pottedplant_rois[:, 2])  # x2
  pottedplant_rois[:, 3] = np.minimum(img_height, pottedplant_rois[:, 3])  # y2
  #

  # get bg_rois
  ## overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    all_rois[:, 1:5].data,
    gt_boxes[:, :4].data)
  max_overlaps, gt_assignment = overlaps.max(1)
  labels = gt_boxes[gt_assignment, [4]]
  ##
  ## Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1)
  bg_rois_per_image = rois_per_image - fg_rois_per_image
  to_replace = bg_inds.numel() < bg_rois_per_image
  bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(bg_rois_per_image), replace=to_replace)).long().cuda()]
  bg_rois = all_rois[bg_inds].contiguous()
  ##
  #

  # Get Variable: rois, labels and roi_scores
  ## get rois
  fg_rois = torch.from_numpy(pottedplant_rois).type(torch.cuda.FloatTensor)
  fg_rois = torch.cat((torch.zeros(len(fg_rois), 1).type(torch.cuda.FloatTensor), fg_rois), 1)  # add 0s at first column.
  fg_rois = Variable(fg_rois, requires_grad=True)
  rois = torch.cat((fg_rois, bg_rois), 0)
  ##
  ## get labels
  labels_np = np.zeros(int(rois_per_image))
  labels_np[0:fg_rois_per_image] = 1
  labels = Variable(torch.from_numpy(labels_np).type(torch.cuda.FloatTensor))
  ##
  ## get roi_scores
  roi_scores = Variable(torch.zeros(256, 1).type(torch.cuda.FloatTensor), requires_grad=True)
  ##
  #

  # do something
  temp_gt = np.tile(selected_pottedplant_gt, (4, 1))
  temp_gt = torch.from_numpy(temp_gt).type(torch.cuda.FloatTensor)
  bbox_target_data = _compute_targets(
    rois[:, 1:5].data, temp_gt, labels.data)
  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)
  time_end = time.time()
  # print('_sample_rois() cost {:f} s.'.format(time_end - time_start))
  #

  """return:
      labels: Variable, torch.cuda.FloatTensor of size 256, require_grad=False
      rois: Variable, [256, 5], first column are all zeros, require_grad=True
      roi_scores: no use. Variable, [256,1]
      bbox_targets: no use. FloatTensor, [256, 84]
      bbox_inside_weights: no use. FloatTensor, [256, 84]
  """
  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
