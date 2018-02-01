# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorboardX as tb

from model.config import cfg
from model.nms_wrapper import nms
from model.test import im_detect
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
import utils.timer
from utils.bbox import bbox_overlaps

try:
  import cPickle as pickle
except ImportError:
  import pickle

import torch
import torch.optim as optim

import numpy as np
import os
import sys
import glob
import time
import cv2


def scale_lr(optimizer, scale):
  """Scale the learning rate of the optimizer"""
  for param_group in optimizer.param_groups:
    param_group['lr'] *= scale

class SolverWrapper(object):
  """
    A wrapper class for the training process
  """

  def __init__(self, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
    self.net = network
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.tbdir = tbdir
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pth'
    filename = os.path.join(self.output_dir, filename)
    torch.save(self.net.state_dict(), filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = self.data_layer._cur
    # current shuffled indexes of the database
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indexes of the validation database
    perm_val = self.data_layer_val._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename

  def from_snapshot(self, sfile, nfile):
    print('Restoring model snapshots from {:s}'.format(sfile))
    self.net.load_state_dict(torch.load(str(sfile)))
    print('Restored.')
    # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
    # tried my best to find the random states so that it can be recovered exactly
    # However the Tensorflow state is currently not available
    with open(nfile, 'rb') as fid:
      st0 = pickle.load(fid, encoding='latin1')
      cur = pickle.load(fid, encoding='latin1')
      perm = pickle.load(fid, encoding='latin1')
      cur_val = pickle.load(fid, encoding='latin1')
      perm_val = pickle.load(fid, encoding='latin1')
      last_snapshot_iter = pickle.load(fid, encoding='latin1')

      np.random.set_state(st0)
      self.data_layer._cur = cur
      self.data_layer._perm = perm
      self.data_layer_val._cur = cur_val
      self.data_layer_val._perm = perm_val

    return last_snapshot_iter

  def construct_graph(self):
    # Set the random seed
    torch.manual_seed(cfg.RNG_SEED)
    # Build the main computation graph
    self.net.create_architecture(self.imdb.num_classes, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
    # Define the loss
    # loss = layers['total_loss']
    # Set learning rate and momentum
    lr = cfg.TRAIN.LEARNING_RATE
    params = []
    for key, value in dict(self.net.named_parameters()).items():
      if value.requires_grad:
        if 'bias' in key:
          params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    self.optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    # Write the train and validation information to tensorboard
    self.writer = tb.writer.FileWriter(self.tbdir)
    self.valwriter = tb.writer.FileWriter(self.tbvaldir)

    return lr, self.optimizer

  def find_previous(self):
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pth')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in pytorch
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.pth'.format(stepsize+1)))
    sfiles = [ss for ss in sfiles if ss not in redfiles]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    redfiles = [redfile.replace('.pth', '.pkl') for redfile in redfiles]
    nfiles = [nn for nn in nfiles if nn not in redfiles]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    return lsf, nfiles, sfiles

  def initialize(self):
    # Initial file lists are empty
    np_paths = []
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    self.net.load_pretrained_cnn(torch.load(self.pretrained_model))
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    last_snapshot_iter = 0
    lr = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)

    return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def restore(self, sfile, nfile):
    # Get the most recent snapshot and restore
    np_paths = [nfile]
    ss_paths = [sfile]
    # Restore model from snapshots
    last_snapshot_iter = self.from_snapshot(sfile, nfile)
    # Set the learning rate
    lr_scale = 1
    stepsizes = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        lr_scale *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)
    scale_lr(self.optimizer, lr_scale)
    lr = cfg.TRAIN.LEARNING_RATE * lr_scale
    return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def remove_snapshot(self, np_paths, ss_paths):
    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      nfile = np_paths[0]
      os.remove(str(nfile))
      np_paths.remove(nfile)

    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      # To make the code compatible to earlier versions of Tensorflow,
      # where the naming tradition for checkpoints are different
      os.remove(str(sfile))
      ss_paths.remove(sfile)

  def test_during_train(self, imdb, iter):
      cfg.hard_negative = {}
      net = self.net
      np.random.seed(cfg.RNG_SEED)
      num_images = len(imdb.image_index)
      save_idx = np.random.choice(num_images, 128)
      save_dir = 'output/test_during_train/iter_{}'.format(iter)
      for i in range(num_images):
          img_name = imdb.image_index[i]
          print('test img {}'.format(img_name))
          gt = imdb.roidb[i]['boxes'].astype(np.float)  # ndarray, [?, 4]
          im = cv2.imread(imdb.image_path_at(i))
          scores, boxes = im_detect(net, im)  # boxes: ndarray, [300, 84]. scores: ndarray, [300, 21]
          max_score_cls = np.argmax(scores[:, 1:], axis=1) + 1  # ndarray, [300], class index of every roi.
          max_scores = scores[range(0,scores.shape[0]), max_score_cls]
          max_score_boxes = np.zeros([0, 4])  # ndarray, [300, 4], 300 predicted boxes.
          for j in range(scores.shape[0]):
              max_score_boxes = np.append(max_score_boxes, boxes[j, 4*max_score_cls[j]:4*(max_score_cls[j] + 1)].reshape(-1, 4), axis=0)
          overlaps = bbox_overlaps(max_score_boxes, gt)
          no_overlap_box_inds = np.nonzero(np.max(overlaps, axis=1) == 0)[0]  # ndarray, [?]
          if no_overlap_box_inds.shape[0] > 0:
              worst_idx = no_overlap_box_inds[np.argmax(max_scores[no_overlap_box_inds])]
              worst_box = max_score_boxes[worst_idx].astype(int)
              if worst_box[2] - worst_box[0] <= 0 or worst_box[3]- worst_box[1] <= 0:
                  continue
              cfg.hard_negative[img_name] = worst_box  # ndarray, [4]
              ## save img with boxes
              if i in save_idx:
                  ## draw gt boxes
                  for k in range(gt.shape[0]):
                      pt1 = (int(gt[k][0]), int(gt[k][1]))
                      pt2 = (int(gt[k][2]), int(gt[k][3]))
                      cv2.rectangle(im, pt1, pt2, color=(0, 255, 0))
                  ##
                  ## draw worst box
                  pt1 = (worst_box[0], worst_box[1])
                  pt2 = (worst_box[2], worst_box[3])
                  cv2.rectangle(im, pt1, pt2, color=(255, 0, 0))
                  ##
                  ## save
                  file_name = os.path.join(save_dir, '{}.jpg'.format(img_name))
                  if not os.path.exists(save_dir):
                      os.makedirs(save_dir)
                  cv2.imwrite(file_name, im)
                  ##
              ##


  def train_model(self, max_iters):
    # define vars
    cfg.hard_negative = {}
    #

    # get test imdb
    imdb_test = get_imdb('voc_2007_trainval')
    imdb_test.competition_mode(False)
    #

    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    # Construct the computation graph
    lr, train_op = self.construct_graph()

    # Find previous snapshots if there is any to restore from
    lsf, nfiles, sfiles = self.find_previous()

    # Initialize the variables or restore them from the last snapshot
    if lsf == 0:
      lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize()
    else:
      lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(str(sfiles[-1]), 
                                                                             str(nfiles[-1]))
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    # Make sure the lists are not empty
    stepsizes.append(max_iters)
    stepsizes.reverse()
    next_stepsize = stepsizes.pop()

    self.net.train()
    self.net.cuda()

    while iter < max_iters + 1:
      cfg.current_iter = iter
      # Learning rate
      if iter == next_stepsize + 1:
        # Add snapshot here before reducing the learning rate
        self.snapshot(iter)
        lr *= cfg.TRAIN.GAMMA
        scale_lr(self.optimizer, cfg.TRAIN.GAMMA)
        next_stepsize = stepsizes.pop()

      utils.timer.timer.tic()
      # Get training data, one batch at a time
      blobs = self.data_layer.forward()

      now = time.time()
      if iter % 3500 == 0:
        ## use traing_dataset to test net
        print('test during training...')
        self.net.eval()
        self.test_during_train(imdb_test, iter)
        ##
      else:
        # Compute the graph without summary
        self.net.train()
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
          self.net.train_step(blobs, self.optimizer)
      utils.timer.timer.toc()

      # Display training information
      if iter % (cfg.TRAIN.DISPLAY) == 0:
        print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
              '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
              (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr))
        print('speed: {:.3f}s / iter'.format(utils.timer.timer.average_time()))

        # for k in utils.timer.timer._average_time.keys():
        #   print(k, utils.timer.timer.average_time(k))

      # Snapshotting
      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        ss_path, np_path = self.snapshot(iter)
        np_paths.append(np_path)
        ss_paths.append(ss_path)

        # Remove the old snapshots if there are too many
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          self.remove_snapshot(np_paths, ss_paths)

      iter += 1

    if last_snapshot_iter != iter - 1:
      self.snapshot(iter - 1)

    self.writer.close()
    self.valwriter.close()


def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
  """Train a Faster R-CNN network."""
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  sw = SolverWrapper(network, imdb, roidb, valroidb, output_dir, tb_dir,
                     pretrained_model=pretrained_model)

  print('Solving...')
  sw.train_model(max_iters)
  print('done solving')
