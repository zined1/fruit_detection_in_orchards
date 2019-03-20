# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import pandas as pd
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
import json 
from .voc_eval import voc_eval
from model.config import cfg

class fruits(imdb):
  def __init__(self, image_set, devkit_path=None):
    imdb.__init__(self, 'fruits_dataset_' + image_set)
    self._image_set = image_set
    self._devkit_path = self._get_default_path() if devkit_path is None \
      else devkit_path
    self._data_path = os.path.join(self._devkit_path, 'fruits_dataset')
    self._classes = ('__background__',  # always index 0
                     'mango')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.png'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.selective_search_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL-VOC specific config options according to the paper
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': False,
                   'matlab_eval': False,
                   'rpn_file': None,
                   'min_size': 2}

    assert os.path.exists(self._devkit_path), \
      'Dataset path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'images',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'sets',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where the dataset is expected to be installed.
    """
    return cfg.DATA_DIR

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def selective_search_roidb(self):
    """
    Return the database of selective search regions of interest.
    Ground-truth ROIs are also included.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path,
                              self.name + '_selective_search_roidb.pkl')

    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} ss roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    if self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      ss_roidb = self._load_selective_search_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
    else:
      roidb = self._load_selective_search_roidb(None)
    with open(cache_file, 'wb') as fid:
      pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote ss roidb to {}'.format(cache_file))

    return roidb

  def rpn_roidb(self):
    if self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_selective_search_roidb(self, gt_roidb):
    filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                            'selective_search_data',
                                            self.name + '.mat'))
    assert os.path.exists(filename), \
      'Selective search data not found at: {}'.format(filename)
    raw_data = sio.loadmat(filename)['boxes'].ravel()

    box_list = []
    for i in range(raw_data.shape[0]):
      boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
      keep = ds_utils.unique_boxes(boxes)
      boxes = boxes[keep, :]
      keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
      boxes = boxes[keep, :]
      box_list.append(boxes)

    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_annotation(self, index):
    """
    Load image and bounding boxes info from file in the dataset
    format.
    """
    filename = os.path.join(self._data_path, 'annotations', index + '.csv')
    df = pd.read_csv(filename)
    num_objs = len(df)
    height, width = (500, 500)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 2), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    for ix, row in list(df.iterrows()):
        x1 = max(0, int(row["x"]) - 1)
        y1 = max(0, int(row["y"]) - 1)
        x2 = min(width, int(row["x"] + row["dx"]) - 1) 
        y2 = min(height, int(row["y"] + row["dy"]) - 1)
        cls = self._class_to_ind["mango"]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

if __name__ == '__main__':
  from datasets.fruits import fruits
  d = fruits('train')