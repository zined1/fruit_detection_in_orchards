#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import json

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__', 'mangoe')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_25000.ckpt',)}
DATASETS= {'fruits_dataset': ('fruits_dataset_train',)}
PATH = "fruits_dataset"


def export(im, im_name, class_name, dets, inds, infos, vis=True, thresh=0.5):
    """Draw detected bounding boxes."""
    infos["num_objects"] = len(inds)
    for i in inds:
        bbox = dets[i,:4]
        score = dets[i, -1]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        infos["loc"].append([x1, y1, x2, y2])
        if vis:
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #cv2.putText(im, class_name, (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    if vis:
        cv2.imshow(im_name, im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return infos

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, args.path, "images", image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    infos = {'num_objects': 0, "loc": []}
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            #vis_detections(im, cls, dets, inds, thresh=CONF_THRESH)
        infos = export(im, image_name, cls, dets, inds, infos, vis=True, thresh=CONF_THRESH)
    return infos

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo for fruits_datasets')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset',
                        choices=DATASETS.keys(), default='fruits_dataset')
    parser.add_argument('--path', dest='path', help='Trained dataset',
                        default=PATH)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    path_dataset = args.path
    print(path_dataset)
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 2, tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    test_files_path = os.path.join(cfg.DATA_DIR, path_dataset, "sets", "test.txt")
    test_files = open(test_files_path, 'r')

    try:
        im_names = test_files.read().split("\n")[:-1]
        output_json = {}
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Demo for data/demo/{}'.format(im_name))
            output_json[im_name] = demo(sess, net, im_name + ".png")
        with open('output.json', 'w') as fp:
            json.dump(output_json, fp)
    finally:
        test_files.close()