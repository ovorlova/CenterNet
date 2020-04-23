from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from tensorboardX import SummaryWriter

import torch.utils.data as data
import sys
sys.path.insert(0, '../../../../eval')
from evaluate import eval
from eval_helpers import *

class MPII(data.Dataset):
  num_classes = 1
  num_joints = 16
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  # mean and std : https://github.com/xingyizhou/CenterNet/issues/654
  flip_idx = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]] 
  def __init__(self, opt, split):
    super(MPII, self).__init__()
    self.edges = [[0,1], [1,2], [2,6], [7,12], [12,11], [11,10], [5,4], [4,3], [3,6], [7,13], [13,14], [14,15], [6,7], [7,8], [8,9]] 
    self.acc_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    self.data_dir = os.path.join(opt.data_dir, 'coco')
    self.img_dir = self.data_dir
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'val.json').format(split)
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations', 
        '{}.json').format(split)
    self.max_objs = 32
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
 # https://github.com/xingyizhou/CenterNet/issues/280
    self.split = split
    self.opt = opt
    self.gtFrames = loadGTFrames('../../data/coco/annotations/', 'val_not_single.json')
    self.gtFramesSingle = loadGTFrames('../../data/coco/annotations/', 'val_single.json')

    print('==> initializing mpii {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    image_ids = self.coco.getImgIds()

    if split == 'train':
      self.images = []
      for img_id in image_ids:
        idxs = self.coco.getAnnIds(imgIds=[img_id])
        if len(idxs) > 0:
          self.images.append(img_id)
    else:
      self.images = image_ids
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes, hms=None):
    # import pdb; pdb.set_trace()
    detections = []
    data = json.load(open(self.annot_path, 'r'))
    imgs = data['images']
    inds = {}
    for img in imgs:
      inds[img['id']] = img['file_name'][7:]
    
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = 1
        cur_id = 0
        for dets in all_bboxes[image_id][cls_ind]:
          bbox = dets[:4]
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = dets[4]
          scores =  np.ones((16, 1), dtype=np.float32)
          if hms is not None:
            for joint in range(16):
              scores[joint] = hms[image_id][0][0][joint][cur_id]
          bbox_out  = list(map(self._to_float, bbox))
          keypoints = np.concatenate([
            np.array(dets[5:37], dtype=np.float32).reshape(-1, 2), 
            scores], axis=1).reshape(48).tolist()
          keypoints  = list(map(self._to_float, keypoints))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score)),
              "keypoints": keypoints
          }
          detections.append(detection)
    data = detections
    dct_image_id = {}
    for i in range(len(data)):
        img_id = data[i]['image_id']
        if inds[img_id] not in dct_image_id:
            dct_image_id[inds[img_id]] = []
        lst = []
        center_score = data[i]['score']
        for key in range(16):
            _id = key
            x = data[i]['keypoints'][key*3]
            y = data[i]['keypoints'][key*3+1]
            score = data[i]['keypoints'][key*3+2]
            lst.append({'id': [_id], 'x': [x], 'y': [y], 'score' : [score]})
        (dct_image_id[inds[img_id]]).append({'score': [center_score], 'annopoints' : [{"point": lst}]})
    final_lst = []
    for key in dct_image_id:
        final_lst.append({'image' : [{'name' : key}], 'annorect' : dct_image_id[key]})
    return final_lst

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir, hms=None):
    json.dump(self.convert_eval_format(results, hms), 
              open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir, hms=None, test=False):
    if test == True:
      self.save_results(results, save_dir, hms)
      print(eval(self.gtFrames, self.gtFramesSingle, self.convert_eval_format(results, hms)))
    else:
      dets = {}
      hms = {}
      for id_ in results:
        dets[id_] = results[id_][0]
        hms[id_] = results[id_][1]
      return eval(self.gtFrames, self.gtFramesSingle, self.convert_eval_format(dets, hms))
