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
##sys.path.append(os.path.join(sys.path[0], '../../../../eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../eval'))
from evaluate import eval, pseudo_eval
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
          '{}.json').format(split)
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations', 
        '{}.json').format(split)

    self.gtFramesSingle = loadGTFrames('../data/coco/annotations/', '{}_single.json'.format(split))
    self.gtFramesMulti = loadGTFrames('../data/coco/annotations/', '{}_multi.json'.format(split))
    fopen = open('../data/coco/annotations/{}_multi_inds.txt'.format(split)) 

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

    lines = fopen.readlines()
    self.multiInds = list(map(int, lines[0][1:-2].split(', ')))    


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

  def convert_eval_format(self, all_bboxes, hms=None, score_=0.0, multi=False):
    
    data = json.load(open(self.annot_path, 'r'))
    imgs = data['images']
    inds = {}
    for img in imgs:
      inds[img['id']] = img['file_name'][7:]
    
    dct_image_id = {}
    for image_id in all_bboxes:
      if multi and image_id not in self.multiInds:
         continue
      detections = []
      for cls_ind in all_bboxes[image_id]:
        category_id = 1
        cur_id = 0
        detections = []
        #dct_image_id[inds[image_id]] = []
        for dets in all_bboxes[image_id][cls_ind]:
          score = dets[4]
          scores =  np.ones((16, 1), dtype=np.float32)
          if hms is not None:
            for joint in range(16):
              scores[joint] = hms[image_id][0][0][joint][cur_id]
          keypoints = np.concatenate([
            np.array(dets[5:37], dtype=np.float32).reshape(-1, 2), 
            scores], axis=1).reshape(48).tolist()
          keypoints  = list(map(self._to_float, keypoints))
          cur_id+=1

          annopoints = []
          for key in range(16):
            _id = key
            x = keypoints[key*3]
            y = keypoints[key*3+1]
            _score = keypoints[key*3+2]
            annopoints.append({'id': [_id], 'x': [x], 'y': [y], 'score' : [_score]})
          #if score>=score_:
          detections.append({'score': [score], 'annopoints' : [{"point": annopoints}]})
        detections.sort(reverse=True, key=lambda x: x['score'][0])
        final_detections = []
        counter = 0
        for detection in detections:
          if detection['score'][0] >= score_:
            final_detections.append(detection)
            counter+=1
          else:
            break
        while (counter < 2):
          final_detections.append(detections[counter])
          counter+=1
          #if score>=score_:
       
        dct_image_id[inds[image_id]] = final_detections
    print("len of dict: ", len(dct_image_id))
    final_lst = []
    for key in dct_image_id:
        final_lst.append({'image' : [{'name' : key}], 'annorect' : dct_image_id[key]})
    return final_lst

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir, hms=None):
    json.dump(self.convert_eval_format(results, hms), 
              open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir, hms=None, test=False, score=0.0):
    #if test == True:
    #  self.save_results(results, save_dir, hms)
    if hms is not None:
      return pseudo_eval(self.gtFramesSingle, self.convert_eval_format(results, hms, score_=score), 
                          self.gtFramesMulti, self.convert_eval_format(results, hms, score_=score, multi=True))
    else:
      return pseudo_eval(self.gtFramesSingle, self.convert_eval_format(results, None), 
                          self.gtFramesMulti, self.convert_eval_format(results, None, multi=True)) 
