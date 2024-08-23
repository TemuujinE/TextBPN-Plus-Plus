# -*- coding: utf-8 -*-
__author__ = "RazyDave"
import warnings
warnings.filterwarnings("ignore")
import os
import re
import ast
import numpy as np
import scipy.io as io
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
import cv2
from util import io as libio

class SynthData(TextDataset):

    def __init__(self, data_root, gt_file_name, ignore_list=None, is_training=True, load_memory=False, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.gt_file_name = gt_file_name
        self.is_training = is_training
        self.load_memory = load_memory
        
        gt_path = os.path.join(data_root, gt_file_name)
        with open(gt_path, 'r') as f:
            self.lines = f.readlines()
            
        print(f"LOADED GT TXT: {len(self.lines)}")
        
        if self.load_memory:
            self.datas = list()
            for item in range(len(self.lines)):
                self.datas.append(self.load_img_gt(item))
            
    @staticmethod
    def parse_points(points):
        polygons = []
        for [x1, y1],[x2, y2], [x3, y3], [x4, y4] in points:
            xx = [x1, x2, x3, x4]
            yy = [y1, y2, y3, y4]
            
            pts = np.stack([xx, yy]).T.astype(np.int32)
            polygons.append(TextInstance(pts, 'c', '#'))
        return polygons
        
    def load_img_gt(self, item):
        img_dir, points = self.lines[item].split('\t')
        points = [np.asarray(val['points']) for val in ast.literal_eval(points)]
        polygons = self.parse_points(points)
        
        image_id = os.path.basename(img_dir)
        image_path = os.path.join(self.data_root, img_dir)

        # Read image data
        image = pil_load_img(image_path)
        try:
            h, w, c = image.shape
            assert (c == 3)
        except:
            image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):

        if self.load_memory:
            data = self.datas[item]
        else:
            data = self.load_img_gt(item)

        if self.is_training:
            return self.get_training_data(data["image"], data["polygons"],
                                          image_id=data["image_id"], image_path=data["image_path"])
        else:
            return self.get_test_data(data["image"], data["polygons"],
                                      image_id=data["image_id"], image_path=data["image_path"])

    def __len__(self):
        return len(self.lines)