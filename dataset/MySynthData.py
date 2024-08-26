# -*- coding: utf-8 -*-
__author__ = "RazyDave"
import warnings
warnings.filterwarnings("ignore")
import os
import re
import ast
import numpy as np
import pandas as pd
import scipy.io as io
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
import cv2
from util import io as libio
from tqdm import tqdm

class SynthData(TextDataset):

    def __init__(self, data_root, gt_file_name, gt_data_dirs, gt_dir = 'gt_word_by_word', train_dir = 'train_images',
                 ignore_list=None, is_training=True, load_memory=False, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.gt_dir = gt_dir
        self.train_dir = train_dir
        self.gt_data_dirs = gt_data_dirs
        self.gt_file_name = gt_file_name
        self.is_training = is_training
        self.load_memory = load_memory
        
        gt_names = []
        for data_dir in tqdm(gt_data_dirs):
            file_names = [f"{gt_name.split('.')[0]}" for gt_name in os.listdir(f"{data_root}/{data_dir}/{gt_dir}")]
            gt_names.extend(file_names)
        print(f"Detection level: {gt_dir}, GT Count: {len(gt_names)}")
        
        image_df = {
            "gt_folder": [],
            "img_name": [],
        }
        for img_name in tqdm(os.listdir(f"{data_root}/{train_dir}"), desc = 'Preparing ground truth image and gt folder pair'):
            if img_name.split('.')[0] in gt_names:
                img_folder = '_'.join(img_name.split('_file_')[0].split('_')[2:])
                
                image_df['gt_folder'].append(img_folder)
                image_df['img_name'].append(img_name)
        image_df = pd.DataFrame(image_df).reset_index(drop=True)
        print(f"Training image count: {len(image_df)}")
        self.image_df = image_df
        
        if self.load_memory:
            self.datas = list()
            for item in tqdm(range(len(self.image_df)), total = len(self.image_df)):
                self.datas.append(self.load_img_gt(item))
            
    #@staticmethod
    def parse_points(self, gt_folder, img_name):
        gt_path = f"{self.data_root}/{gt_folder}/{self.gt_dir}/{img_name.split('.')[0]}.txt"
        with open(gt_path, 'r') as f:
            _, gt_bboxes = f.readline().strip().split('\t')
        gt_bboxes = ast.literal_eval(gt_bboxes)
        
        polygons = []
        for dict_pair in gt_bboxes:
            word = dict_pair['transcription']
            [x1, y1], [x2, y2], [x3, y3], [x4, y4] = dict_pair['points']
            
            xx = [x1, x2, x3, x4]
            yy = [y1, y2, y3, y4]
            
            pts = np.stack([xx, yy]).T.astype(np.int32)
            polygons.append(TextInstance(pts, 'c', word))
        return polygons
        
    def load_img_gt(self, item):
        gt_folder, img_name = self.image_df.iloc[item]
        polygons = self.parse_points(gt_folder, img_name)
        image_path = f"{self.data_root}/{self.train_dir}/{img_name}"

        image = pil_load_img(image_path)
        try:
            _, _, c = image.shape
            assert (c == 3)
        except:
            image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = img_name
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
        return len(self.image_df)