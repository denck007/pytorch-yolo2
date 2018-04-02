#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
from image import *

class listDataset(Dataset):

    def __init__(self, imagelist, labelFolder, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
        with open(imagelist, 'r') as file:
           self.lines = file.readlines()

        if shuffle:
           random.shuffle(self.lines)
        #print("found {} images".format(len(self.lines)))
        # find the index of the label file for each of the images
        if labelFolder is not None:
            self.labelfiles = os.listdir(labelFolder)
            #print("found {} labels".format(len(self.labelfiles)))
            imgs = [self.lines[ii][self.lines[ii].rfind("/")+1:] for ii in range(len(self.lines))]
            imgs = [imgs[ii][:imgs[ii].rfind(".")] for ii in range(len(imgs))]
            labels = [label[:label.rfind(".")] for label in os.listdir(labelFolder)]
            imageIdxs = []
            labelIdxs = []
            imagesWithNoLabel = []
            for ii in range(len(imgs)):
                try:
                    imageIdxs.append(ii)
                    labelIdxs.append(labels.index(imgs[ii]))
                except:
                    imagesWithNoLabel.append(ii)
            if len(imagesWithNoLabel) > 0:
                print("Found {} images with no label file".format(len(imagesWithNoLabel)))
                #for l in imagesWithNoLabel:
                #    print("\t" + imgs[l])                
            self.imgIdxToLabelIdx = dict(zip(imageIdxs,labelIdxs))
            self.labelfiles = [os.path.join(labelFolder,lab) for lab in self.labelfiles]

        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def _getImgPathLabelPath(self,index):
        
        try:
            imgpath = self.lines[index].rstrip()
            labelpath = self.labelfiles[self.imgIdxToLabelIdx[index]]
            return  [imgpath,labelpath]
        except:
            print("Error trying to read imgIdxToLabelIdx: {}".format(self.imgIdxToLabelIdx[index]))
            print("Image file involved is {}".format(self.lines))
            return self._getImgPathLabelPath(index+1) # if it is bad, just keep going till a good one is found

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath,labelpath = _getImgPathLabelPath(index)

        if self.train and index % 64== 0:
            if self.seen < 4000*64:
               width = 13*32
               self.shape = (width, width)
            elif self.seen < 8000*64:
               width = (random.randint(0,3) + 13)*32
               self.shape = (width, width)
            elif self.seen < 12000*64:
               width = (random.randint(0,5) + 12)*32
               self.shape = (width, width)
            elif self.seen < 16000*64:
               width = (random.randint(0,7) + 11)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64:
               width = (random.randint(0,9) + 10)*32
               self.shape = (width, width)

        if self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            img, label = load_data_detection(imgpath, labelpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            label = torch.zeros(50*5)
            #if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1,5)
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)
