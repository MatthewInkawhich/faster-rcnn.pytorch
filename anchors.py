# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
#from scipy.misc import imread
from imageio import imread, imwrite
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import datasets.xview
import pdb

# My imports
#sys.path.insert(0, '/raid/inkawhmj/WORK/data/xView_data_utilities')
#import aug_util as aug
#import wv_util as wv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################################
### Settings
################################################
data_path = 'data/xView-voc-700'
image_path = data_path + '/JPEGImages/img_104_6_rot0.png'
net = 'res101'
cfg_file = 'cfgs/xview/A.yml'
class_agnostic = False
num_anchors_per_pos = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS)

bbox_row = 50
bbox_col = 50


################################################
### Helpers
################################################
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def load_pascal_boxes(index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    #gt_classes = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        #cls = self._class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        #gt_classes[ix] = cls

    return boxes
            #'gt_classes': gt_classes}


def draw_bboxes(img, boxes, color='red', linesize=3):
    """
    A helper function to draw bounding box rectangles on images
    Args:
        img: image to be drawn on in array format
        boxes: An (N,4) array of bounding boxes
    Output:
        Image with drawn bounding boxes
    """
    #source = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    w2,h2 = (img.width, img.height)

    idx = 0

    for b in boxes:
        xmin,ymin,xmax,ymax = b
        
        for j in range(linesize):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline=color)
    return img


def draw_anchor_centers(img, anchors, color='red', linesize=3):
    # find center position mini-boxes
    centered_mini_boxes = []
    for box in anchors:
        xc = (box[2] + box[0]) // 2
        yc = (box[3] + box[1]) // 2
        minibox = [xc, yc, xc+1, yc+1]
        centered_mini_boxes.append(minibox)
    
    # draw miniboxes
    dimage = draw_bboxes(img, centered_mini_boxes, color=color, linesize=linesize)
    return dimage
        
    
def draw_anchor_types(img, index, anchors, linesize=3):
    color_choices = ['red', 'green', 'blue', 'purple', 'orange', 'white']
    # Get anchors corresponding to index
    index *= num_anchors_per_pos
    assert index >= 0 and index < len(anchors)
    box_set = anchors[index:index+num_anchors_per_pos]
    # Plot boxes in set
    for i in range(len(cfg.ANCHOR_SCALES)):
        for j in range(i, num_anchors_per_pos, len(cfg.ANCHOR_SCALES)):
            dimage = draw_bboxes(img, [box_set[j]], color=color_choices[i], linesize=linesize)
    return dimage


def draw_gt_boxes(img, image_index, color='yellow', linesize=3):
    boxes_np = load_pascal_boxes(image_index)
    boxes = boxes_np.tolist()
    dimage = draw_bboxes(img, boxes, color=color, linesize=linesize)
    return dimage
    
 

################################################
### Main
################################################
if __name__ == '__main__':

  #####  Load configs
  cfg_from_file(cfg_file)
  cfg.USE_GPU_NMS = torch.cuda.is_available()
  np.random.seed(cfg.RNG_SEED)
  print("loaded configs")

  classes = datasets.xview.read_classes_from_file(data_path)

  ##### Initilize the network
  if net == 'vgg16':
    fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=class_agnostic)
  elif net == 'res101':
    fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=class_agnostic)
  elif net == 'res50':
    fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=class_agnostic)
  elif net == 'res152':
    fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()
  fasterRCNN.to(device)
  fasterRCNN.eval()
  print('built model successfully!')

  ##### Prep the image for the model
  # Create tensor placeholders
  im_data = torch.empty(1, dtype=torch.float32, device=device)
  im_info = torch.empty(1, dtype=torch.float32, device=device)
  num_boxes = torch.empty(1, dtype=torch.int64, device=device)
  gt_boxes = torch.empty(1, dtype=torch.float32, device=device)

  # Read image to numpy array
  im_in = np.array(imread(image_path))
  orig = np.copy(im_in)
  # If image is grayscale, convert to rgb
  if len(im_in.shape) == 2:
    im_in = im_in[:,:,np.newaxis]
    im_in = np.concatenate((im_in,im_in,im_in), axis=2)
  # rgb -> bgr
  im = im_in[:,:,::-1]
  
  # Scale image so smallest size is length cfg.TEST.SCALES
  blobs, im_scales = _get_image_blob(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"
  im_blob = blobs
  im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  # Convert numpy arrays to torch tensors
  im_data_pt = torch.from_numpy(im_blob)
  im_data_pt = im_data_pt.permute(0, 3, 1, 2)
  im_info_pt = torch.from_numpy(im_info_np)
  # Copy temporary torch tensors into placeholders
  with torch.no_grad():
          im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
          im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
          gt_boxes.resize_(1, 1, 5).zero_()
          num_boxes.resize_(1).zero_()


  ##### Forward pass image through detection model
  anchors = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, get_anchors=True)
  anchors = anchors.detach().cpu().numpy().astype(np.int32)


  ##### Draw on image
  dimage = Image.fromarray(orig)
  
  # Choose anchors to show  
  anchors_to_show = []
  
  for i in range(0, len(anchors), num_anchors_per_pos):
    anchors_to_show.append(anchors[i])

  # Extract image index from image_path
  image_index = image_path.split('/')[-1].split('.')[0]

  #dimage = draw_anchor_centers(dimage, anchors_to_show, linesize=1)
  dimage = draw_anchor_types(dimage, 1550, anchors)
  dimage = draw_gt_boxes(dimage, image_index)

  ##### Show image
  plt.imshow(dimage)
  plt.show()
