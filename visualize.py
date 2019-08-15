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
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import time
import cv2
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
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
from model.utils.net_utils import save_net, load_net, vis_detections, vis_color_coded
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet, myresnet, myresnet2
import pdb
import datasets.xview
import anchors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################################
### Settings
################################################
data_path = 'data/xView-voc-600'

#image_path = data_path + '/JPEGImages/img_492_8_rot0.png'
#image_path = data_path + '/JPEGImages/img_322_30_rot0.png'  # stride
#image_path = data_path + '/JPEGImages/img_1397_25_rot0.png'  # stride
#image_path = data_path + '/JPEGImages/img_237_28_rot0.png'  # orchard sparse
#image_path = data_path + '/JPEGImages/img_2499_16_rot0.png'  # buildings blend in
image_path = data_path + '/JPEGImages/img_2026_16_rot0.png'  # housing development
#image_path = data_path + '/JPEGImages/img_1175_33_rot0.png'   # scale
#image_path = data_path + '/JPEGImages/img_546_29_rot0.png'  # scale
#image_path = data_path + '/JPEGImages/img_763_19_rot0.png'
#image_path = data_path + '/JPEGImages/img_110_16_rot0.png'
#image_path = data_path + '/JPEGImages/img_1127_16_rot0.png'
#image_path = data_path + '/JPEGImages/img_1444_51_rot0.png'
#image_path = data_path + '/JPEGImages/img_2009_38_rot0.png'
#image_path = data_path + '/JPEGImages/img_110_16_rot0.png'

cfg_file = 'cfgs/xview/600_A_8.yml'
meta_path = 'data/xView-meta'
dataset = 'xview_600'
load_dir = 'models'
net = 'res101'
checksession = 1
layer_cfg = [3,4,23]
checkepoch = 6
#checkpoint = 23376
#checkpoint = 8000
checkpoint = 46754
class_agnostic = False
#conf_thresh_for_det = 0.05
vis_thresh = 0.4
conf_thresh_for_det = vis_thresh
iou_thresh = 0.5
text = False
plot_gt = True
save = False




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


def _load_gt_boxes(data_path, index, class_to_ind):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 5), dtype=np.uint32)
    #gt_classes = np.zeros((num_objs), dtype=np.int32)
    
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        cls = class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2, cls]
        #gt_classes[ix] = cls

    return boxes #, gt_classes



################################################
### Main
################################################
if __name__ == '__main__':

  # Load configs
  cfg_from_file(cfg_file)
  cfg.USE_GPU_NMS = torch.cuda.is_available()
  np.random.seed(cfg.RNG_SEED)
  print("loaded configs")

  input_dir = os.path.join(load_dir, net, dataset)
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}_{}.pth'.format(cfg.EXP_DIR, checksession, checkepoch, checkpoint))

  classes = datasets.xview.read_classes_from_file(meta_path)

  # Load imdb
  #imdb, roidb, ratio_list, ratio_index = combined_roidb('xview_'+dataset.split('_')[1]+'_val', training=False, shift=cfg.PIXEL_SHIFT)

  # initilize the network here.
  if net == 'vgg16':
    fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=class_agnostic)
  elif net == 'res101':
    fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=class_agnostic)
  elif net == 'res50':
    fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=class_agnostic)
  elif net == 'res152':
    fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=class_agnostic)
  elif net == 'res101_custom2':
    fasterRCNN = myresnet2(classes, layer_cfg, 101, pretrained=False, class_agnostic=class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()
  fasterRCNN.to(device)
  fasterRCNN.eval()
  print('built model successfully!')
  print("loading checkpoint %s" % (load_name))

  if torch.cuda.is_available():
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=('cpu'))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  # Create tensor placeholders
  im_data = torch.empty(1, dtype=torch.float32, device=device)
  im_info = torch.empty(1, dtype=torch.float32, device=device)
  num_boxes = torch.empty(1, dtype=torch.int64, device=device)
  gt_boxes = torch.empty(1, dtype=torch.float32, device=device)


  # Load gt boxes
  class_to_ind = dict(zip(classes, range(len(classes))))
  gt_boxes_all = _load_gt_boxes(data_path, image_path.split('/')[-1].split('.')[0], class_to_ind)
  #print("gt_boxes_all:", gt_boxes_all, gt_boxes_all.shape)
  #exit()

  # Read image to numpy array
  im_in = np.array(imread(image_path))
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

  det_tic = time.time()

  # Forward pass image through detection model
  rois, cls_prob, bbox_pred, \
  rpn_loss_cls, rpn_loss_box, \
  RCNN_loss_cls, RCNN_loss_bbox, \
  rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

  print("rois:", rois, rois.size())
  print("cls_prob:", cls_prob, cls_prob.size())
  print("bbox_pred:", bbox_pred, bbox_pred.size())
  print("RCNN_loss_cls:", RCNN_loss_cls)
  print("RCNN_loss_bbox:", RCNN_loss_bbox)
  print("rois_label:", rois_label)

  scores = cls_prob.data
  boxes = rois.data[:, :, 1:5]

  if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred.data
      if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
      # Optionally normalize targets by a precomputed mean and stdev
        box_deltas = box_deltas.view(-1, 4) * torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=torch.float32, device=device) \
                     + torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=torch.float32, device=device)
   
        if class_agnostic:
            box_deltas = box_deltas.view(1, -1, 4)
        else:
            box_deltas = box_deltas.view(1, -1, 4 * len(classes))

      # transform and clip boxes to lay on image
      pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
      pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
  else:
      # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  # Rescale prediction boxes to original image scale
  pred_boxes /= im_scales[0]

  scores = scores.squeeze()
  pred_boxes = pred_boxes.squeeze()
  det_toc = time.time()
  detect_time = det_toc - det_tic
  im2show = np.copy(im)

  # bgr -> rgb
  im2show = im2show[:,:,::-1]

  # Plot ground truth if desired
  im2show = Image.fromarray(im2show)
  if plot_gt:
      im2show = anchors.draw_gt_boxes(im2show, data_path, image_path.split('/')[-1].split('.')[0], color="blue", linesize=2)
  im2show = np.array(im2show)

  for j in range(1, len(classes)):
      # Gather indices of the boxes that detected class j with '> thresh' confidence 
      inds = torch.nonzero(scores[:,j]>conf_thresh_for_det).view(-1)
      # If there is a detection
      if inds.numel() > 0:
        cls_scores = scores[:,j][inds]
        _, order = torch.sort(cls_scores, 0, True)
        if class_agnostic:
          cls_boxes = pred_boxes[inds, :]
        else:
          cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
        
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        cls_dets = cls_dets[order]
        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]
        # Keep adding boxes to existing im2show
        im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), thresh=vis_thresh)
        #im2show = vis_color_coded(im2show, j, classes[j], cls_dets.cpu().numpy(), gt_boxes_all, thresh=vis_thresh, iou_thresh=iou_thresh, text=text)



# Plot/Save image
image_id = image_path.split('/')[-1].split('.')[0].split('_')[1]
chip_id = image_path.split('/')[-1].split('.')[0].split('_')[2]
model_id = cfg_file.split('/')[-1].split('.')[0].replace('_', '-')
save_name = image_path.split('/')[-1].split('.')[0] + '__' + model_id + '.pdf'

#plt.axis('off')
plt.title('Img: {}-{}   Model: {}'.format(image_id, chip_id, model_id))
plt.imshow(im2show)
if save:
    plt.savefig('/raid/inkawhmj/WORK/xview_project/images/' + save_name)
plt.show()


