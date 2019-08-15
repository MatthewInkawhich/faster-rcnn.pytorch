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
import csv
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
import itertools
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_color_coded
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet, myresnet, myresnet2

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


# Compute IoU function
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
     
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
     
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
     
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
     
    # return the intersection over union value
    return iou


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


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--iou_thresh', dest='iou_thresh',
                      help='IoU threshold to remove duplicate',
                      type=float, default=0.8)
  parser.add_argument('--layer_cfg', dest='layer_cfg', default='[3,4,23]', type=str)
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis_thresh = 0.6

# Counters and sums
total_ff_time = 0
total_chip_time = 0
total_chip_count = 0

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  layer_cfg = list(map(int, args.layer_cfg.strip('[]').split(',')))
  print("layer_cfg:", layer_cfg)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)

  # Set imdbval_names
  args.imdbval_name = "xview_" + args.dataset.split('_')[-1] + "_val"
  imdbval_name_ff = "xview_ff_val"

  args.set_cfgs = None

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_HFLIPPED = False
  cfg.TRAIN.USE_VFLIPPED = False

  # Load chipped imdb
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, training=False, shift=cfg.PIXEL_SHIFT)
  imdb.competition_mode(on=True)
  # Load ff imdb
  imdb_ff, roidb_ff, ratio_list_ff, ratio_index_ff = combined_roidb(imdbval_name_ff, training=False, shift=cfg.PIXEL_SHIFT)
  imdb_ff.competition_mode(on=True)

  print('{:d} roidb entries (chip)'.format(len(roidb)))
  print('{:d} roidb entries (ff)'.format(len(roidb_ff)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}_{}.pth'.format(cfg.EXP_DIR, args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101_custom':
    fasterRCNN = myresnet(imdb.classes, layer_cfg, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101_custom2':
    fasterRCNN = myresnet2(imdb.classes, layer_cfg, 101, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()







  #load_name = 'models/res101/xview_600/faster_rcnn_xview_600-A-16_1_6_23376.pth'
  #print("load checkpoint %s" % (load_name))
  #checkpoint = torch.load(load_name)
  #print(checkpoint['model'])
  #modified_sd = {}
  #for key, value in checkpoint['model'].items():
  #    modified_sd[key] = value
  #    if args.net == 'res101_custom2':
  #      if 'RCNN_top.0.' in key:
  #          new_key = key.replace('RCNN_top.0', 'layer4')
  #          modified_sd[new_key] = value
      
  #for each in modified_sd:
  #    print("each:", each)

  #fasterRCNN.load_state_dict(modified_sd)

  #print("model loaded!!!")
  #print(fasterRCNN)




  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])





  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)
  print('load model successfully!')

  class_to_ind = dict(zip(imdb.classes, range(len(imdb.classes))))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  #max_per_image = cfg.MAX_NUM_GT_BOXES * 2
  max_per_image = 20000

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = cfg.EXP_DIR
  num_images = len(imdb.image_index)
  num_images_ff = len(imdb_ff.image_index)
  all_boxes = [[[] for _ in xrange(num_images_ff)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb_ff, save_name)

  # full frame dataloader
  dataset_ff = roibatchLoader(roidb_ff, ratio_list_ff, ratio_index_ff, 1, \
                        imdb_ff.num_classes, training=False, normalize = False)
  dataloader_ff = torch.utils.data.DataLoader(dataset_ff, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)
  data_iter_ff = iter(dataloader_ff)

  # chip dataloader
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)
  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  # Get ff_to_chip_map
  print("Collecting ff_to_chip_map")
  ff_to_chip_map = imdb.get_ff_to_chip_map(os.path.join(imdb_ff._devkit_path, 'ImageSets', 'Main', 'val.txt'))
  print("Done!")
  
  # Read chip_offsets.csv into dict
  chip_offsets = imdb.get_chip_offsets()

  # Iterate over all ff images
  for i in range(num_images_ff):
      # temp changes for chip size vis
      data_ff = next(data_iter_ff)
      image_path_ff = data_ff[4][0]
      print(image_path_ff)
      img_id_ff = image_path_ff.split('/')[-1].split('.')[0]
      #img_id_ff = 'img_2032'
      print(img_id_ff)
      gt_boxes_all = _load_gt_boxes('data/xView-voc-ff', img_id_ff, class_to_ind)
      ff_id_num = img_id_ff.split('_')[-1]
      if vis:
          #im = cv2.imread(imdb_ff.image_path_at(i, tif=True))
          #im = cv2.imread('data/xView-meta/train_images/' + ff_id_num + '.tif')
          im = np.array(Image.open('data/xView-meta/train_images/' + ff_id_num + '.tif'))
          im2show = np.copy(im)


      # Iterate over all chips associated with current ff image
      ff_det_tic = time.time()
      for chip_idx, chip_path in ff_to_chip_map[img_id_ff]:
          #print(chip_idx, "\t", chip_path)
          total_chip_count += 1
          chip_num = chip_path.split('/')[-1].split('.')[0].split('_')[2]
          data = dataset[chip_idx]
          data = list(data)
          # Get data to expected format (normally the dataloader does this)
          data[0].unsqueeze_(0)
          data[1].unsqueeze_(0)
          data[2].unsqueeze_(0)
          data[3] = torch.tensor([data[3]])
          data[4] = [data[4]]
          #print("data[0]:", data[0])
          #print("data[1]:", data[1])
          #print("data[2]:", data[2])
          #print("data[3]:", data[3])
          #print("data[4]:", data[4])
          #exit()
          assert (chip_path == data[4][0]), "Chip paths do not match!"

          # Fetch offsets associated with this chip
          x_offset, y_offset = chip_offsets[(img_id_ff, chip_num)]

          # Load data into holder tensors
          with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

          #print("im_data:", im_data, im_data.size())
          #print("im_data:", im_data, im_data.size())
          #print("gt_boxes:", gt_boxes)
          #print("num_boxes:", num_boxes)

          det_tic = time.time()
          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

          #print("Temp stop")
          #exit()

          scores = cls_prob.data
          boxes = rois.data[:, :, 1:5]

          if cfg.TEST.BBOX_REG:
              # Apply bounding-box regression deltas
              box_deltas = bbox_pred.data
              if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
              # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

              pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
              pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
          else:
              # Simply repeat the boxes, once for each class
              pred_boxes = np.tile(boxes, (1, scores.shape[1]))

          pred_boxes /= data[1][0][2].item()

          scores = scores.squeeze()
          pred_boxes = pred_boxes.squeeze()
          det_toc = time.time()
          detect_time = det_toc - det_tic
          misc_tic = time.time()


          #print("scores:", scores.size(), scores.element_size() * scores.nelement())
          #print("pred_boxes:", pred_boxes.shape, pred_boxes.element_size() * pred_boxes.nelement())
          #print("box_deltas:", box_deltas.size(), box_deltas.element_size() * box_deltas.nelement())

          for j in xrange(1, imdb.num_classes):
              inds = torch.nonzero(scores[:,j]>thresh).view(-1)
              # if there is det
              if inds.numel() > 0:
                #print("scores:", scores, scores.size(), scores.dtype)
                cls_scores = scores[:,j][inds]
                #print("cls_scores:", cls_scores, cls_scores.size(), cls_scores.dtype)
                _, order = torch.sort(cls_scores, 0, True)
                #print("order:", order, order.size(), order.dtype)
                if args.class_agnostic:
                  cls_boxes = pred_boxes[inds, :]
                else:
                  cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                #print("cls_boxes:", cls_boxes, cls_boxes.size(), cls_boxes.dtype)
                
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                #print("cls_dets:", cls_dets, cls_dets.size(), cls_dets.dtype)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                #print("keep:", keep, keep.size(), keep.dtype)
                cls_dets = cls_dets[keep.view(-1).long()]
                #print("cls_dets:", cls_dets, cls_dets.size(), cls_dets.dtype)
                
                # Create a variable for the cls_dets numpy
                cls_dets_np = cls_dets.cpu().numpy()

                # Add the offsets to the bbox coords depending on chip number
                cls_dets_np[:,0] += x_offset
                cls_dets_np[:,1] += y_offset
                cls_dets_np[:,2] += x_offset
                cls_dets_np[:,3] += y_offset


                # if all_boxes[j][i] is empty list, set it to cls_dets
                if all_boxes[j][i] == []:
                    all_boxes[j][i] = cls_dets_np
                else:
                    # Else, all_boxes[j][i] is not an empty list, so append cls_dets to existing dets
                    all_boxes[j][i] = np.append(all_boxes[j][i], cls_dets_np, axis=0)

              else:
                # Only set all_boxes[j][i] to empty_array if there is nothing currently there
                if all_boxes[j][i] == []:
                    all_boxes[j][i] = empty_array
                

          # Limit to max_per_image detections *over all classes*
          if max_per_image > 0:
              image_scores = np.hstack([all_boxes[j][i][:, -1]
                                        for j in xrange(1, imdb.num_classes)])
              #print("image_scores:", image_scores.shape)
              if len(image_scores) > max_per_image:
                  image_thresh = np.sort(image_scores)[-max_per_image]
                  for j in xrange(1, imdb.num_classes):
                      keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                      all_boxes[j][i] = all_boxes[j][i][keep, :]

          misc_toc = time.time()
          nms_time = misc_toc - misc_tic

          print("chip:{}/{} det:{:.5f} nms:{:.5f}  tot:{:.5f}".format(chip_num, len(ff_to_chip_map[img_id_ff])-1, detect_time, nms_time, detect_time+nms_time))
          total_chip_time += detect_time + nms_time
          

      # Final NMS
      #print("\n\nFinal NMS + Manual IoU-based duplicate filter***")
      for j in xrange(1, imdb.num_classes):
          # Skip if no instances
          if all_boxes[j][i].size == 0:
              continue
             
          #print("Instances of {} before final NMS:".format(j), all_boxes[j][i].shape[0])
          # Else, create torch tensors needed
          dets_ff = torch.tensor(all_boxes[j][i], dtype=torch.float32).to(device)
          #print("dets_ff:", dets_ff, dets_ff.size(), dets_ff.dtype)
          cls_scores = dets_ff[:,4]
          #print("cls_scores:", cls_scores, cls_scores.size(), cls_scores.dtype)
          # Sort based on score
          _, order = torch.sort(cls_scores, 0, True)
          #print("order:", order, order.size(), order.dtype)
          cls_boxes = dets_ff[:, 0:4]
          #print("cls_boxes:", cls_boxes, cls_boxes.size(), cls_boxes.dtype)
          # Reassemble into [X,5] shape tensor
          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
          #print("cls_dets:", cls_dets, cls_dets.size(), cls_dets.dtype)
          cls_dets = cls_dets[order]
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          #print("keep:", keep, keep.size(), keep.dtype)
          cls_dets = cls_dets[keep.view(-1).long()]
          #print("cls_dets:", cls_dets, cls_dets.size(), cls_dets.dtype, "\n")

          # Replace all_boxes[j][i] with nms-reduced set
          all_boxes[j][i] = cls_dets.cpu().numpy()
          #print("Instances of {} after final NMS:".format(j), all_boxes[j][i].shape[0])

          # If vis flag is set, plot boxes on ff
          if vis:
              #im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), vis_thresh)
              im2show = vis_color_coded(im2show, j, imdb.classes[j], cls_dets.cpu().numpy(), gt_boxes_all, thresh=vis_thresh, iou_thresh=0.5, text=False)

#          ### Manual IoU-based duplicate filter
#          # Iterate over all pairs or boxes
#          dets_ff_list = all_boxes[j][i].tolist()
#          # Get unique index pairs
#          idx_pairs = itertools.combinations(range(len(dets_ff_list)), 2)
#          for idx_a, idx_b in idx_pairs:
#              #print("idx_a:", idx_a)
#              #print("idx_b:", idx_b)
#              box_a = dets_ff_list[idx_a][0:4]
#              score_a = dets_ff_list[idx_a][4]
#              box_b = dets_ff_list[idx_b][0:4]
#              score_b = dets_ff_list[idx_b][4]
#              print("box_a:", box_a)
#              print("box_b:", box_b)
#              #print("score_a:", score_a)
#              #print("score_b:", score_b)
#              
#                  
#              iou = bb_intersection_over_union(box_a, box_b)
#              #print("iou:", iou)
#              if box_a[0] >= 3110 and box_a[2] <= 3260 and box_b[1] >= 500 and box_b[3] <= 280:
#                  print("box_a:", box_a[0], box_a[1], box_a[2], box_a[3])
#                  print("box_b:", box_b[0], box_b[1], box_b[2], box_b[3])
#                  print("iou:", iou)
#
#
#              if iou >= args.iou_thresh:
#                  idx_togo = idx_a if score_a < score_b else idx_b
#                  print("REMOVING: ", idx_togo)
#                  del dets_ff_list[idx_togo]
#
#          dets_ff_np = np.array(dets_ff_list, dtype=np.float32)
#          #print("dets_ff_np:", dets_ff_np.shape)
#          all_boxes[j][i] = dets_ff_np
#          
#          print("Instances of {} after final IoU-based removals:".format(j), all_boxes[j][i].shape[0], "\n")


      ff_det_toc = time.time()
      ff_det_time = ff_det_toc - ff_det_tic
      print("full frame: {}/{} tot:{:.5f}\n".format(i, num_images_ff-1, ff_det_time))
      total_ff_time += ff_det_time

      # If vis, show ff with all boxes
      if vis:
          #cv2.imwrite('result.png', im2show)
          #plt.imshow(im2show[617:1630, 718:1960, :])
          plt.imshow(im2show[1900:2950, 0:1000, :])
          plt.savefig('result.pdf')
          plt.show()
          pdb.set_trace()

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb_ff.evaluate_detections(all_boxes, output_dir)

  print("Average time per chip (sec):", total_chip_time / total_chip_count)
  print("Average time per full frame (sec):", total_ff_time / imdb_ff.num_images)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
