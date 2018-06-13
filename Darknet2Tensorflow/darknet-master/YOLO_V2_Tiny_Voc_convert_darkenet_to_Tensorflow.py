# -*- coding: utf-8 -*-
# --------------------------------------
# @File    : YOLO_V2 
# Description :Yolo V2 by tensorflow
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import cv2
import time
import colorsys
import random
from datetime import datetime
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

slim = tf.contrib.slim


#ANCHOR = [[0.57273, 0.677385],[1.87446, 2.06253],[3.33843, 5.47434],[7.88282, 3.52778],[9.77052, 9.16828]]#for coco dataset

ANCHOR = [[1.08,1.19],  [3.42,4.41],  [6.63,11.38],  [9.42,5.11],  [16.62,10.52]] #for voc dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir', 'data', 'Input image foldir name')
tf.app.flags.DEFINE_string('output_img', 'results/detected.jpg', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'data/voc.names', 'File with class names')
tf.app.flags.DEFINE_string('yolov2_ckpt_weight', 'models/yolov2-tiny-voc.ckpt', 'File with yolov2.ckpt names')
tf.app.flags.DEFINE_string('darknet_weights_file', 'weight/yolov2-tiny-voc.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('input_size', 416, 'The input image size for network')

tf.app.flags.DEFINE_float('conf_threshold', 0.2, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.5, 'IoU threshold')
tf.app.flags.DEFINE_integer('max_output_size', 10, 'the maximum number of boxes to be selected by non max suppression')

#
tf.app.flags.DEFINE_boolean('load_darknet_weight', True, 'Whether load darknet weights,other load ckpt by Saver.restore')
tf.app.flags.DEFINE_boolean('save_model_weight', True, 'Whether save  darknet weights as tensorflow ckpt')
tf.app.flags.DEFINE_boolean('save_model_pb', True, 'Whether save  model weights as tensorflow pb')

class YOLOV2_Tiny_Voc(object):
 ##################### 构造函数：初始化yolo中参数#################################################################
 def __init__(self,weights_file, verbose=True):
  # 后面程序打印描述功能的标志位
  self.verbose = verbose

  # 检测超参数
  self.threshold = FLAGS.conf_threshold 
  self.iou_threshold = FLAGS.iou_threshold 
  self.max_output_size = FLAGS.max_output_size
  self.class_names = self._read_coco_labels(FLAGS.class_names)

  
  self.sess = tf.Session()
  self.input_size = FLAGS.input_size
  self.images = self._input_process(self.input_size)
  self.predicts = self._build_network(self.images)
  self.output_sizes = self.input_size//32, self.input_size//32 # downsample 32(2^5) times
  self.boxes, \
  self.scores, \
  self.classes  = self._build_detector(model_output=self.predicts,
           output_sizes=self.output_sizes,
           num_class=len(self.class_names),
           threshold=self.threshold,
           iou_threshold=self.iou_threshold,
           max_output_size=self.max_output_size,
           anchors=ANCHOR)
  
  if FLAGS.load_darknet_weight:
   self.load_ops = self._load_weights(tf.global_variables(scope='yolov2_tiny_voc'), weights_file)
   self.sess.run(self.load_ops)
   if FLAGS.save_model_weight:
    saver = tf.train.Saver()
    saver.save(self.sess,FLAGS.yolov2_ckpt_weight)

  else:
   self._load_ckpts(FLAGS.yolov2_ckpt_weight) # import weight form ckpt
   
  if(FLAGS.save_model_pb):
      self._save_graph_to_file(self.sess, self.sess.graph_def ,"models/yolov2-tiny-voc_frozen_graph.pb") 
  
  #self.sess.run(tf.global_variables_initializer())
  
  #tensorboard for graph
  writer =tf.summary.FileWriter("logs/",graph = self.sess.graph)
  writer.close()
 ####################################################################################################################
 def _read_coco_labels(self,file_name):
     f = open(file_name)
     class_names = []
     for l in f.readlines():
         l = l.strip()
         class_names.append(l)
     return class_names

 def _save_graph_to_file(self, sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph, ["detected_scores","detected_boxes","detected_classes"])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return
  
 def _input_process(self,image_size):
   self.input_images = tf.placeholder(tf.uint8, shape=[None, None, 3],name="input")
   image = tf.to_float(self.input_images)/255.0
   image = tf.image.resize_images(image, tf.constant([image_size, image_size]))
   return  tf.expand_dims(image, 0)
    
 @tf.contrib.framework.add_arg_scope
 def _fixed_padding(self, inputs, kernel_size, mode='CONSTANT', **kwargs):
     """
     Pads the input along the spatial dimensions independently of input size.
 
     Args:
       inputs: A tensor of size [batch, channels, height_in, width_in] or
         [batch, height_in, width_in, channels] depending on data_format.
       kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                    Should be a positive integer.
       data_format: The input format ('NHWC' or 'NCHW').
       mode: The mode for tf.pad.
 
     Returns:
       A tensor with the same format as the input with the data either intact
       (if kernel_size == 1) or padded (if kernel_size > 1).
     """
     pad_total = kernel_size - 1
     pad_beg = pad_total // 2
     pad_end = pad_total - pad_beg
 
     if kwargs['data_format'] == 'NCHW':
              padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                           [pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)
     else:
              padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                         [pad_beg, pad_end], [0, 0]], mode=mode)
     return padded_inputs
    
 def _conv2d_fixed_padding(self, inputs, filters, kernel_size, strides=1):
     if strides > 1:
         inputs = self._fixed_padding(inputs, kernel_size)
     inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding=('SAME' if strides == 1 else 'VALID'))
     return inputs
    
    
    # reorg layer(带passthrough的重组层)
 def _reorg(self,x,stride):
  return tf.space_to_depth(x,block_size=stride)
  # 或者return tf.extract_image_patches(x,ksizes=[1,stride,stride,1],strides=[1,stride,stride,1],
  #                                 rates=[1,1,1,1],padding='VALID')    
  
  
 def _print_activations(self, t):
  print(t.op.name, ' ', t.get_shape().as_list())

 def _build_network(self,
                    images,
                    num_outputs=125,
                    alpha=0.1,
                    keep_prob=0.5,
                    is_training=False,
                    reuse=False,
                    data_format='NHWC',
                    scope='yolov2_tiny_voc'):
  
  batch_norm_params = {
         'is_training': is_training,
         'decay': 0.9997,
         'epsilon': 1e-5,
         'scale': True,
     }
  
        # Set activation_fn and parameters for conv2d, batch_norm.
  with slim.arg_scope([slim.conv2d, slim.batch_norm, self._fixed_padding], data_format=data_format, reuse=reuse):
          with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                             biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=alpha), padding='SAME'):
                with tf.variable_scope(scope):
     
                     net = slim.conv2d(images, 16, 3)    #[1, 416, 416, 32]
                     self._print_activations(net)
                     net = slim.max_pool2d(net, 2, padding='VALID', scope='pool1')    #[1, 208, 208, 32]
                     self._print_activations(net)
                       
                     net = slim.conv2d(net, 32, 3)    #[1, 208, 208, 64]
                     self._print_activations(net)
                     net = slim.max_pool2d(net, 2, padding='VALID', scope='pool2')    #[1, 104, 104, 64]
                     self._print_activations(net)
                       
                     net = slim.conv2d(net, 64, 3)   
                     self._print_activations(net)
                     net = slim.max_pool2d(net, 2, padding='VALID', scope='pool3')   
                     self._print_activations(net)
                     
                     net = slim.conv2d(net, 128, 3)   
                     self._print_activations(net)
                     net = slim.max_pool2d(net, 2, padding='VALID', scope='pool4')   
                     self._print_activations(net)
                     
                     net = slim.conv2d(net, 256, 3)   
                     self._print_activations(net)
                     net = slim.max_pool2d(net, 2, padding='VALID', scope='pool5')   
                     self._print_activations(net)
                     
                     net = slim.conv2d(net, 512, 3)   
                     self._print_activations(net)
                     net = slim.max_pool2d(net, 2, stride=1, padding='SAME', scope='pool6')   
                     self._print_activations(net)
                     
                     net = slim.conv2d(net, 1024, 3)    #[1, 13, 13, 1024]
                     self._print_activations(net)
                     
                     #detection
                     net = slim.conv2d(net, 1024, 3)    #[1, 13, 13, 1024]
                     self._print_activations(net)
                     
                     predicts = slim.conv2d(net, num_outputs, 1, biases_initializer=tf.zeros_initializer(),activation_fn=None,normalizer_fn=None,)
                     self._print_activations(predicts)
                      
                     return predicts

 def _build_detector(self,
     model_output,
     output_sizes=(13,13),
     num_class=80,threshold=0.5,
     iou_threshold=0.5,
     max_output_size=10,
     anchors=None):
  '''
   model_output:darknet19网络输出的特征图
   output_sizes:darknet19网络输出的特征图大小，默认是13*13(默认输入416*416，下采样32)
  '''
  H, W = output_sizes
  num_anchors = len(anchors) # 这里的anchor是在configs文件中设置的
  with tf.name_scope("yolov2_detertor"):
   with tf.variable_scope("slice_shape"):
    # 13*13*num_anchors*(num_class+5)，第一个维度自适应batchsize
    detection_result = tf.reshape(model_output,[-1,H*W,num_anchors,num_class+5],name="feature_maps")
   
    # darknet19网络输出转化——偏移量、置信度、类别概率
    xy_offset = tf.nn.sigmoid(detection_result[:,:,:,0:2]) # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
    wh_offset = tf.exp(detection_result[:,:,:,2:4]) #相对于anchor的wh比例，通过e指数解码
    obj_probs = tf.nn.sigmoid(detection_result[:,:,:,4]) # 置信度，sigmoid函数归一化到0-1
    class_probs = tf.nn.softmax(detection_result[:,:,:,5:],name="scores") # 网络回归的是'得分',用softmax转变成类别概率
  
   with tf.variable_scope("meshgrid"):
    # 构建特征图每个cell的左上角的xy坐标
    height_index = tf.range(H,dtype=tf.float32) # range(0,13)
    width_index = tf.range(W,dtype=tf.float32) # range(0,13)
    # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
    x_cell,y_cell = tf.meshgrid(height_index,width_index)
    x_cell = tf.reshape(x_cell,[1,-1,1]) # 和上面[H*W,num_anchors,num_class+5]对应
    y_cell = tf.reshape(y_cell,[1,-1,1])
      
   with tf.variable_scope("coord-decode"):        
    anchors = tf.constant(anchors, dtype=tf.float32,name="anchor")  # 将传入的anchors转变成tf格式的常量列表
    # decode
    bbox_x = (x_cell + xy_offset[:,:,:,0]) / W
    bbox_y = (y_cell + xy_offset[:,:,:,1]) / H
    bbox_w = (anchors[:,0] * wh_offset[:,:,:,0]) / W
    bbox_h = (anchors[:,1] * wh_offset[:,:,:,1]) / H
    # 中心坐标+宽高box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
    bboxes = tf.stack([bbox_x-bbox_w/2, bbox_y-bbox_h/2,
           bbox_x+bbox_w/2, bbox_y+bbox_h/2], axis=3)
    
   with tf.variable_scope("class-specific"):
    # 类别置信度分数：[1,169,num_anchors,1]*[1,169,1,num_class]=[1,169,num_anchors,类别置信度C num_class]
    scores = tf.expand_dims(obj_probs, -1) * class_probs
    
    scores = tf.reshape(scores, [-1, num_class],name="scores")  # [1*169*num_anchors*num_class, C]
    boxes = tf.reshape(bboxes, [-1, 4],name="boxes")  # [1*169*num_anchors*num_class, 4]
    
    
    # 只选择类别置信度最大的值作为box的类别、分数
    box_classes = tf.argmax(scores, axis=1,name="classes") # 边界框box的类别
    box_class_scores = tf.reduce_max(scores, axis=1,name="class_scores") # 边界框box的分数
    
   with tf.variable_scope("select-threshold"):
    # 利用类别置信度阈值self.threshold，过滤掉类别置信度低的
    filter_mask = box_class_scores >= threshold
    scores = tf.boolean_mask(box_class_scores, filter_mask)
    boxes = tf.boolean_mask(boxes, filter_mask)
    box_classes = tf.to_int32(tf.boolean_mask(box_classes, filter_mask))
    
   with tf.variable_scope("NMS"):
    # NMS (不区分不同的类别)
    #_boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],
    #                     boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]], axis=1)
    #Bounding boxes are supplied as [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners 
    #and the coordinates can be provided as normalized ,(xmin,ymin,xmax,ymax) => (ymin, xmin, ymax, xmax)
    xmin,ymin,xmax,ymax = tf.unstack(tf.transpose(boxes))
    _boxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
    nms_indices = tf.image.non_max_suppression(_boxes, scores,max_output_size, iou_threshold)
   

  scores = tf.identity(tf.gather(scores, nms_indices), name="detected_scores")
  boxes = tf.identity(tf.gather(boxes, nms_indices),name="detected_boxes")#(xmin,ymin,xmax,ymax)
  classes = tf.identity(tf.gather(box_classes, nms_indices),name="detected_classes")
  print(boxes,scores,classes)
   
  return boxes,scores,classes

 def _load_weights(self, var_list, weights_file):
     """
     Loads and converts pre-trained weights.
     :param var_list: list of network variables.
     :param weights_file: name of the binary file.
     :return: list of assign ops
     """
     with open(weights_file, "rb") as fp:
         _ = np.fromfile(fp, dtype=np.int32, count=4)# This is verry import for count,it include the version of yolo
         weights = np.fromfile(fp, dtype=np.float32)
 
     ptr = 0
     i = 0
     assign_ops = []
     while i < len(var_list) - 1:
         var1 = var_list[i]
         #print(i,var1)
         var2 = var_list[i + 1]
         # do something only if we process conv layer
         if 'Conv' in var1.name.split('/')[-2]:
             # check type of next layer,BatchNorm param first of weight
             if 'BatchNorm' in var2.name.split('/')[-2]:
                 # load batch norm params, It's equal to l.biases,l.scales,l.rolling_mean,l.rolling_variance
                 gamma, beta, mean, var = var_list[i + 1:i + 5]
                 batch_norm_vars = [beta, gamma, mean, var]
                 for var in batch_norm_vars:
                     shape = var.shape.as_list()
                     num_params = np.prod(shape)
                     var_weights = weights[ptr:ptr + num_params].reshape(shape)
                     ptr += num_params
                     print(var,ptr)
                     assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
 
                 # we move the pointer by 4, because we loaded 4 variables
                 i += 4
             elif 'Conv' in var2.name.split('/')[-2]:
                 # load biases,not use the batch norm,So just only load biases
                 bias = var2
                 bias_shape = bias.shape.as_list()
                 bias_params = np.prod(bias_shape)
                 bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                 ptr += bias_params
                 print(bias,ptr)
                 assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                 # we loaded 1 variable
                 i += 1
             
             # we can load weights of conv layer
             shape = var1.shape.as_list()
             num_params = np.prod(shape)
 
             var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
             # remember to transpose to column-major
             var_weights = np.transpose(var_weights, (2, 3, 1, 0))
             ptr += num_params
             print(var1,ptr)
             assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
             i += 1
         elif 'Local' in var1.name.split('/')[-2]:
             # load biases
             bias = var2
             bias_shape = bias.shape.as_list()
             bias_params = np.prod(bias_shape)
             bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
             ptr += bias_params
             print(bias,ptr)
             assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
             i += 1
             
             # we can load weights of conv layer
             shape = var1.shape.as_list()
             num_params = np.prod(shape)
 
             var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
             # remember to transpose to column-major
             var_weights = np.transpose(var_weights, (2, 3, 1, 0))
             ptr += num_params
             print(var1,ptr)
             assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
             i += 1
         elif 'Fc' in var1.name.split('/')[-2]:
             # load biases
             bias = var2
             bias_shape = bias.shape.as_list()
             bias_params = np.prod(bias_shape)
             bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
             ptr += bias_params
             print(bias,ptr)
             assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
             i += 1
             
             # we can load weights of conv layer
             shape = var1.shape.as_list()
             num_params = np.prod(shape)

             var_weights = weights[ptr:ptr + num_params].reshape(shape[1], shape[0])
             # remember to transpose to column-major
             var_weights = np.transpose(var_weights, (1, 0))
             ptr += num_params
             print(var1,ptr)
             assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
             i += 1
             
     return assign_ops
    
 # print all op names
 def _print_tensor_name(self, chkpt_fname):
     reader = pywrap_tensorflow.NewCheckpointReader(chkpt_fname)
     var_to_shape_map = reader.get_variable_to_shape_map()
     print("all tensor name:")
     for key in var_to_shape_map:
         print("tensor_name: ", key )
         #print(reader.get_tensor(key)) # Remove this is you want to print only variable names

 def _load_ckpts(self,weights_file):
  # print
  if self.verbose:
   print("Start to load weights from file:%s" % (weights_file))
   self._print_tensor_name(weights_file)
   
  saver = tf.train.Saver() #initial
  saver.restore(self.sess, weights_file) # saver.restore/saver.save

 def draw_detection(self,img, bboxes, scores, cls_inds, labels, thr=0.3):
  # Generate colors for drawing bounding boxes.
  hsv_tuples = [(x/float(len(labels)), 1., 1.)  for x in range(len(labels))]
  colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
  colors = list(
   map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
  random.seed(10101)  # Fixed seed for consistent colors across runs.
  random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
  random.seed(None)  # Reset seed to default.
  # draw image
  imgcv = np.copy(img)
  h, w, _ = imgcv.shape
  
  for i in range(len(scores)):#ratio convert  
   bboxes[i][0] = int(bboxes[i][0]* (1.0 * w))
   bboxes[i][1] = int(bboxes[i][1]* (1.0 * h))
   bboxes[i][2] = int(bboxes[i][2]* (1.0 * w))
   bboxes[i][3] = int(bboxes[i][3]* (1.0 * h))
   
  bboxes = bboxes.astype(np.int32)    
  
  for i, box in enumerate(bboxes):
   if scores[i] < thr:
    continue
   cls_indx = cls_inds[i]
 
   thick = int((h + w) / 300)
   cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),colors[cls_indx], thick//3)
   mess = '%s: %.3f' % (labels[cls_indx], scores[i])
   if box[1] < 20:
    text_loc = (box[0] + 2, box[1] + 15)
   else:
    text_loc = (box[0], box[1] - 10)
   # cv2.rectangle(imgcv, (box[0], box[1]-20), ((box[0]+box[2])//3+120, box[1]-8), (125, 125, 125), -1)  # puttext the background of the word
   cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (0,0,255), thick//3)
  return imgcv

 ######################## Do detection from a given image by cv2#########################################
 def detect_from_image(self, image):
  """Do detection given a cv image"""
  start_time = time.time()
  boxes, scores, box_classes = self.sess.run([self.boxes,self.scores,self.classes],feed_dict={self.input_images: image})
  duration = time.time() - start_time
  print ('%s: yolo.run(), duration = %.3f' %(datetime.now(), duration))

  return scores, boxes, box_classes
 ##########################################################################################################
 
 
 def detect_from_file(self,image_file,imshow=True,deteted_boxes_file="boxes.txt",
       detected_image_file="detected_image.jpg"): 
  # read image
  image = cv2.imread(image_file)
  image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  
  scores, boxes, box_classes = self.detect_from_image(image_rgb)
  
  img_detection = self.draw_detection(image, boxes, scores, box_classes, self.class_names)
  
  if(deteted_boxes_file):
   f = open(deteted_boxes_file, "w")
   predict_boxes = []
   for i in range(len(scores)):
    # predict data：[class_names,(xmin,ymin,xmax,ymax),scores]
    predict_boxes.append((self.class_names[box_classes[i]], boxes[i, 0],boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i]))
   #f.write(predict_boxes)
   print(predict_boxes)
   
  if detected_image_file:
      cv2.imwrite(os.path.join("out",detected_image_file), img_detection)
      
  cv2.imshow("detection_results", img_detection)
  cv2.waitKey(0)

if __name__ == '__main__':
 #load the darknet yolov2 weight, wget https://pjreddie.com/media/files/yolov2.weights 
 yolov2_tiny_net = YOLOV2_Tiny_Voc(weights_file=FLAGS.darknet_weights_file) 
 images = tf.gfile.Glob(FLAGS.image_dir+"/*.jpg")
 for f in images:
  yolov2_tiny_net.detect_from_file(image_file=f) 
 
 
