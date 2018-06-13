# -*- coding: utf-8 -*-
# --------------------------------------
# @File    : YOLO_V3 
# Description :Yolo V3 by tensorflow
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

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir', 'data', 'Input image foldir name')
tf.app.flags.DEFINE_string('output_img', 'results/detected.jpg', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'data/coco.names', 'File with class names')
tf.app.flags.DEFINE_string('yolov3_ckpt_weight', 'models/yolov3.ckpt', 'File with yolov3.ckpt names')
tf.app.flags.DEFINE_string('darknet_weights_file', 'weight/yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('input_size', 416, 'The input image size for network')

tf.app.flags.DEFINE_float('conf_threshold', 0.4, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')
tf.app.flags.DEFINE_integer('max_output_size', 10, 'the maximum number of boxes to be selected by non max suppression')


tf.app.flags.DEFINE_boolean('load_darknet_weight', True, 'Whether load darknet weights,other load ckpt by Saver.restore')
tf.app.flags.DEFINE_boolean('save_model_weight', True, 'Whether save  darknet weights as tensorflow ckpt')
tf.app.flags.DEFINE_boolean('save_model_pb', True, 'Whether save  model weights as tensorflow pb')

class YOLOV3(object):
	##################### 构造函数：初始化yolo中参数#################################################################
	def __init__(self,weights_file, verbose=False):
		# 后面程序打印描述功能的标志位
		self.verbose = verbose

		# 检测超参数
		self.threshold = FLAGS.conf_threshold 
		self.iou_threshold = FLAGS.iou_threshold 
		self.max_output_size = FLAGS.max_output_size
		self.class_names = self.load_coco_names(FLAGS.class_names)

		
		self.sess = tf.Session()
		self.input_size = FLAGS.input_size
 		self.images = self._input_process(self.input_size)
 		self.boxes, \
 		self.scores, \
 		self.classes = self.yolo_v3(self.images, len(self.class_names), score_threshold=self.threshold, iou_threshold=self.iou_threshold, data_format='NHWC')
		
		if FLAGS.load_darknet_weight:
			self.load_ops = self._load_weights(tf.global_variables(scope='yolov3'), weights_file)
			self.sess.run(self.load_ops)
			if FLAGS.save_model_weight:
				saver = tf.train.Saver()
				saver.save(self.sess,FLAGS.yolov3_ckpt_weight)

		else:
			self._load_ckpts(FLAGS.yolov3_ckpt_weight) # import weight form ckpt
			
			
		if(FLAGS.save_model_pb):
			self._save_graph_to_file(self.sess, self.sess.graph_def ,"models/yolov3_frozen_graph.pb") 
		
		#self.sess.run(tf.global_variables_initializer())
		
		#tensorboard for graph
		writer =tf.summary.FileWriter("logs/",graph = self.sess.graph)
		writer.close()
	####################################################################################################################

	def load_coco_names(self, file_name):
	    names = {}
	    with open(file_name) as f:
	        for id, name in enumerate(f):
	            names[id] = name.strip()
	    return names
	   
	def _save_graph_to_file(self, sess, graph, graph_file_name):
	    output_graph_def = graph_util.convert_variables_to_constants(
	        sess, graph, ["detected_scores","detected_boxes","detected_classes"])
	    with gfile.FastGFile(graph_file_name, 'wb') as f:
	        f.write(output_graph_def.SerializeToString())
	    return
  
	def _input_process(self,image_size):
		 self.input_images = tf.placeholder(tf.uint8, shape=[None, None, 3],name="input")
		 # normalize values to range [0..1]
		 image = tf.to_float(self.input_images)/255.0
		 image = tf.image.resize_images(image, tf.constant([image_size, image_size]))
		 return  tf.expand_dims(image, 0)
		
	def _load_weights(self, var_list, weights_file):
	    """
	    Loads and converts pre-trained weights.
	    :param var_list: list of network variables.
	    :param weights_file: name of the binary file.
	    :return: list of assign ops
	    """
	    with open(weights_file, "rb") as fp:
	        _ = np.fromfile(fp, dtype=np.int32, count=5)# This is verry import for count,it include the version of yolo
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
		if self.verbose:
			print("Start to load weights from file:%s" % (weights_file))
			self._print_tensor_name(weights_file)
			
		saver = tf.train.Saver() #initial
		saver.restore(self.sess, weights_file) # saver.restore/saver.save
    
	def _get_size(self, shape, data_format):
	    if len(shape) == 4:
	        shape = shape[1:]
	    return shape[1:3] if data_format == 'NCHW' else shape[0:2]
	   
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
	
	def _darknet53_block(self, inputs, filters):
	    shortcut = inputs
	    inputs = self._conv2d_fixed_padding(inputs, filters, 1)
	    inputs = self._conv2d_fixed_padding(inputs, filters * 2, 3)
	    
	    inputs = inputs + shortcut
	    return inputs
	
	def _darknet53(self, inputs):
	    """
	    Builds Darknet-53 model.
	    """
	    inputs = self._conv2d_fixed_padding(inputs, 32, 3)
	    inputs = self._conv2d_fixed_padding(inputs, 64, 3, strides=2)
	    
	    inputs = self._darknet53_block(inputs, 32)
	    
	    inputs = self._conv2d_fixed_padding(inputs, 128, 3, strides=2)
	
	    for i in range(2):
	        inputs = self._darknet53_block(inputs, 64)
	
	    inputs = self._conv2d_fixed_padding(inputs, 256, 3, strides=2)
	
	    for i in range(8):
	        inputs = self._darknet53_block(inputs, 128)
	
	    route_1 = inputs#52X52X256
	    inputs = self._conv2d_fixed_padding(inputs, 512, 3, strides=2)
	
	    for i in range(8):
	        inputs = self._darknet53_block(inputs, 256)
	
	    route_2 = inputs#26X26X512
	    inputs = self._conv2d_fixed_padding(inputs, 1024, 3, strides=2)
	
	    for i in range(4):
	        inputs = self._darknet53_block(inputs, 512)#inputs 13X13X1024
	    
	    
	    return route_1, route_2, inputs
	    #route_1    Tensor: Tensor("detector/darknet-53/add_10:0", shape=(?, 52, 52, 256), dtype=float32)    
	    #route_2    Tensor: Tensor("detector/darknet-53/add_18:0", shape=(?, 26, 26, 512), dtype=float32)    
	    #inputs    Tensor: Tensor("detector/darknet-53/add_22:0", shape=(?, 13, 13, 1024), dtype=float32)
	
	def _yolo_block(self, inputs, filters):
	    with tf.name_scope("yolo_block"):
	        inputs = self._conv2d_fixed_padding(inputs, filters, 1)
	        inputs = self._conv2d_fixed_padding(inputs, filters * 2, 3)
	        inputs = self._conv2d_fixed_padding(inputs, filters, 1)
	        inputs = self._conv2d_fixed_padding(inputs, filters * 2, 3)
	        inputs = self._conv2d_fixed_padding(inputs, filters, 1)
	        route = inputs
	        inputs = self._conv2d_fixed_padding(inputs, filters * 2, 3)
	        return route, inputs
	  
	def _upsample(self, inputs, out_shape, data_format='NCHW'):
	    # we need to pad with one pixel, so we set kernel_size = 3
	    inputs = self._fixed_padding(inputs, 3, mode='SYMMETRIC')
	
	    # tf.image.resize_bilinear accepts input in format NHWC
	    if data_format == 'NCHW':
	        inputs = tf.transpose(inputs, [0, 2, 3, 1])
	
	    if data_format == 'NCHW':
	        height = out_shape[3]
	        width = out_shape[2]
	    else:
	        height = out_shape[2]
	        width = out_shape[1]
	
	    # we padded with 1 pixel from each side and upsample by factor of 2, so new dimensions will be
	    # greater by 4 pixels after interpolation
	    new_height = height + 4
	    new_width = width + 4
	
	    inputs = tf.image.resize_bilinear(inputs, (new_height, new_width))
	
	    # trim back to desired size
	    inputs = inputs[:, 2:-2, 2:-2, :]
	
	    # back to NCHW if needed
	    if data_format == 'NCHW':
	        inputs = tf.transpose(inputs, [0, 3, 1, 2])
	
	    inputs = tf.identity(inputs, name='upsampled')
	    return inputs  
	  
	def _ratio_detection_layer(self, inputs, num_classes, anchors, img_size, data_format):
	    with tf.name_scope("detection_layer"):
	        num_anchors = len(anchors)#3
	        predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1, normalizer_fn=None,
	                                  activation_fn=None, biases_initializer=tf.zeros_initializer())
	    
	        shape = predictions.get_shape().as_list()
	        #[None, 13, 13, 255], scale1
	        #[None, 26, 26, 255], scale2
	        #[None, 52, 52, 255], scale3
	
	        grid_size = self._get_size(shape, data_format)
	        #[13, 13]
	        #[26, 26]
	        #[52, 52]
	       
	        with tf.name_scope("anchor_meshgrid"):
	            grid_x = tf.range(grid_size[0], dtype=tf.float32)
	            grid_y = tf.range(grid_size[1], dtype=tf.float32)
	            a, b = tf.meshgrid(grid_x, grid_y)
	        
	            x_offset = tf.reshape(a, (-1, 1))
	            y_offset = tf.reshape(b, (-1, 1))
	        
	            #the offset from the top left corner of the image by x_y_offset, 1 stand for stride
	            x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
	            x_y_offset = tf.cast(x_y_offset, tf.float32)
	            x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])
	    
	        with tf.name_scope("slice"):
	            dim = grid_size[0] * grid_size[1]    #169,676,2704
	            bbox_attrs = 5 + num_classes         #85
	            
	            if data_format == 'NCHW':
	                predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
	                predictions = tf.transpose(predictions, [0, 2, 1])
	
	            predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])
	            #Tensor("detector/yolo-v3/Predict_1/detection_layer/Reshape:0", shape=(?, 507, 85), dtype=float32)
	            #Tensor("detector/yolo-v3/Predict_2/detection_layer/Reshape:0", shape=(?, 2028, 85), dtype=float32)
	            #Tensor("detector/yolo-v3/Predict_3/detection_layer/Reshape:0", shape=(?, 8112, 85), dtype=float32)
	            
	            box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)
	            box_centers = tf.nn.sigmoid(box_centers)
	            confidence = tf.nn.sigmoid(confidence)
	            classes = tf.nn.sigmoid(classes)
	        
	        with tf.name_scope("box_coord"):
	            box_centers = box_centers + x_y_offset
	            # the ratio offset from the top left corner of the image
	            box_centers = box_centers /grid_size
	        
	            stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
	            #(32, 32)
	            #(16, 16)
	            #(8, 8)
	        
	            anchors = [(1.0*a[0] / stride[0], 1.0*a[1] / stride[1]) for a in anchors]
	            #[(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]
	            #[(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)]
	            #[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)]
	            
	            anchors = tf.tile(anchors, [dim, 1])
	            box_sizes = tf.exp(box_sizes) * anchors
	            # the ratio size of the image
	            box_sizes = box_sizes /grid_size
	        
	        #(xcenter, ycenter, w, h, confidence ,classes_probability)
	        predictions = tf.concat([box_centers, box_sizes, confidence,classes], axis=-1)
	        
	        return predictions

	def yolo_v3(self, 
			inputs, 
			num_classes,
			score_threshold=0.5, 
			iou_threshold=0.5, 
			max_output_size=20,
			is_training=False, 
			data_format='NCHW', 
			scope='yolov3',
			reuse=False):
	    """
	    Creates YOLO v3 model.
	
	    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
	        Dimension batch_size may be undefined.
	    :param num_classes: number of predicted classes.
	    :param is_training: whether is training or not.
	    :param data_format: data format NCHW or NHWC.
	    :param reuse: whether or not the network and its variables should be reused.
	    :return:
	    """
	    # it will be needed later on
	    img_size = inputs.get_shape().as_list()[1:3]#(416,416)
	
	    # transpose the inputs to NCHW
	    if data_format == 'NCHW':
	        inputs = tf.transpose(inputs, [0, 3, 1, 2])
	
	    # set batch norm params
	    batch_norm_params = {
	        'decay': _BATCH_NORM_DECAY,
	        'epsilon': _BATCH_NORM_EPSILON,
	        'scale': True,
	        'is_training': is_training,
	        'fused': None,  # Use fused batch norm if possible.
	    }
	
	    # Set activation_fn and parameters for conv2d, batch_norm.
	    with slim.arg_scope([slim.conv2d, slim.batch_norm, self._fixed_padding], data_format=data_format, reuse=reuse):
	        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
	                            biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
	        	with tf.variable_scope(scope):
		            with tf.variable_scope('darknet-53'):
		                route_1, route_2, inputs = self._darknet53(inputs)
		                #route_1    Tensor: Tensor("detector/darknet-53/add_10:0", shape=(?, 52, 52, 256), dtype=float32)    
		                #route_2    Tensor: Tensor("detector/darknet-53/add_18:0", shape=(?, 26, 26, 512), dtype=float32)    
		                #inputs    Tensor: Tensor("detector/darknet-53/add_22:0", shape=(?, 13, 13, 1024), dtype=float32)
		
		            with tf.variable_scope('detector'):
		                with tf.name_scope("Predict_1"):
		                    route, inputs = self._yolo_block(inputs, 512)
		                    detect_1 = self._ratio_detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
		                    detect_1 = tf.identity(detect_1, name='scale_1')#Tensor("detector/yolo-v3/Detection_0/detect_1:0", shape=(?, 507, 85), dtype=float32)
		
		                with tf.name_scope("Predict_2"):
		                    with tf.name_scope("upsample"):
		                        inputs = self._conv2d_fixed_padding(route, 256, 1)
		                        upsample_size = route_2.get_shape().as_list()
		                        inputs = self._upsample(inputs, upsample_size, data_format)
		                        inputs = tf.concat([inputs, route_2], axis=1 if data_format == 'NCHW' else 3)
		    
		                    route, inputs = self._yolo_block(inputs, 256)
		                    detect_2 = self._ratio_detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
		                    detect_2 = tf.identity(detect_2, name='scale_2')#Tensor("detector/yolo-v3/Detection_2/detect_2:0", shape=(?, 2028, 85), dtype=float32)
		                
		                with tf.name_scope("Predict_3"):
		                    with tf.name_scope("upsample"):
		                        inputs = self._conv2d_fixed_padding(route, 128, 1)
		                        upsample_size = route_1.get_shape().as_list()
		                        inputs = self._upsample(inputs, upsample_size, data_format)
		                        inputs = tf.concat([inputs, route_1], axis=1 if data_format == 'NCHW' else 3)
		    
		                    _, inputs = self._yolo_block(inputs, 128)
		                    detect_3 = self._ratio_detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
		                    detect_3 = tf.identity(detect_3, name='scale_3')#Tensor("detector/yolo-v3/Detection_3/detect_3:0", shape=(?, 8112, 85), dtype=float32)
		    
		                #(xcenter, ycenter, w, h, confidence ,classes_probability)
		                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
		                
		                
		                with tf.name_scope("coord_transform"):
		                    center_x, center_y, width, height, confidence, class_prob = tf.split(detections, [1, 1, 1, 1, 1, -1], axis=-1)
		                    w2 = width * 0.5
		                    h2 = height * 0.5       
		                    bboxes = tf.concat([center_x - w2, center_y - h2, center_x + w2, center_y + h2], axis=-1)
		                    
		                with tf.name_scope("select-threhold"):
		                    # score filter
		                    box_scores = confidence * class_prob      # (?, 20)
		                    box_label = tf.to_int32(tf.argmax(box_scores, axis=-1))    # (?, )
		                    box_scores_max = tf.reduce_max(box_scores, axis=-1)     # (?, )
		             
		                    pred_mask = box_scores_max > score_threshold
		                    bboxes = tf.boolean_mask(bboxes, pred_mask)
		                    scores = tf.boolean_mask(box_scores_max, pred_mask)
		                    classes = tf.to_int32(tf.boolean_mask(box_label, pred_mask))
		             
		                with tf.name_scope("NMS"):
		                    # non_max_suppression
		                    xmin,ymin,xmax,ymax = tf.unstack(tf.transpose(bboxes))
		                    _boxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
		                    nms_indices = tf.image.non_max_suppression(_boxes, scores,
		                                                           max_output_size=max_output_size,
		                                                           iou_threshold=iou_threshold)   
	    
	    bboxes = tf.identity(tf.gather(bboxes, nms_indices),name="detected_boxes")
	    scores = tf.identity(tf.gather(scores, nms_indices), name="detected_scores")
	    classes = tf.identity(tf.gather(classes, nms_indices),name="detected_classes")
	    print(bboxes,scores,classes)
	
	    return bboxes,scores,classes

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
		
		if(deteted_boxes_file):
			f = open(deteted_boxes_file, "w")
			predict_boxes = []
			for i in range(len(scores)):
				# predict data：[class_names,(xmin,ymin,xmax,ymax),scores]
				predict_boxes.append((self.class_names[box_classes[i]], boxes[i, 0],boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i]))
			#f.write(predict_boxes)
			print(predict_boxes)
			
		img_detection = self.draw_detection(image, boxes, scores, box_classes, self.class_names)	
		if detected_image_file:
			   cv2.imwrite(os.path.join("out",detected_image_file), img_detection)
			   
		cv2.imshow("detection_results", img_detection)
		cv2.waitKey(0)

if __name__ == '__main__':
	#load the darknet yolov3 weight, wget https://pjreddie.com/media/files/yolov3.weights 
	yolov3_net = YOLOV3(weights_file=FLAGS.darknet_weights_file) 
	images = tf.gfile.Glob(FLAGS.image_dir+"/*.jpg")
	for f in images:
		yolov3_net.detect_from_file(image_file=f) 
	
	
