# -*- coding: utf-8 -*-
# --------------------------------------
# @File    : decode$.py
# Description :解码darknet19网络得到的参数, using tensorflow
# --------------------------------------

import tensorflow as tf
import numpy as np

def decode(model_output,output_sizes=(13,13),num_class=80,threshold=0.5,iou_threshold=0.5,anchors=None):
	'''
	 model_output:darknet19网络输出的特征图
	 output_sizes:darknet19网络输出的特征图大小，默认是13*13(默认输入416*416，下采样32)
	'''
	H, W = output_sizes
	num_anchors = len(anchors) # 这里的anchor是在configs文件中设置的
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
		box_classes = tf.boolean_mask(box_classes, filter_mask)
		
	
	with tf.variable_scope("NMS"):
		# NMS (不区分不同的类别)
# 		_boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],
# 								   boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]], axis=1)
		#Bounding boxes are supplied as [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners 
		#and the coordinates can be provided as normalized ,(xmin,ymin,xmax,ymax) => (ymin, xmin, ymax, xmax)
		xmin,ymin,xmax,ymax = tf.unstack(tf.transpose(boxes))
		_boxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
		nms_indices = tf.image.non_max_suppression(_boxes, scores,10, iou_threshold)
		

		        	
# 	scores = tf.gather(scores, nms_indices, name="scores")
# 	boxes = tf.gather(boxes, nms_indices,name="boxes")
# 	classes = tf.gather(box_classes, nms_indices,name="classes")
	scores = tf.identity(tf.gather(scores, nms_indices), name="detected_scores")
	boxes = tf.identity(tf.gather(boxes, nms_indices),name="detected_boxes")
	classes = tf.identity(tf.gather(box_classes, nms_indices),name="detected_classes")
	print(boxes,scores,classes)
		

	return boxes,scores,classes#bboxes, obj_probs, class_probs
