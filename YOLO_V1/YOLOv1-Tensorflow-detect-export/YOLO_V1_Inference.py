# -*- coding: utf-8 -*-
# --------------------------------------
# @File    : yolo1_tf$.py
# Description :Yolo V1 by tensorflow。yolo1的预测过程。
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import time
from datetime import datetime
from tensorflow.python import pywrap_tensorflow


slim = tf.contrib.slim

# leaky_relu激活函数
def leaky_relu(x,alpha=0.1):
	return tf.maximum(alpha*x,x)

def activation_leaky_relu(alpha):
     with tf.variable_scope("leaky_relu"):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
        return op

class Yolo(object):
	##################### 构造函数：初始化yolo中S、B、C参数#################################################################
	def __init__(self,weights_file,input_image=None,verbose=True):
		# 后面程序打印描述功能的标志位
		self.verbose = verbose

		# 检测超参数
		self.S = 7 # cell数量
		self.B = 2 # 每个网格的边界框数
		self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
						"bus", "car", "cat", "chair", "cow", "diningtable",
						"dog", "horse", "motorbike", "person", "pottedplant",
						"sheep", "sofa", "train","tvmonitor"]
		self.C = len(self.classes) # 类别数

		# 边界框的中心坐标xy——相对于每个cell左上点的偏移量
		self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)] * self.S * self.B),
												[self.B, self.S, self.S]), [1, 2, 0])
		self.y_offset = np.transpose(self.x_offset, [1, 0, 2])

		self.threshold = 0.2 # 类别置信度分数阈值
		self.iou_threshold = 0.4 # IOU阈值，小于0.4的会过滤掉

		self.max_output_size = 10 # NMS选择的边界框的最大数量

		self.sess = tf.Session()
 		#self._build_net() # 【1】搭建网络模型(预测):模型的主体网络部分，这个网络将输出[batch,7*7*30]的张量
 		self.images = self._input_process(448)
		self._build_network(self.images)
        
		self._build_detector() # 【2】解析网络的预测结果：先判断预测框类别，再NMS
	
		self._load_weights(weights_file) # 【3】导入权重文件
		writer =tf.summary.FileWriter("logs/",graph = self.sess.graph)
		writer.close()
		#self.detect_from_file(image_file=input_image) # 【4】从预测输入图片，并可视化检测边界框、将obj的分类结果和坐标保存成txt。
	####################################################################################################################
	def _input_process(self,image_size):
		 self.input_images = tf.placeholder(tf.uint8, shape=[None, None, 3],name="input")
		 image = (tf.to_float(self.input_images) / 255.0)*2-1 
		 image = tf.image.resize_images(image, tf.constant([image_size, image_size]))
		 return  tf.expand_dims(image, 0)
    	
    	
	# 【1】搭建网络模型(预测):模型的主体网络部分，这个网络将输出[batch,7*7*30]的张量
	def _build_net(self):
		# 打印状态信息
		if self.verbose:
			print("Start to build the network ...")

		# 输入、输出用占位符，因为尺寸一般不会改变
		self.images = tf.placeholder(tf.float32,[None,448,448,3]) # None表示不确定，为了自适应batchsize

		# 搭建网络模型
		net = self._conv_layer(self.images, 1, 64, 7, 2)
		net = self._maxpool_layer(net, 1, 2, 2)
		net = self._conv_layer(net, 2, 192, 3, 1)
		net = self._maxpool_layer(net, 2, 2, 2)
		net = self._conv_layer(net, 3, 128, 1, 1)
		net = self._conv_layer(net, 4, 256, 3, 1)
		net = self._conv_layer(net, 5, 256, 1, 1)
		net = self._conv_layer(net, 6, 512, 3, 1)
		net = self._maxpool_layer(net, 6, 2, 2)
		net = self._conv_layer(net, 7, 256, 1, 1)
		net = self._conv_layer(net, 8, 512, 3, 1)
		net = self._conv_layer(net, 9, 256, 1, 1)
		net = self._conv_layer(net, 10, 512, 3, 1)
		net = self._conv_layer(net, 11, 256, 1, 1)
		net = self._conv_layer(net, 12, 512, 3, 1)
		net = self._conv_layer(net, 13, 256, 1, 1)
		net = self._conv_layer(net, 14, 512, 3, 1)
		net = self._conv_layer(net, 15, 512, 1, 1)
		net = self._conv_layer(net, 16, 1024, 3, 1)
		net = self._maxpool_layer(net, 16, 2, 2)
		net = self._conv_layer(net, 17, 512, 1, 1)
		net = self._conv_layer(net, 18, 1024, 3, 1)
		net = self._conv_layer(net, 19, 512, 1, 1)
		net = self._conv_layer(net, 20, 1024, 3, 1)
		net = self._conv_layer(net, 21, 1024, 3, 1)
		net = self._conv_layer(net, 22, 1024, 3, 2)
		net = self._conv_layer(net, 23, 1024, 3, 1)
		net = self._conv_layer(net, 24, 1024, 3, 1)
		net = self._flatten(net)
		net = self._fc_layer(net, 25, 512, activation=leaky_relu)
		net = self._fc_layer(net, 26, 4096, activation=leaky_relu)
		net = self._fc_layer(net, 27, self.S*self.S*(self.B*5+self.C))

		# 网络输出，[batch,7*7*30]的张量
		self.predicts = net
		
		
	def print_activations(self, t):
		print(t.op.name, ' ', t.get_shape().as_list())

	def _build_network(self,
					  images,
                      num_outputs=1470,
                      alpha=0.1,
                      keep_prob=0.5,
                      is_training=False,
                      scope='yolo'):
		with tf.variable_scope(scope):
			with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=activation_leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
						net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')#[None, 448, 448, 3]  => [None, 454, 454, 3]
						self.print_activations(net)
						net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')    #[None, 448, 448, 3]  => [None, 224, 224, 64]
						self.print_activations(net)
						net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')        #[None, 224, 224, 64]  => [None, 112, 112, 64]
						self.print_activations(net)
						net = slim.conv2d(net, 192, 3, scope='conv_4')    #[None, 112, 112, 64]  => [None, 112, 112, 192]
						self.print_activations(net)
						net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')    #[None, 112, 112, 192]  => [None, 56, 56, 192]
						self.print_activations(net)
						net = slim.conv2d(net, 128, 1, scope='conv_6')    #[None, 56, 56, 192]  => [None, 56, 56, 128]
						self.print_activations(net)
						net = slim.conv2d(net, 256, 3, scope='conv_7')    #[None, 56, 56, 192]  => [None, 56, 56, 256]
						self.print_activations(net)
						net = slim.conv2d(net, 256, 1, scope='conv_8')    #[None, 56, 56, 256]  => [None, 56, 56, 256]
						self.print_activations(net)
						net = slim.conv2d(net, 512, 3, scope='conv_9')    #[None, 448, 448, 3]  => [None, 56, 56, 512]
						self.print_activations(net)
						net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')    #[None, 56, 56, 512]  => [None, 28, 28, 512]
						self.print_activations(net)
						net = slim.conv2d(net, 256, 1, scope='conv_11')    #[None, 28, 28, 512]  => [None, 28, 28, 256]
						self.print_activations(net)
						net = slim.conv2d(net, 512, 3, scope='conv_12')    #[None, 28, 28, 256]  => [None, 28, 28, 512]
						self.print_activations(net)
						net = slim.conv2d(net, 256, 1, scope='conv_13')    #[None, 28, 28, 512]  => [None, 28, 28, 256]
						self.print_activations(net)
						net = slim.conv2d(net, 512, 3, scope='conv_14')    #[None, 28, 28, 256]  => [None, 28, 28, 512]
						self.print_activations(net)
						net = slim.conv2d(net, 256, 1, scope='conv_15')    #[None, 28, 28, 512]  => [None, 28, 28, 256]
						self.print_activations(net)
						net = slim.conv2d(net, 512, 3, scope='conv_16')    #[None, 28, 28, 256]  => [None, 28, 28, 512]
						self.print_activations(net)
						net = slim.conv2d(net, 256, 1, scope='conv_17')    #[None, 28, 28, 512]  => [None, 28, 28, 256]
						self.print_activations(net)
						net = slim.conv2d(net, 512, 3, scope='conv_18')    #[None, 28, 28, 256]  => [None, 28, 28, 512]
						self.print_activations(net)
						net = slim.conv2d(net, 512, 1, scope='conv_19')    #[None, 28, 28, 512]  => [None, 28, 28, 512]
						self.print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv_20')    #[None, 28, 28, 512]  => [None, 28, 28, 1024]
						self.print_activations(net)
						net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')    #[None, 28, 28, 1024]  => [None, 14, 14, 1024]
						self.print_activations(net)
						net = slim.conv2d(net, 512, 1, scope='conv_22')    #[None, 14, 14, 1024]  => [None, 14, 14, 512]
						self.print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv_23')    #[None, 14, 14, 512]  => [None, 14, 14, 1024]
						self.print_activations(net)
						net = slim.conv2d(net, 512, 1, scope='conv_24')    #[None, 14, 14, 1024]  => [None, 14, 14, 512]
						self.print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv_25')    #[None, 14, 14, 512]  => [None, 14, 14, 1024]
						self.print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv_26')    #[None, 14, 14, 1024]  => [None, 14, 14, 1024]
						self.print_activations(net)
						net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')    #[None, 14, 14, 1024]  => [None, 16, 16, 1024]
						self.print_activations(net)
						net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')    #[None, 16, 16, 1024]  => [None, 7, 7, 1024]
						self.print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv_29')    #[None, 7, 7, 1024]  => [None, 7, 7, 1024]
						self.print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv_30')    #[None, 7, 7, 1024]  => [None, 7, 7, 1024]
						self.print_activations(net)
						net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')    #[None, 7, 7, 1024]  => [None, 1024, 7, 7]
						self.print_activations(net)
						net = slim.flatten(net, scope='flat_32')    #[None, 1024, 7, 7]  => [None, 50176]
						self.print_activations(net)
						net = slim.fully_connected(net, 512, scope='fc_33')    #[None, 50176]  => [None, 512]
						self.print_activations(net)
						net = slim.fully_connected(net, 4096, scope='fc_34')    #[None, 512]  => [None, 4096]
						self.print_activations(net)
						net = slim.dropout(net, keep_prob=keep_prob,
										   is_training=is_training, scope='dropout_35')    #[None, 4096]  => [None, 4096]
						self.print_activations(net)
						net = slim.fully_connected(net, num_outputs,
												   activation_fn=None, scope='fc_36')    #[None, 4096]  => [None, 1470]
						self.print_activations(net)
						self.predicts = net

	# 【2】解析网络的预测结果：先判断预测框类别，再NMS
	def _build_detector(self):
		with tf.variable_scope("postprocess"):
			# 原始图片的宽和高
			#_, h, w, _ = tf.unstack(tf.shape(self.images))
			#height = tf.to_float(h)#input image height of the network 448
			#width = tf.to_float(w)#input image width of the network 448
			with tf.variable_scope("slice"):
				# 网络回归[batch,7*7*30]：
				idx1 = self.S*self.S*self.C
				idx2 = idx1 + self.S*self.S*self.B
				# 1.类别概率[:,:7*7*20]  20维
				class_probs = tf.reshape(self.predicts[0,:idx1],[self.S,self.S,self.C])
				# 2.置信度[:,7*7*20:7*7*(20+2)]  2维
				confs = tf.reshape(self.predicts[0,idx1:idx2],[self.S,self.S,self.B])
				# 3.边界框[:,7*7*(20+2):]  8维 -> (x,y,w,h)
				boxes = tf.reshape(self.predicts[0,idx2:],[self.S,self.S,self.B,4])
		
			with tf.variable_scope("coord_decode"):
				# x，y预测的是相对 grid cell 左上角的坐标offset,So将x，y转换为相对于图像左上角的坐标
				#boxes[:, :, :, 0],boxes[:, :, :, 1],the bounding box x and y coordinates to be offsets of a particular grid cell location
				# w，h的预测是宽度和高度的平方根,So平方乘以图像的宽度和高度,the bounding box width and height by the image width and height
				#normallize the box coordinates to  0-1 of the whole image
				boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(self.x_offset, dtype=tf.float32)) / self.S ,#self.width,
						  (boxes[:, :, :, 1] + tf.constant(self.y_offset, dtype=tf.float32)) / self.S ,#self.height,
						  tf.square(boxes[:, :, :, 2]),#self.width,
						  tf.square(boxes[:, :, :, 3])], axis=3)#self.height], axis=3)
		
			with tf.variable_scope("class-specific"):
				# 类别置信度分数：[S,S,B,1]*[S,S,1,C]=[S,S,B,类别置信度C]
				scores = tf.expand_dims(confs, -1) * tf.expand_dims(class_probs, 2)
		
				scores = tf.reshape(scores, [-1, self.C])  # [S*S*B, C]
				boxes = tf.reshape(boxes, [-1, 4])  # [S*S*B, 4]
		
				# 只选择类别置信度最大的值作为box的类别、分数
				box_classes = tf.argmax(scores, axis=1) # 边界框box的类别
				box_class_scores = tf.reduce_max(scores, axis=1) # 边界框box的分数
		
			with tf.variable_scope("select-threshold"):
				# 利用类别置信度阈值self.threshold，过滤掉类别置信度低的
				filter_mask = box_class_scores >= self.threshold
				scores = tf.boolean_mask(box_class_scores, filter_mask)
				boxes = tf.boolean_mask(boxes, filter_mask)
				box_classes = tf.boolean_mask(box_classes, filter_mask)
		
			with tf.variable_scope("NMS"):
				# NMS (不区分不同的类别)
				# 中心坐标+宽高box (x, y, w, h) -> xmin=x-w/2 -> 左上+右下box (ymin, xmin, ymax, xmax)，因为NMS函数是这种计算方式
				_boxes = tf.stack([boxes[:, 1] - 0.5 * boxes[:, 2], boxes[:, 0] - 0.5 * boxes[:, 3],
								       boxes[:, 1] + 0.5 * boxes[:, 2], boxes[:, 0] + 0.5 * boxes[:, 3]], axis=1)
				#Bounding boxes are supplied as [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners 
				#and the coordinates can be provided as normalized ,ymin, xmin, ymax, xmax
				nms_indices = tf.image.non_max_suppression(_boxes, scores,
																self.max_output_size, self.iou_threshold)
		        	
		self.scores = tf.gather(scores, nms_indices, name="scores")
		self.boxes = tf.gather(boxes, nms_indices,name="boxes")
		self.box_classes = tf.gather(box_classes, nms_indices,name="classes")

	# print all op names
	def _print_tensor_name(self, chkpt_fname):
	    reader = pywrap_tensorflow.NewCheckpointReader(chkpt_fname)
	    var_to_shape_map = reader.get_variable_to_shape_map()
	    for key in var_to_shape_map:
	        print("tensor_name: ", key )
	        #print(reader.get_tensor(key)) # Remove this is you want to print only variable names

	# 【3】导入权重文件
	def _load_weights(self,weights_file):
		# 打印状态信息
		if self.verbose:
			print("Start to load weights from file:%s" % (weights_file))
			self._print_tensor_name(weights_file)

		# 导入权重
		saver = tf.train.Saver() # 初始化
		saver.restore(self.sess,weights_file) # saver.restore导入/saver.save保存

	# 【4】从预测输入图片，并可视化检测边界框、将obj的分类结果和坐标保存成txt。
	# image_file是输入图片文件路径；
	# deteted_boxes_file="boxes.txt"是最后坐标txt；detected_image_file="detected_image.jpg"是检测结果可视化图片
	def detect_from_file(self,image_file,imshow=True,deteted_boxes_file="boxes.txt",
						 detected_image_file="detected_image.jpg"): 
		# read image
		image = cv2.imread(image_file)
		img_h, img_w, _ = image.shape
		scores, boxes, box_classes = self._detect_from_image(image)
		predict_boxes = []
		for i in range(len(scores)):
			# 预测框数据为：[概率,x,y,w,h,类别置信度]
			predict_boxes.append((self.classes[box_classes[i]], boxes[i, 0],
								  boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i]))
		self.show_results(image, predict_boxes, imshow, deteted_boxes_file, detected_image_file)


	################# 对应【1】:定义conv/maxpool/flatten/fc层#############################################################
	# 卷积层：x输入；id：层数索引；num_filters：卷积核个数；filter_size：卷积核尺寸；stride：步长
	def _conv_layer(self,x,id,num_filters,filter_size,stride):

		# 通道数
		in_channels = x.get_shape().as_list()[-1]
		# 均值为0标准差为0.1的正态分布，初始化权重w；shape=行*列*通道数*卷积核个数
		weight = tf.Variable(tf.truncated_normal([filter_size,filter_size,in_channels,num_filters],mean=0.0,stddev=0.1))
		bias = tf.Variable(tf.zeros([num_filters,])) # 列向量

		# padding, 注意: 不用padding="SAME",否则可能会导致坐标计算错误
		pad_size = filter_size // 2
		pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
		x_pad = tf.pad(x, pad_mat)
		conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
		output = leaky_relu(tf.nn.bias_add(conv, bias))

		# 打印该层信息
		if self.verbose:
			print('Layer%d:type=conv,num_filter=%d,filter_size=%d,stride=%d,output_shape=%s'
					%(id,num_filters,filter_size,stride,str(output.get_shape())))

		return output

	# 池化层：x输入；id：层数索引；pool_size：池化尺寸；stride：步长
	def _maxpool_layer(self,x,id,pool_size,stride):
		output = tf.layers.max_pooling2d(inputs=x,
										 pool_size=pool_size,
										 strides=stride,
										 padding='SAME')
		if self.verbose:
			print('Layer%d:type=MaxPool,pool_size=%d,stride=%d,out_shape=%s'
			%(id,pool_size,stride,str(output.get_shape())))
		return output

	# 扁平层：因为接下来会连接全连接层，例如[n_samples, 7, 7, 32] -> [n_samples, 7*7*32]
	def _flatten(self,x):
		tran_x = tf.transpose(x,[0,3,1,2]) # [batch,行,列,通道数channels] -> [batch,通道数channels,列,行]
		nums = np.product(x.get_shape().as_list()[1:]) # 计算的是总共的神经元数量，第一个表示batch数量所以去掉
		return tf.reshape(tran_x,[-1,nums]) # [batch,通道数channels,列,行] -> [batch,通道数channels*列*行],-1代表自适应batch数量

	# 全连接层：x输入；id：层数索引；num_out：输出尺寸；activation：激活函数
	def _fc_layer(self,x,id,num_out,activation=None):
		num_in = x.get_shape().as_list()[-1] # 通道数/维度
		# 均值为0标准差为0.1的正态分布，初始化权重w；shape=行*列*通道数*卷积核个数
		weight = tf.Variable(tf.truncated_normal(shape=[num_in,num_out],mean=0.0,stddev=0.1))
		bias = tf.Variable(tf.zeros(shape=[num_out,])) # 列向量
		output = tf.nn.xw_plus_b(x,weight,bias)

		# 正常全连接层是leak_relu激活函数；但是最后一层是liner函数
		if activation:
			output = activation(output)

		# 打印该层信息
		if self.verbose:
			print('Layer%d:type=Fc,num_out=%d,output_shape=%s'  
				  % (id, num_out, str(output.get_shape())))
		return output
	####################################################################################################################


	######################## 对应【4】:可视化检测边界框、将obj的分类结果和坐标保存成txt#########################################
	def _detect_from_image(self, image):
		"""Do detection given a cv image"""
		img_h, img_w, _ = image.shape
		print(image.shape)
# 		img_resized = cv2.resize(image, (448, 448))
# 		img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
# 		img_resized_np = np.asarray(img_RGB)
# 		_images = np.zeros((1, 448, 448, 3), dtype=np.float32)
# 		_images[0] = (img_resized_np / 255.0) * 2.0 - 1.0
		start_time = time.time()
		scores, boxes, box_classes = self.sess.run([self.scores, self.boxes, self.box_classes],
												   feed_dict={self.input_images: image})
		duration = time.time() - start_time
		print ('%s: yolo.run(), duration = %.3f' %(datetime.now(), duration))
        
		print(boxes)
		for i in range(len(scores)):#ratio convert  
			boxes[i][0] *= (1.0 * img_w)
			boxes[i][1] *= (1.0 * img_h)
			boxes[i][2] *= (1.0 * img_w)
			boxes[i][3] *= (1.0 * img_h)
		
		return scores, boxes, box_classes

	def show_results(self, image, results, imshow=True, deteted_boxes_file=None,
					 detected_image_file=None):
		"""Show the detection boxes"""
		img_cp = image.copy()
		if deteted_boxes_file:
			f = open(deteted_boxes_file, "w")
		#  draw boxes
		for i in range(len(results)):
			x = int(results[i][1])
			y = int(results[i][2])
			w = int(results[i][3]) // 2
			h = int(results[i][4]) // 2
			if self.verbose:
				print("class: %s, [x, y, w, h]=[%d, %d, %d, %d], confidence=%f"
					  % (results[i][0],x, y, w, h, results[i][-1]))

				# 中心坐标 + 宽高box(x, y, w, h) -> xmin = x - w / 2 -> 左上 + 右下box(xmin, ymin, xmax, ymax)
				cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)

				# 在边界框上显示类别、分数(类别置信度)
				cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1) # puttext函数的背景
				cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

			if deteted_boxes_file:
				# 保存obj检测结果为txt文件
				f.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' +
						str(w) + ',' + str(h) + ',' + str(results[i][5]) + '\n')
		if imshow:
			cv2.imshow('YOLO_small detection', img_cp)
			cv2.waitKey(0)
		if detected_image_file:
			cv2.imwrite(detected_image_file, img_cp)
		if deteted_boxes_file:
			f.close()
	####################################################################################################################

if __name__ == '__main__':
	yolo_net = Yolo(weights_file='pretrained/YOLO_small.ckpt')
	images = tf.gfile.Glob("image/*.jpg")
	for f in images:
		yolo_net.detect_from_file(image_file=f) # 【4】从预测输入图片，并可视化检测边界框、将obj的分类结果和坐标保存成txt。
	
	
