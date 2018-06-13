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
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir', 'data', 'Input image foldir name')
tf.app.flags.DEFINE_string('output_img', 'results/detected.jpg', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'data/coco.names', 'File with class names')
tf.app.flags.DEFINE_string('yolov1_ckpt_weight', 'models/yolov1-tiny.ckpt', 'File with yolov2.ckpt names')
tf.app.flags.DEFINE_string('darknet_weights_file', 'weight/tiny-yolov1.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('input_size', 448, 'The input image size for network')

tf.app.flags.DEFINE_float('conf_threshold', 0.1, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.6, 'IoU threshold')
tf.app.flags.DEFINE_integer('max_output_size', 10, 'the maximum number of boxes to be selected by non max suppression')

#
tf.app.flags.DEFINE_boolean('load_darknet_weight', True, 'Whether load darknet weights,other load ckpt by Saver.restore')
tf.app.flags.DEFINE_boolean('save_model_weight', True, 'Whether save  darknet weights as tensorflow ckpt')
tf.app.flags.DEFINE_boolean('save_model_pb', True, 'Whether save  model weights as tensorflow pb')

# leaky_relu激活函数
def leaky_relu(x,alpha=0.1):
	return tf.maximum(alpha*x,x)

def activation_leaky_relu(alpha):
     with tf.variable_scope("leaky_relu"):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
        return op

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

class YOLOV1_Tiny(object):
	##################### 构造函数：初始化yolo中S、B、C参数#################################################################
	def __init__(self,weights_file, verbose=True):
		# 后面程序打印描述功能的标志位
		self.verbose = verbose

		# 检测超参数
		self.S = 7 # cell数量 448/2^6
		self.B = 2 # 每个网格的边界框数
		self.classes_names = ["aeroplane", "bicycle", "bird", "boat", "bottle",
							"bus", "car", "cat", "chair", "cow", "diningtable",
							"dog", "horse", "motorbike", "person", "pottedplant",
							"sheep", "sofa", "train","tvmonitor"]
		self.C = len(self.classes_names) # 类别数
		# 边界框的中心坐标xy——相对于每个cell左上点的偏移量
		self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)] * self.S * self.B),
												[self.B, self.S, self.S]), [1, 2, 0])
		self.y_offset = np.transpose(self.x_offset, [1, 0, 2])
		self.threshold = FLAGS.conf_threshold # 类别置信度分数阈值
		self.iou_threshold = FLAGS.iou_threshold # IOU阈值，小于0.4的会过滤掉
		self.max_output_size = 10 # NMS选择的边界框的最大数量

		self.sess = tf.Session()
 		self.images = self._input_process(FLAGS.input_size)
		self.predicts = self._build_network(self.images,num_outputs=self.S*self.S*(self.B*5+self.C))# 【1】搭建网络模型(预测):模型的主体网络部分，这个网络将输出[batch,7*7*30]的张量
		self.boxes, \
  		self.scores, \
    	self.classes = self._build_detector(self.predicts) # 【2】解析网络的预测结果：先判断预测框类别，再NMS

		if FLAGS.load_darknet_weight:
		   	self.load_ops = self._load_weights(tf.global_variables(scope='yolov1_tiny'), weights_file)
		   	self.sess.run(self.load_ops)
		   	if FLAGS.save_model_weight:
		   		saver = tf.train.Saver()
		    	saver.save(self.sess,FLAGS.yolov1_ckpt_weight)
		
		else:
		   	self._load_ckpts(FLAGS.yolov1_ckpt_weight) # import weight form ckpt
   
		if(FLAGS.save_model_pb):
			self._save_graph_to_file(self.sess, self.sess.graph_def ,"models/yolov1_tiny_frozen_graph.pb") 

		writer =tf.summary.FileWriter("logs/",graph = self.sess.graph)
		writer.close()
	####################################################################################################################

	def _load_weights(self, var_list, weights_file):
	    """
	    Loads and converts pre-trained weights.
	    :param var_list: list of network variables.
	    :param weights_file: name of the binary file.
	    :return: list of assign ops
	    """
	    with open(weights_file, "rb") as fp:
	        _ = np.fromfile(fp, dtype=np.int32, count=4)
	
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
	            # check type of next layer
	            if 'BatchNorm' in var2.name.split('/')[-2]:
	                # load batch norm params
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
	                # load biases
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
	            # load biases first
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
# 	            var_weights = weights[ptr:ptr + num_params].reshape(shape)

  	            var_weights = weights[ptr:ptr + num_params].reshape(shape[1], shape[0])
  	            # remember to transpose to column-major
  	            var_weights = np.transpose(var_weights, (1, 0))
	            ptr += num_params
	            print(var1,ptr)
	            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
	            i += 1
	    return assign_ops

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
    	
	def _print_activations(self, t):
		print(t.op.name, ' ', t.get_shape().as_list())

	def _build_network(self,
					  images,
                      num_outputs=1470,#self.S*self.S*(self.B*5+self.C)
                      alpha=0.1,
                      keep_prob=0.5,
                      is_training=False,
                      data_format='NHWC',
                      reuse=False,
                      scope='yolov1_tiny'):
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
	                            biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=alpha)):
			    with tf.variable_scope(scope):
						net = self._conv2d_fixed_padding(images, 16, 3)
						self._print_activations(net)
						net = slim.max_pool2d(net, 2, scope='pool_1')
						self._print_activations(net)
						
						
						net = self._conv2d_fixed_padding(net, 32, 3)  
						self._print_activations(net)
						net = slim.max_pool2d(net, 2, scope='pool_2')
						self._print_activations(net)
						
						net = self._conv2d_fixed_padding(net, 64, 3)  
						self._print_activations(net)
						net = slim.max_pool2d(net, 2, scope='pool_3')
						self._print_activations(net)
						
						net = self._conv2d_fixed_padding(net, 128, 3)  
						self._print_activations(net)
						net = slim.max_pool2d(net, 2, scope='pool_4')
						self._print_activations(net)
						
						net = self._conv2d_fixed_padding(net, 256, 3)  
						self._print_activations(net)
						net = slim.max_pool2d(net, 2, scope='pool_5')
						self._print_activations(net)
						
						net = self._conv2d_fixed_padding(net, 512, 3)  
						self._print_activations(net)
						net = slim.max_pool2d(net, 2, scope='pool_6')
						self._print_activations(net)
						
						net = self._conv2d_fixed_padding(net, 1024, 3)  
						self._print_activations(net)
						net = self._conv2d_fixed_padding(net, 256, 3)  
						self._print_activations(net)
						
						net = tf.transpose(net, [0, 3, 1, 2], name='trans')   #can't be removed
						self._print_activations(net)
						net = slim.flatten(net, scope='flat') 
						self._print_activations(net)

						predicts = slim.fully_connected(net, num_outputs, activation_fn=None, scope='Fc')
						self._print_activations(predicts)
						return predicts


	# 【2】解析网络的预测结果：先判断预测框类别，再NMS
	def _build_detector(self, model_output):
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
				class_probs = tf.reshape(model_output[0,:idx1],[self.S,self.S,self.C])
				# 2.置信度[:,7*7*20:7*7*(20+2)]  2维
				confs = tf.reshape(model_output[0,idx1:idx2],[self.S,self.S,self.B])
				# 3.边界框[:,7*7*(20+2):]  8维 -> (x,y,w,h)
				boxes = tf.reshape(model_output[0,idx2:],[self.S,self.S,self.B,4])
		
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
		
		scores = tf.identity(tf.gather(scores, nms_indices), name="detected_scores")
		boxes = tf.identity(tf.gather(boxes, nms_indices),name="detected_boxes")#(xcenter,ycenter,w,h)
		classes = tf.identity(tf.gather(box_classes, nms_indices),name="detected_classes")
		print(boxes,scores,classes)
   
		return boxes,scores,classes

	# print all op names
	def _print_tensor_name(self, chkpt_fname):
	    reader = pywrap_tensorflow.NewCheckpointReader(chkpt_fname)
	    var_to_shape_map = reader.get_variable_to_shape_map()
	    for key in var_to_shape_map:
	        print("tensor_name: ", key )
	        #print(reader.get_tensor(key)) # Remove this is you want to print only variable names

	# 【3】导入权重文件
	def _load_ckpts(self,weights_file):
		# 打印状态信息
		if self.verbose:
			print("Start to load weights from file:%s" % (weights_file))
			self._print_tensor_name(weights_file)

		# 导入权重
		saver = tf.train.Saver() # 初始化
		saver.restore(self.sess,weights_file) # saver.restore导入/saver.save保存

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

	# 【4】从预测输入图片，并可视化检测边界框、将obj的分类结果和坐标保存成txt。
	# image_file是输入图片文件路径；
	# deteted_boxes_file="boxes.txt"是最后坐标txt；detected_image_file="detected_image.jpg"是检测结果可视化图片
	def detect_from_file(self,image_file,imshow=True,deteted_boxes_file="boxes.txt",
						 detected_image_file="detected_image.jpg"): 
		# read image
		image = cv2.imread(image_file)
		img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		scores, boxes, box_classes = self.detect_from_image(img_RGB)
		predict_boxes = []
		for i in range(len(scores)):
			# 预测框数据为：[概率,x,y,w,h,类别置信度]
			predict_boxes.append((self.classes_names[box_classes[i]], boxes[i, 0],
								  boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i]))
		self.show_results(image, predict_boxes, imshow, deteted_boxes_file, detected_image_file)

	######################## 对应【4】:可视化检测边界框、将obj的分类结果和坐标保存成txt#########################################
	def detect_from_image(self, image):
		"""Do detection given a cv image"""
		img_h, img_w, _ = image.shape
		start_time = time.time()
		scores, boxes, box_classes = self.sess.run([self.scores, self.boxes, self.classes],
												   feed_dict={self.input_images: image})
		duration = time.time() - start_time
		print ('%s: yolo.run(), duration = %.3f' %(datetime.now(), duration))
        

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
			cv2.imshow('YOLO_v1_tiny detection', img_cp)
			cv2.waitKey(0)
		if detected_image_file:
			cv2.imwrite(detected_image_file, img_cp)
		if deteted_boxes_file:
			f.close()
	####################################################################################################################

if __name__ == '__main__':
	yolov1_tiny_net = YOLOV1_Tiny(weights_file=FLAGS.darknet_weights_file)
	images = tf.gfile.Glob(FLAGS.image_dir+"/*.jpg")
	for f in images:
		yolov1_tiny_net.detect_from_file(image_file=f) # 【4】从预测输入图片，并可视化检测边界框、将obj的分类结果和坐标保存成txt。
	
	
