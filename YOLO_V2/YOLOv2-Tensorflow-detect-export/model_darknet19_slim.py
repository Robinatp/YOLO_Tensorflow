# -*- coding: utf-8 -*-
# --------------------------------------
# Description :yolo2网络模型——darknet19.
# --------------------------------------

import os
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow


slim = tf.contrib.slim

################# 基础层：conv/pool/reorg(带passthrough的重组层) #############################################
# 激活函数
def leaky_relu(x):
	return tf.nn.leaky_relu(x,alpha=0.1,name='leaky_relu') # 或者tf.maximum(0.1*x,x)

# Conv+BN：yolo2中每个卷积层后面都有一个BN层
def conv2d(x,filters_num,filters_size,pad_size=0,stride=1,batch_normalize=True,
		   activation=leaky_relu,use_bias=False,name='conv2d'):
	# padding，注意: 不用padding="SAME",否则可能会导致坐标计算错误
	if pad_size > 0:
		x = tf.pad(x,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
	# 有BN层，所以后面有BN层的conv就不用偏置bias，并先不经过激活函数activation
	out = tf.layers.conv2d(x,filters=filters_num,kernel_size=filters_size,strides=stride,
						   padding='VALID',activation=None,use_bias=use_bias,name=name)
	#print_activations(out)
	# BN，如果有，应该在卷积层conv和激活函数activation之间
	if batch_normalize:
		out = tf.layers.batch_normalization(out,axis=-1,momentum=0.9,training=False,name=name+'_bn')
	if activation:
		out = activation(out)
	#print_activations(out)
	return out

# max_pool
def maxpool(x,size=2,stride=2,name='maxpool'):
	out = tf.layers.max_pooling2d(x,pool_size=size,strides=stride)
	#print_activations(out)
	return out

# reorg layer(带passthrough的重组层)
def reorg(x,stride):
	return tf.space_to_depth(x,block_size=stride)
	# 或者return tf.extract_image_patches(x,ksizes=[1,stride,stride,1],strides=[1,stride,stride,1],
	# 								rates=[1,1,1,1],padding='VALID')
	
	
	
# print all op names
def print_tensor_name(chkpt_fname):
	reader = pywrap_tensorflow.NewCheckpointReader(chkpt_fname)
	var_to_shape_map = reader.get_variable_to_shape_map()
	for key in var_to_shape_map:
	    print("tensor_name: ", key )
	    #print(reader.get_tensor(key)) # Remove this is you want to print only variable names	
	
#########################################################################################################

################################### Darknet19 ###########################################################
# 默认是coco数据集，最后一层维度是anchor_num*(class_num+5)=5*(80+5)=425
def darknet(images,n_last_channels=425):
	net = conv2d(images, filters_num=32, filters_size=3, pad_size=1, name='conv1')
	net = maxpool(net, size=2, stride=2, name='pool1')

	net = conv2d(net, 64, 3, 1, name='conv2')
	net = maxpool(net, 2, 2, name='pool2')

	net = conv2d(net, 128, 3, 1, name='conv3_1')
	net = conv2d(net, 64, 1, 0, name='conv3_2')
	net = conv2d(net, 128, 3, 1, name='conv3_3')
	net = maxpool(net, 2, 2, name='pool3')

	net = conv2d(net, 256, 3, 1, name='conv4_1')
	net = conv2d(net, 128, 1, 0, name='conv4_2')
	net = conv2d(net, 256, 3, 1, name='conv4_3')
	net = maxpool(net, 2, 2, name='pool4')

	net = conv2d(net, 512, 3, 1, name='conv5_1')
	net = conv2d(net, 256, 1, 0,name='conv5_2')
	net = conv2d(net,512, 3, 1, name='conv5_3')
	net = conv2d(net, 256, 1, 0, name='conv5_4')
	net = conv2d(net, 512, 3, 1, name='conv5_5')
	shortcut = net # 存储这一层特征图，以便后面passthrough层
	net = maxpool(net, 2, 2, name='pool5')

	net = conv2d(net, 1024, 3, 1, name='conv6_1')
	net = conv2d(net, 512, 1, 0, name='conv6_2')
	net = conv2d(net, 1024, 3, 1, name='conv6_3')
	net = conv2d(net, 512, 1, 0, name='conv6_4')
	net = conv2d(net, 1024, 3, 1, name='conv6_5')

	net = conv2d(net, 1024, 3, 1, name='conv7_1')
	net = conv2d(net, 1024, 3, 1, name='conv7_2')
	# shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
	# 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图
	shortcut = conv2d(shortcut, 64, 1, 0, name='conv_shortcut')
	shortcut = reorg(shortcut, 2)
	net = tf.concat([shortcut, net], axis=-1) # channel整合到一起
	net = conv2d(net, 1024, 3, 1, name='conv8')

	# detection layer:最后用一个1*1卷积去调整channel，该层没有BN层和激活函数
	output = conv2d(net, filters_num=n_last_channels, filters_size=1, batch_normalize=False,
				 activation=None, use_bias=True, name='conv_dec')

	return output
#########################################################################################################
def print_activations(t):
		print(t.op.name, ' ', t.get_shape().as_list())

def activation_leaky_relu(alpha):
     with tf.variable_scope("leaky_relu"):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
        return op
		
		
def build_network(images,
                    num_outputs=425,
                    alpha=0.1,
                    keep_prob=0.5,
                    is_training=False,
                    scope='yolov2'):
	batch_norm_params = {
        'is_training': is_training,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }
	with tf.variable_scope(scope):
		with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=activation_leaky_relu(alpha),
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                padding='SAME'):
			with slim.arg_scope([slim.batch_norm], **batch_norm_params):
						net = slim.conv2d(images, 32, 3,  scope='conv1')    #[1, 416, 416, 32]
						print_activations(net)
						net = slim.max_pool2d(net, 2, padding='VALID', scope='pool1')    #[1, 208, 208, 32]
						print_activations(net)
						net = slim.conv2d(net, 64, 3, scope='conv2')    #[1, 208, 208, 64]
						print_activations(net)
						net = slim.max_pool2d(net, 2, padding='VALID', scope='pool2')    #[1, 104, 104, 64]
						print_activations(net)
						net = slim.conv2d(net, 128, 3, scope='conv3_1')    #[1, 104, 104, 128]
						print_activations(net)
						net = slim.conv2d(net, 64, 1, scope='conv3_2')    #[1, 104, 104, 64]
						print_activations(net)
						net = slim.conv2d(net, 128, 3, scope='conv3_3')    #[1, 104, 104, 128]
						print_activations(net)
						net = slim.max_pool2d(net, 2, padding='VALID', scope='pool3')    #[1, 52, 52, 128]
						print_activations(net)
						net = slim.conv2d(net, 256, 3, scope='conv4_1')    #[1, 52, 52, 256]
						print_activations(net)
						net = slim.conv2d(net, 128, 1, scope='conv4_2')    #[1, 52, 52, 128]
						print_activations(net)
						net = slim.conv2d(net, 256, 3, scope='conv4_3')    #[1, 52, 52, 256]
						print_activations(net)
						net = slim.max_pool2d(net, 2, padding='VALID', scope='pool4')    #[1, 26, 26, 256]
						print_activations(net)
						net = slim.conv2d(net, 512, 3, scope='conv5_1')    #[1, 26, 26, 512]
						print_activations(net)
						net = slim.conv2d(net, 256, 1, scope='conv5_2')    #[1, 26, 26, 256]
						print_activations(net)
						net = slim.conv2d(net, 512, 3, scope='conv5_3')    #[1, 26, 26, 512]
						print_activations(net)
						net = slim.conv2d(net, 256, 1, scope='conv5_4')    #[1, 26, 26, 256]
						print_activations(net)
						net = slim.conv2d(net, 512, 3, scope='conv5_5')    #[1, 26, 26, 512]
						print_activations(net)
						shortcut = net # 存储这一层特征图，以便后面passthrough层
						net = slim.max_pool2d(net, 2, padding='VALID', scope='pool5')    #[1, 13, 13, 512]
						print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv6_1')    #[1, 13, 13, 1024]
						print_activations(net)
						net = slim.conv2d(net, 512, 1, scope='conv6_2')    #[1, 13, 13, 512]
						print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv6_3')    #[1, 13, 13, 1024]
						print_activations(net)
						net = slim.conv2d(net, 512, 1, scope='conv6_4')    #[1, 13, 13, 512])
						print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv6_5')    #[1, 13, 13, 1024]
						print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv7_1')    #[1, 13, 13, 1024]
						print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv7_2')    #[1, 13, 13, 1024]
						print_activations(net)
						shortcut = slim.conv2d(shortcut, 64, 1, scope='conv_shortcut')    #[None, 16, 16, 1024]  => [None, 7, 7, 1024]
						print_activations(shortcut)
						shortcut = reorg(shortcut, 2)
						print_activations(shortcut)
						net = tf.concat([shortcut, net], axis=-1) # channel整合到一起
						print_activations(net)
						net = slim.conv2d(net, 1024, 3, scope='conv8')    #[1, 13, 13, 1024]
						print_activations(net)
						predicts = slim.conv2d(net, num_outputs, 1, scope='conv_dec',activation_fn=None,normalizer_fn=None,)    #[None, 7, 7, 1024]  => [None, 7, 7, 1024]
						print_activations(predicts)
						return predicts
						

						
if __name__ == '__main__':
	
	yolo_darknet=[]
	yolo_slim=[]

	x = tf.random_normal([1, 416, 416, 3])
	
# 	model_output = darknet(x)
# 	print("darknet-variables")
# 	valiables = slim.get_variables()
# 	for var in  valiables:
# 		yolo_darknet.append(str(var.op.name))
# 		print(var.op.name)
# 	init_fn = slim.assign_from_checkpoint_fn("./pretrainded/yolov2_checkpoint_dir/yolo2_coco.ckpt", slim.get_variables())
	
	print("build_network")
	output = build_network(x)
	print("build_network-variables")
	valiables = slim.get_variables("yolov2")
	for var in  valiables:
		yolo_slim.append(str(var.op.name))
		print(var.op.name)
	

# 	print("dict")
# 	dict={}
# 	for new, old in zip(yolo_slim,yolo_darknet):
# 		dict[new]=old
# 		print(new, old)
# 	print(dict)


	variable_retore={'yolov2/conv1/weights': 'conv1/kernel', 'yolov2/conv6_4/BatchNorm/beta': 'conv6_4_bn/beta', 'yolov2/conv4_2/BatchNorm/moving_variance': 'conv4_2_bn/moving_variance', 
 'yolov2/conv4_2/weights': 'conv4_2/kernel', 'yolov2/conv6_1/BatchNorm/beta': 'conv6_1_bn/beta', 'yolov2/conv_dec/biases': 'conv_dec/bias', 'yolov2/conv5_1/BatchNorm/beta': 'conv5_1_bn/beta', 
 'yolov2/conv3_1/BatchNorm/beta': 'conv3_1_bn/beta', 'yolov2/conv6_4/BatchNorm/moving_variance': 'conv6_4_bn/moving_variance',
 'yolov2/conv6_4/BatchNorm/gamma': 'conv6_4_bn/gamma', 'yolov2/conv5_5/BatchNorm/beta': 'conv5_5_bn/beta', 'yolov2/conv4_1/BatchNorm/moving_mean': 'conv4_1_bn/moving_mean',
 'yolov2/conv5_4/weights': 'conv5_4/kernel', 'yolov2/conv4_3/BatchNorm/moving_mean': 'conv4_3_bn/moving_mean', 'yolov2/conv1/BatchNorm/beta': 'conv1_bn/beta', 'yolov2/conv6_3/weights': 'conv6_3/kernel',
 'yolov2/conv6_1/BatchNorm/moving_mean': 'conv6_1_bn/moving_mean', 'yolov2/conv5_1/BatchNorm/moving_mean': 'conv5_1_bn/moving_mean',
 'yolov2/conv4_1/BatchNorm/moving_variance': 'conv4_1_bn/moving_variance', 'yolov2/conv5_3/BatchNorm/moving_variance': 'conv5_3_bn/moving_variance', 'yolov2/conv6_3/BatchNorm/gamma': 'conv6_3_bn/gamma',
 'yolov2/conv6_5/weights': 'conv6_5/kernel', 'yolov2/conv4_2/BatchNorm/moving_mean': 'conv4_2_bn/moving_mean', 'yolov2/conv6_2/BatchNorm/beta': 'conv6_2_bn/beta',
 'yolov2/conv5_5/BatchNorm/moving_variance': 'conv5_5_bn/moving_variance', 
 'yolov2/conv3_2/BatchNorm/moving_mean': 'conv3_2_bn/moving_mean', 'yolov2/conv2/BatchNorm/moving_mean': 'conv2_bn/moving_mean', 'yolov2/conv8/BatchNorm/gamma': 'conv8_bn/gamma', 
 'yolov2/conv5_3/BatchNorm/beta': 'conv5_3_bn/beta', 'yolov2/conv2/weights': 'conv2/kernel', 'yolov2/conv4_3/BatchNorm/moving_variance': 'conv4_3_bn/moving_variance', 'yolov2/conv3_2/weights': 'conv3_2/kernel',
 'yolov2/conv7_2/weights': 'conv7_2/kernel', 'yolov2/conv3_3/BatchNorm/gamma': 'conv3_3_bn/gamma', 'yolov2/conv1/BatchNorm/gamma': 'conv1_bn/gamma', 'yolov2/conv8/BatchNorm/moving_variance': 'conv8_bn/moving_variance',
 'yolov2/conv6_5/BatchNorm/moving_variance': 'conv6_5_bn/moving_variance', 'yolov2/conv_shortcut/BatchNorm/gamma': 'conv_shortcut_bn/gamma', 'yolov2/conv5_5/BatchNorm/gamma': 'conv5_5_bn/gamma',
 'yolov2/conv7_1/BatchNorm/beta': 'conv7_1_bn/beta', 'yolov2/conv6_2/weights': 'conv6_2/kernel', 'yolov2/conv6_1/BatchNorm/gamma': 'conv6_1_bn/gamma', 
 'yolov2/conv6_2/BatchNorm/moving_variance': 'conv6_2_bn/moving_variance', 'yolov2/conv4_1/weights': 'conv4_1/kernel', 'yolov2/conv_shortcut/weights': 'conv_shortcut/kernel', 
 'yolov2/conv6_5/BatchNorm/beta': 'conv6_5_bn/beta', 'yolov2/conv5_4/BatchNorm/moving_variance': 'conv5_4_bn/moving_variance', 'yolov2/conv_dec/weights': 'conv_dec/kernel', 
 'yolov2/conv7_1/BatchNorm/moving_variance': 'conv7_1_bn/moving_variance', 'yolov2/conv_shortcut/BatchNorm/beta': 'conv_shortcut_bn/beta', 'yolov2/conv3_2/BatchNorm/beta': 'conv3_2_bn/beta', 
 'yolov2/conv5_3/weights': 'conv5_3/kernel', 'yolov2/conv4_2/BatchNorm/beta': 'conv4_2_bn/beta', 'yolov2/conv5_5/weights': 'conv5_5/kernel', 'yolov2/conv8/weights': 'conv8/kernel', 
 'yolov2/conv7_1/BatchNorm/gamma': 'conv7_1_bn/gamma', 'yolov2/conv1/BatchNorm/moving_mean': 'conv1_bn/moving_mean', 'yolov2/conv4_2/BatchNorm/gamma': 'conv4_2_bn/gamma', 'yolov2/conv7_2/BatchNorm/beta': 'conv7_2_bn/beta',
 'yolov2/conv7_1/weights': 'conv7_1/kernel', 'yolov2/conv5_2/weights': 'conv5_2/kernel', 'yolov2/conv5_2/BatchNorm/moving_variance': 'conv5_2_bn/moving_variance', 
 'yolov2/conv6_2/BatchNorm/moving_mean': 'conv6_2_bn/moving_mean', 'yolov2/conv3_2/BatchNorm/gamma': 'conv3_2_bn/gamma', 'yolov2/conv2/BatchNorm/beta': 'conv2_bn/beta', 
 'yolov2/conv6_3/BatchNorm/moving_variance': 'conv6_3_bn/moving_variance', 'yolov2/conv5_4/BatchNorm/gamma': 'conv5_4_bn/gamma', 'yolov2/conv1/BatchNorm/moving_variance': 'conv1_bn/moving_variance',
 'yolov2/conv4_3/BatchNorm/beta': 'conv4_3_bn/beta', 'yolov2/conv4_3/weights': 'conv4_3/kernel', 
 'yolov2/conv2/BatchNorm/moving_variance': 'conv2_bn/moving_variance', 'yolov2/conv2/BatchNorm/gamma': 'conv2_bn/gamma', 'yolov2/conv3_3/BatchNorm/moving_mean': 'conv3_3_bn/moving_mean', 
 'yolov2/conv3_1/BatchNorm/gamma': 'conv3_1_bn/gamma', 'yolov2/conv_shortcut/BatchNorm/moving_mean': 'conv_shortcut_bn/moving_mean', 'yolov2/conv5_4/BatchNorm/moving_mean': 'conv5_4_bn/moving_mean', 
 'yolov2/conv6_2/BatchNorm/gamma': 'conv6_2_bn/gamma', 'yolov2/conv7_2/BatchNorm/gamma': 'conv7_2_bn/gamma', 'yolov2/conv7_1/BatchNorm/moving_mean': 'conv7_1_bn/moving_mean',
 'yolov2/conv7_2/BatchNorm/moving_variance': 'conv7_2_bn/moving_variance', 'yolov2/conv_shortcut/BatchNorm/moving_variance': 'conv_shortcut_bn/moving_variance', 'yolov2/conv5_1/BatchNorm/gamma': 'conv5_1_bn/gamma',
 'yolov2/conv5_2/BatchNorm/beta': 'conv5_2_bn/beta', 'yolov2/conv5_3/BatchNorm/moving_mean': 'conv5_3_bn/moving_mean', 'yolov2/conv5_3/BatchNorm/gamma': 'conv5_3_bn/gamma', 
 'yolov2/conv3_3/BatchNorm/beta': 'conv3_3_bn/beta', 'yolov2/conv5_4/BatchNorm/beta': 'conv5_4_bn/beta', 'yolov2/conv3_3/BatchNorm/moving_variance': 'conv3_3_bn/moving_variance', 
 'yolov2/conv7_2/BatchNorm/moving_mean': 'conv7_2_bn/moving_mean', 'yolov2/conv3_2/BatchNorm/moving_variance': 'conv3_2_bn/moving_variance', 'yolov2/conv4_3/BatchNorm/gamma': 'conv4_3_bn/gamma', 
 'yolov2/conv4_1/BatchNorm/beta': 'conv4_1_bn/beta', 'yolov2/conv3_1/BatchNorm/moving_variance': 'conv3_1_bn/moving_variance', 
 'yolov2/conv3_1/BatchNorm/moving_mean': 'conv3_1_bn/moving_mean', 'yolov2/conv6_1/BatchNorm/moving_variance': 'conv6_1_bn/moving_variance', 
 'yolov2/conv8/BatchNorm/beta': 'conv8_bn/beta', 'yolov2/conv6_3/BatchNorm/beta': 'conv6_3_bn/beta', 
 'yolov2/conv5_5/BatchNorm/moving_mean': 'conv5_5_bn/moving_mean', 'yolov2/conv8/BatchNorm/moving_mean': 'conv8_bn/moving_mean', 'yolov2/conv4_1/BatchNorm/gamma': 'conv4_1_bn/gamma', 
 'yolov2/conv5_2/BatchNorm/moving_mean': 'conv5_2_bn/moving_mean', 'yolov2/conv3_1/weights': 'conv3_1/kernel', 'yolov2/conv6_1/weights': 'conv6_1/kernel', 
 'yolov2/conv6_4/weights': 'conv6_4/kernel', 'yolov2/conv3_3/weights': 'conv3_3/kernel',
 'yolov2/conv5_1/BatchNorm/moving_variance': 'conv5_1_bn/moving_variance', 'yolov2/conv5_1/weights': 'conv5_1/kernel', 'yolov2/conv6_5/BatchNorm/moving_mean': 'conv6_5_bn/moving_mean', 
 'yolov2/conv6_4/BatchNorm/moving_mean': 'conv6_4_bn/moving_mean', 'yolov2/conv5_2/BatchNorm/gamma': 'conv5_2_bn/gamma', 'yolov2/conv6_3/BatchNorm/moving_mean': 'conv6_3_bn/moving_mean', 
 'yolov2/conv6_5/BatchNorm/gamma': 'conv6_5_bn/gamma'}
	
	
# 	for k,v in variable_retore.items():
# 		print(k,v)

	saver = tf.train.Saver()
	with tf.Session() as sess:
# 		init_fn(sess)
		
		writer =tf.summary.FileWriter("logs/",graph = sess.graph)
		writer.close()
		# 必须先restore模型才能打印shape;导入模型时，上面每层网络的name不能修改，否则找不到
		#print("ckpt name")
		#print_tensor_name("./pretrainded/yolov2_checkpoint_dir/yolo2_coco.ckpt")
		#saver.restore(sess, "./pretrainded/yolov2_checkpoint_dir/yolo2_coco.ckpt")
		
		reader = pywrap_tensorflow.NewCheckpointReader("./pretrainded/yolov2_checkpoint_dir/yolo2_coco.ckpt")
		var_to_shape_map = reader.get_variable_to_shape_map()
		for key in var_to_shape_map:
		    print("tensor_name: ", key)
		    #print(reader.get_tensor(key))
		with tf.variable_scope('', reuse = True):
			for k,v in variable_retore.items():
				print(k,v)
				sess.run(tf.get_variable(k).assign(reader.get_tensor(v)))
		
		
		saver.save(sess,"models/model_slim.ckpt")
		print(sess.run(output).shape) # (1,13,13,425)
		