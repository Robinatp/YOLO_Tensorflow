# -*- coding: utf-8 -*-
# --------------------------------------
# Description :YOLO_v2主函数.
# --------------------------------------

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import time
from datetime import datetime

from model_darknet19_slim import build_network
from postprocess import decode
from utils import preprocess_image, postprocess, _draw_detection
from config import anchors, class_names
from tensorflow.contrib import slim
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph, ["detected_scores","detected_boxes","detected_classes"])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def main():
    input_size = (416,416)
    image_file = './yolo2_data/car.jpg'
    image = cv2.imread(image_file)

    # copy、resize416*416、归一化、在第0维增加存放batchsize维度
    image_cp = preprocess_image(image,input_size)

    # 【1】输入图片进入darknet19网络得到特征图，并进行解码得到：左上+右下boxes(xmin,ymin,xmax,ymax)、置信度scores、类别classes
    tf_image = tf.placeholder(tf.float32,[1,input_size[0],input_size[1],3],name= "input")
    model_output = build_network(tf_image) # darknet19网络输出的特征图
    output_sizes = input_size[0]//32, input_size[1]//32 # 特征图尺寸是图片下采样32倍
    output_decoded = decode(model_output=model_output,output_sizes=output_sizes,
                               num_class=len(class_names),anchors=anchors)  # 解码
    
    model_path = "models/model_slim.ckpt"
    init_fn = slim.assign_from_checkpoint_fn(model_path,slim.get_variables())
    
    with tf.Session() as sess:
        init_fn(sess)
        
        writer =tf.summary.FileWriter("logs/",graph = sess.graph)
        writer.close()
        
        start_time = time.time()
        boxes,scores,classes = sess.run(output_decoded,feed_dict={tf_image:image_cp})
        duration = time.time() - start_time
        print ('%s: yolov2.run(), duration = %.3f' %(datetime.now(), duration))
        print(boxes, scores, classes)
    
    # 【2】绘制筛选后的边界框
    img_detection = _draw_detection(image, boxes, scores, classes, class_names)
    
    cv2.imwrite("./yolo2_data/detection.jpg", img_detection)
    cv2.imshow("detection_results", img_detection)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()