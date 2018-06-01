from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import sys
import time
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
from matplotlib import gridspec
from matplotlib import pyplot as plt

import tensorflow as tf
from collections import defaultdict



# This is needed since the notebook is stored in the object_detection folder.
TF_API="/home/ubuntu/eclipse-workspace/models/research/object_detection"
sys.path.append(os.path.split(TF_API)[0])
sys.path.append(TF_API)

from utils import np_box_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
from core import box_list


label_map_filename = "pascal_label_map.pbtxt"
class ObjectDetectModel(object):
  """Class to load ObjectDetect model and run inference."""
  def __init__(self, frozen_graph):
    """Creates and loads pretrained  model."""
    if not frozen_graph.endswith('.pb'):
        raise ValueError('frozen_graph is not a correct pb file!')
    self.is_debug = True
    
    self.graph, \
    self.boxes_tensor,\
    self.scores_tensor, \
    self.classes_tensor ,\
    self.image_tensor = self.load_graph(frozen_graph)#boxes_tensor, scores_tensor, classes_tensor,num_detections_tensor, image_tensor

    self.sess = tf.Session(graph=self.graph)
    
    self.category_index = self.load_category(label_map_filename)
    self.image_height = 0
    self.image_weight = 0
    self.boxes = None
    self.scores = None
    self.classes = None
    self.qrcode_index = -1
    self.secret_cede = None
    self.plain_code = None
  

  def load_graph(self, frozen_graph):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        boxes_tensor,scores_tensor, classes_tensor ,\
        image_tensor= \
            tf.import_graph_def(
                graph_def, 
                input_map=None, 
                return_elements=[
                  'detected_boxes:0',
                  'detected_scores:0',
                  'detected_classes:0',
                  'input:0'
                  ], 
                name="", 
                producer_op_list=None
            )
        with tf.variable_scope("coord_transform"):  
            #change the coordinates of xcenter, ycenter, w, h to ymin, xmin, ymax, xmax for object detection api     
            xcenter, ycenter, w, h = tf.unstack(tf.transpose(boxes_tensor))
            ymin = ycenter - h / 2.
            xmin = xcenter - w / 2.
            ymax = ycenter + h / 2.
            xmax = xcenter + w / 2.
            boxes_tensor = tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
        
        if(self.is_debug):
            writer = tf.summary.FileWriter("./logs_objectdetection_load", graph=graph)
            writer.close() 
        
    return graph, \
        boxes_tensor,\
        scores_tensor, \
        classes_tensor , \
        image_tensor

  def read_tensor_from_image_file(self,
                                file_name,
                                input_height=None,
                                input_width=None,
                                input_mean=0,
                                input_std=255):
      if not tf.gfile.Exists(file_name):
          raise ValueError('file_name is not Exists!')
      input_name = "file_reader"
      output_name = "normalized"
      file_reader = tf.read_file(file_name, input_name)
      if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
      elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
      elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
      else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
      if not input_height and not input_width:
          float_caster = tf.cast(image_reader, tf.float32)
          dims_expander = tf.expand_dims(float_caster, 0)
          resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
          normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
      with tf.Session() as sess:
          result = sess.run(normalized)
      return result
  
  def save_image_to_image_file(self, image, file_name):
        convert_image = tf.image.convert_image_dtype(image, dtype=tf.uint8)  
        cropped_image = tf.image.encode_jpeg(convert_image)
        with tf.Session() as sess:
            with tf.gfile.GFile(file_name, 'wb') as f:  
                string = sess.run(cropped_image)
                f.write(string)

  def load_image_into_numpy_array(self, image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

  def load_category(self, label_file):
        label_map = label_map_util.load_labelmap(label_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index
        
  def bboxes_select(self,classes, scores, bboxes, threshold=0.1):
        """Sort bounding boxes by decreasing order and keep only the top_k
        """
        mask = scores > threshold
        classes = classes[mask]
        scores = scores[mask]
        bboxes = bboxes[mask]
        return classes, scores, bboxes

  def run(self, image_np):
    """Runs inference on a single image.
    Args:
      image:  raw input image.
    Returns:
    """

    self.image_height, self.image_weight = image_np.shape[0:2]
    print(self.image_height, self.image_weight)
    # Actual detection.
    start_time = time.time()
    (boxes, scores, classes) = self.sess.run(
            [self.boxes_tensor, self.scores_tensor, self.classes_tensor],
            feed_dict={self.image_tensor: image_np})
    duration = time.time() - start_time
    print ('%s: ObjectDetectModel.run(), duration = %.3f' %(datetime.now(), duration))

    self.boxes, self.scores, self.classes = self.bboxes_select(boxes, scores, classes,0.01)
    
    
    if(self.is_debug):
        print('boxes:',self.boxes)
        print('scores:',self.scores)
        print('classes:',self.classes)
        print('center_coordinates:',self.get_center_coordinates(self.boxes))
    
    return self.boxes, self.scores, self.classes, self.category_index


  

  def slice_qrcode(self, image_np):
        if(self.qrcode_index > 0 and  self.boxes is not None):
            ymin, xmin, ymax, xmax = self.boxes[self.qrcode_index]
            QR_width = xmax - xmin
            QR_height = ymax - ymin
            crop_bounding_image = tf.image.crop_to_bounding_box(image_np,
                                                            int(ymin * self.image_height), 
                                                            int(xmin * self.image_weight), 
                                                            int(QR_height * self.image_height), 
                                                            int(QR_width * self.image_weight) )
            crop_file_name = os.path.join("crop_result", 'qrcode.jpg')
            convert_image = tf.image.convert_image_dtype(crop_bounding_image, dtype=tf.uint8)  
            cropped_image = tf.image.encode_jpeg(convert_image)
            with tf.Session() as sess:
                with tf.gfile.GFile(crop_file_name, 'wb') as f:  
                    string = sess.run(cropped_image)
                    f.write(string)
                    return convert_image.eval()
        
                            
  def get_center_coordinates(self, boxes):
        ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(boxes))
        width = xmax - xmin
        height = ymax - ymin
        ycenter = ymin + height / 2.
        xcenter = xmin + width / 2. 
        center_coordinates = tf.transpose(tf.stack([ycenter, xcenter, height, width], axis=0))
        with tf.Session() as sess:
            center_box = sess.run(center_coordinates)
        #ycenter, xcenter, height, width
        return center_box
    
    
    
def vis_bounding_box(image_np,boxes, scores, classes,category_index):
    vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes,
                classes.astype(np.int32),
                scores,
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.01,
                line_thickness=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.axis('off')
    plt.title('input image')
    plt.show()
  

def run_object_detection(ObjectDetectModel, path):
    """Inferences classify model and visualizes result."""
    print(path)
#     image = Image.open(path)
#     image_np = ObjectDetectModel.load_image_into_numpy_array(image)
    image = cv2.imread(path)
    image_np = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    boxes, scores, classes ,category_index= ObjectDetectModel.run(image_np)

    vis_bounding_box(image_np, boxes, scores, classes, category_index)
    ObjectDetectModel.save_image_to_image_file(image_np,"detection.jpg")
 

def main():
    BASE_PATH = 'image'
    TEST_IMAGES = os.listdir(BASE_PATH)
    TEST_IMAGES.sort()
    print(TEST_IMAGES)

    MODEL = ObjectDetectModel("models/yolov1_frozen_graph.pb")
    print('model loaded successfully!')
    
    for image in TEST_IMAGES:
        image_path = os.path.join(BASE_PATH, image)
        run_object_detection(MODEL,image_path)
    
main()



