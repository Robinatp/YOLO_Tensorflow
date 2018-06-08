# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.contrib import slim
import time
from datetime import datetime
from YOLOV3 import yolo_v3, load_weights, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', 'images/giraffe.jpg', 'Input image')
tf.app.flags.DEFINE_string('output_img', 'out/detected.jpg', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'model_data/coco_classes.txt', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'weight/yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def draw_boxes(boxes, img, cls_names, size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)

def _draw_boxes(bboxes,scores,classes, img, cls_names, size):
    draw = ImageDraw.Draw(img)

    for i in range(len(classes)):
        color = tuple(np.random.randint(0, 256, 3))
        
        box = convert_to_original_size(bboxes[i], np.array(size), np.array(img.size))
        draw.rectangle(box, outline=color)
        draw.text(box[:2], '{} {:.2f}%'.format(cls_names[classes[i]], scores[i] * 100), fill=color)


def convert_to_original_size(box, size, original_size):
    #ratio = 1.0*original_size / size
    box = box.reshape(2, 2) * original_size
    return list(box.reshape(-1))


def main(argv=None):
    
    BASE_PATH = 'images'
    TEST_IMAGES = os.listdir(BASE_PATH)
    TEST_IMAGES.sort()
    print(TEST_IMAGES)
    
    
#     img = Image.open(FLAGS.input_img)
#     w,h = img.size
#     img_resized = img.resize(size=(FLAGS.size, FLAGS.size))

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

    with tf.variable_scope('detector'):
        detections, pre_bboxes,pre_scores,pre_classes = yolo_v3(inputs, len(classes), data_format='NHWC')#Tensor("detector/yolo-v3/concat:0", shape=(?, 10647, 85), dtype=float32)
        #load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    #coordinates of top left and bottom right points+num_class_confidence
#     detections = detections_boxes(detections)#shape=(?, 10647, 85), dtype=float32)
    

    #saver = tf.train.Saver()
    model_path = "models/yolov3.ckpt"
    init_fn = slim.assign_from_checkpoint_fn(model_path,slim.get_variables())
    
    with tf.Session() as sess:
        init_fn(sess)
        #sess.run(load_ops)

        writer =tf.summary.FileWriter("logs/",graph = sess.graph)
        writer.close()
        #saver.save(sess,"models/yolov3.ckpt")
        
        for img in TEST_IMAGES:
            image_path = os.path.join(BASE_PATH, img)
       
            image = Image.open(image_path)
            w,h = image.size
            img_resized = image.resize(size=(FLAGS.size, FLAGS.size))
          
            start_time = time.time()
            detected_boxes,pbboxes,pscores,pclasses = sess.run([detections,pre_bboxes,pre_scores,pre_classes], 
                                                               feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
            duration = time.time() - start_time
            print ('%s: yolov2.run(), duration = %.3f' %(datetime.now(), duration))
            print(pbboxes,pscores,pclasses)

#             filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold,
#                                          iou_threshold=FLAGS.iou_threshold)
 
            #draw_boxes(filtered_boxes, image, classes, (FLAGS.size, FLAGS.size))
            _draw_boxes(pbboxes,pscores,pclasses, image, classes, (FLAGS.size, FLAGS.size))
             
            plt.imshow(image)
            plt.show()
 
            image.save(FLAGS.output_img)


if __name__ == '__main__':
    tf.app.run()
