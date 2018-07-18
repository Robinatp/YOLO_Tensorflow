![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## Quick Start

- Download Darknet weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).
- Convert the Darknet YOLO_v2/YOLO_v3 model to a tensorflw  model by tf_slim.
- Save all models(ckpt,pb) in 'models/' and Test the model on the small test set `data/`.

```bash
#create dirs
mkdir models
mkdir weight

# prepare the data and download darknet weight from website
cd weight
1,method I
sh download.sh

2,method II
#https://pjreddie.com/darknet/yolo/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
#./darknet detect cfg/yolov3.cfg weight/yolov3.weights data/dog.jpg
#./darknet detect cfg/yolov3-tiny.cfg weight/yolov3-tiny.weights data/dog.jpg

#https://pjreddie.com/darknet/yolov2/
wget https://pjreddie.com/media/files/yolov2.weights
wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
#./darknet detect cfg/yolov2.cfg weight/yolov2.weights data/dog.jpg
#./darknet detector test cfg/voc.data cfg/yolov2-tiny-voc.cfg weight/yolov2-tiny-voc.weights data/dog.jpg

#https://pjreddie.com/darknet/yolov1/
wget http://pjreddie.com/media/files/yolov1/yolov1.weights
wget http://pjreddie.com/media/files/yolov1/tiny-yolov1.weights
#./darknet yolo test cfg/yolov1.cfg weight/yolov1.weights data/dog.jpg
#./darknet yolo test cfg/yolov1-tiny.cfg  weight/tiny-yolov1.weights data/person.jpg

# run the graph
cd ..
python YOLO_V1_Tiny_convert_darkenet_to_Tensorflow.py	
python YOLO_V2_Tiny_Voc_convert_darkenet_to_Tensorflow.py	
python YOLO_V2_convert_darkenet_to_Tensorflow.py	
python YOLO_V3_Tiny_convert_darkenet_to_Tensorflow.py
python YOLO_V3_convert_darkenet_to_Tensorflow.py

Genetate the ckpt in models, the pb in models, logs

#look at the graph structure
 tensorboard --logdir logs


```
