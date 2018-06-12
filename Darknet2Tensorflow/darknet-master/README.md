![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## Quick Start

- Download Darknet weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).
- Convert the Darknet YOLO_v2/YOLO_v2 model to a tensorflw  model by tf_slim.
- Save all models(ckpt,pb) in 'models/' and Test the model on the small test set `data/`.

```bash
mkdir models
mkdir weight
cd weight
wget https://pjreddie.com/media/files/yolov2.weights
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
python YOLO_V2_convert_darkenet_to_Tensorflow.py
python YOLO_V3_convert_darkenet_to_Tensorflow.py
```
