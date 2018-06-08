# tensorflow-yolo-v3

Implementation of YOLO v3 object detector in Tensorflow (TF-Slim). Full tutorial can be found [here](https://pjreddie.com/media/files/yolov3.weights), train for coco dataset.

Tested on Tensorflow 1.8.0 on Ubuntu 16.04.


## 文件说明：<br>
1、convert_ckpt_and_inference.py：load yolov3.weights and convert it to models/yolov3.ckpt which is used by YOLO_V3_inference.py <br>
2、YOLO_V3_inference.py: a demo inference for yolo_v3<br>
3、YOLOV3.py：配置文件，包含anchor尺寸、coco数据集的80个classes类别名称，使用tensorflow做postprocess<br>
4、yolo_v3.py：配置文件，包含anchor尺寸、coco数据集的80个classes类别名称,使用numpy做postprocess<br>


## 运行YOLO_V3_inference.py即可得到效果图：<br>
1、horses.jpg：输入的待检测图片<br><br>
![image](images/horses.jpg)<br>
2、out/detected1.jpg：检测结果可视化<br><br>
![image](out/detected1.jpg)<br>
3、kite.jpg：输入的待检测图片<br><br>
![image](images/kite.jpg)<br>
4、out/detected1.jpg：检测结果可视化<br><br>
![image](out/detected2.jpg)<br>
5、YOLOV3_graph.png：the Graph for YOLO_V3 from tensorboard
![image](YOLOV3_graph.png)<br>

## Todo list:
- [x] YOLO v3 architecture
- [x] Basic working demo
- [ ] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [ ] Training pipeline
- [ ] More backends


