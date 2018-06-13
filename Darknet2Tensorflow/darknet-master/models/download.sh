
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

