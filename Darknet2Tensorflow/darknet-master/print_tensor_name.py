import os
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow


# print all op names
def print_tensor_name(chkpt_fname):
    reader = pywrap_tensorflow.NewCheckpointReader(chkpt_fname)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key )
        #print(reader.get_tensor(key)) # Remove this is you want to print only variable names 
      
    #yolov1  
#     print(reader.get_tensor("yolo/Conv_23/BatchNorm/beta"))   
#     print(reader.get_tensor("yolo/Conv_23/BatchNorm/gamma"))
#     print(reader.get_tensor("yolo/Conv_23/BatchNorm/moving_mean")) 
#     print(reader.get_tensor("yolo/Conv_23/BatchNorm/moving_variance")) 
#     print(reader.get_tensor("yolo/Conv_23/weights")) 
    
    #yolov1 local
#     print(reader.get_tensor("yolo/Local/biases"))   
#     print(reader.get_tensor("yolo/Local/weights"))
    
    #yolov1 connected
#     print(reader.get_tensor("yolo/Fc/biases"))   
#     print(reader.get_tensor("yolo/Fc/weights"))


    #yolov1-tiny
    print(reader.get_tensor("yolov1_tiny/Conv_7/BatchNorm/beta"))   
    print(reader.get_tensor("yolov1_tiny/Conv_7/BatchNorm/gamma"))
    print(reader.get_tensor("yolov1_tiny/Conv_7/BatchNorm/moving_mean")) 
    print(reader.get_tensor("yolov1_tiny/Conv_7/BatchNorm/moving_variance")) 
    print(reader.get_tensor("yolov1_tiny/Conv_7/weights")) 
    
    #yolov2
#     print(reader.get_tensor("yolov2/Conv_21/BatchNorm/beta"))   
#     print(reader.get_tensor("yolov2/Conv_21/BatchNorm/gamma"))
#     print(reader.get_tensor("yolov2/Conv_21/BatchNorm/moving_mean")) 
#     print(reader.get_tensor("yolov2/Conv_21/BatchNorm/moving_variance")) 
#     print(reader.get_tensor("yolov2/Conv_22/weights")) 


    #yolov3
#     print(reader.get_tensor("detector/darknet-53/Conv/BatchNorm/beta")) 
#     print(reader.get_tensor("detector/darknet-53/Conv/BatchNorm/gamma")) 
#     print(reader.get_tensor("detector/darknet-53/Conv/BatchNorm/moving_mean")) 
#     print(reader.get_tensor("detector/darknet-53/Conv/BatchNorm/moving_variance")) 
    return var_to_shape_map

print_tensor_name("models/yolov1-tiny.ckpt")