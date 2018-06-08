# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]


def darknet53(inputs):
    """
    Builds Darknet-53 model.
    """
    inputs = _conv2d_fixed_padding(inputs, 32, 3)
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2)
    
    inputs = _darknet53_block(inputs, 32)
    
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)

    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)

    route_1 = inputs#52X52X256
    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 256)

    route_2 = inputs#26X26X512
    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)

    for i in range(4):
        inputs = _darknet53_block(inputs, 512)#inputs 13X13X1024
    
    
    return route_1, route_2, inputs
    #route_1    Tensor: Tensor("detector/darknet-53/add_10:0", shape=(?, 52, 52, 256), dtype=float32)    
    #route_2    Tensor: Tensor("detector/darknet-53/add_18:0", shape=(?, 26, 26, 512), dtype=float32)    
    #inputs    Tensor: Tensor("detector/darknet-53/add_22:0", shape=(?, 13, 13, 1024), dtype=float32)


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    
    inputs = inputs + shortcut
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, mode='CONSTANT', **kwargs):
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


def _yolo_block(inputs, filters):
    with tf.name_scope("yolo_block"):
        inputs = _conv2d_fixed_padding(inputs, filters, 1)
        inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = _conv2d_fixed_padding(inputs, filters, 1)
        inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = _conv2d_fixed_padding(inputs, filters, 1)
        route = inputs
        inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs


def _get_size(shape, data_format):
    if len(shape) == 4:
        shape = shape[1:]
    return shape[1:3] if data_format == 'NCHW' else shape[0:2]


def _detection_layer(inputs, num_classes, anchors, img_size, data_format):
    with tf.name_scope("detection_layer"):
        num_anchors = len(anchors)#3
        predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1, normalizer_fn=None,
                                  activation_fn=None, biases_initializer=tf.zeros_initializer())
    
        shape = predictions.get_shape().as_list()
        grid_size = _get_size(shape, data_format)
        dim = grid_size[0] * grid_size[1]
        bbox_attrs = 5 + num_classes#85
    
        if data_format == 'NCHW':
            predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
            predictions = tf.transpose(predictions, [0, 2, 1])
    
        predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])
    
        stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
        print(stride)
    
        anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
    
        box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)
    
        box_centers = tf.nn.sigmoid(box_centers)
        confidence = tf.nn.sigmoid(confidence)
    
        grid_x = tf.range(grid_size[0], dtype=tf.float32)
        grid_y = tf.range(grid_size[1], dtype=tf.float32)
        a, b = tf.meshgrid(grid_x, grid_y)
    
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
    
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])
    
        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride
    
        anchors = tf.tile(anchors, [dim, 1])
        box_sizes = tf.exp(box_sizes) * tf.to_float(anchors)
        box_sizes = box_sizes * stride
    
        #(xcenter, ycenter, w, h)
        detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
    
        classes = tf.nn.sigmoid(classes)
        predictions = tf.concat([detections, classes], axis=-1)
        return predictions
    
def _ratio_detection_layer(inputs, num_classes, anchors, img_size, data_format):
    with tf.name_scope("detection_layer"):
        num_anchors = len(anchors)#3
        predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1, normalizer_fn=None,
                                  activation_fn=None, biases_initializer=tf.zeros_initializer())
    
        shape = predictions.get_shape().as_list()
        #[None, 13, 13, 255], scale1
        #[None, 26, 26, 255], scale2
        #[None, 52, 52, 255], scale3

        grid_size = _get_size(shape, data_format)
        #[13, 13]
        #[26, 26]
        #[52, 52]
       
        with tf.name_scope("anchor_meshgrid"):
            grid_x = tf.range(grid_size[0], dtype=tf.float32)
            grid_y = tf.range(grid_size[1], dtype=tf.float32)
            a, b = tf.meshgrid(grid_x, grid_y)
        
            x_offset = tf.reshape(a, (-1, 1))
            y_offset = tf.reshape(b, (-1, 1))
        
            #the offset from the top left corner of the image by x_y_offset, 1 stand for stride
            x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
            x_y_offset = tf.cast(x_y_offset, tf.float32)
            x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])
    
        with tf.name_scope("slice"):
            dim = grid_size[0] * grid_size[1]    #169,676,2704
            bbox_attrs = 5 + num_classes         #85
            
            if data_format == 'NCHW':
                predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
                predictions = tf.transpose(predictions, [0, 2, 1])

            predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])
            #Tensor("detector/yolo-v3/Predict_1/detection_layer/Reshape:0", shape=(?, 507, 85), dtype=float32)
            #Tensor("detector/yolo-v3/Predict_2/detection_layer/Reshape:0", shape=(?, 2028, 85), dtype=float32)
            #Tensor("detector/yolo-v3/Predict_3/detection_layer/Reshape:0", shape=(?, 8112, 85), dtype=float32)
            
            box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)
            box_centers = tf.nn.sigmoid(box_centers)
            confidence = tf.nn.sigmoid(confidence)
            classes = tf.nn.sigmoid(classes)
        
        with tf.name_scope("box_coord"):
            box_centers = box_centers + x_y_offset
            # the ratio offset from the top left corner of the image
            box_centers = box_centers /grid_size
        
            stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
            #(32, 32)
            #(16, 16)
            #(8, 8)
        
            anchors = [(1.0*a[0] / stride[0], 1.0*a[1] / stride[1]) for a in anchors]
            #[(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]
            #[(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)]
            #[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)]
            
            anchors = tf.tile(anchors, [dim, 1])
            box_sizes = tf.exp(box_sizes) * anchors
            # the ratio size of the image
            box_sizes = box_sizes /grid_size
        
        #(xcenter, ycenter, w, h, confidence ,classes_probability)
        predictions = tf.concat([box_centers, box_sizes, confidence,classes], axis=-1)
        
        return predictions


def _upsample(inputs, out_shape, data_format='NCHW'):
    # we need to pad with one pixel, so we set kernel_size = 3
    inputs = _fixed_padding(inputs, 3, mode='SYMMETRIC')

    # tf.image.resize_bilinear accepts input in format NHWC
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    if data_format == 'NCHW':
        height = out_shape[3]
        width = out_shape[2]
    else:
        height = out_shape[2]
        width = out_shape[1]

    # we padded with 1 pixel from each side and upsample by factor of 2, so new dimensions will be
    # greater by 4 pixels after interpolation
    new_height = height + 4
    new_width = width + 4

    inputs = tf.image.resize_bilinear(inputs, (new_height, new_width))

    # trim back to desired size
    inputs = inputs[:, 2:-2, 2:-2, :]

    # back to NCHW if needed
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.identity(inputs, name='upsampled')
    return inputs


def yolo_v3(inputs, num_classes, score_threshold=0.5, iou_threshold=0.5, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v3 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]#(416,416)

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], data_format=data_format, reuse=reuse):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
            with tf.variable_scope('darknet-53'):
                route_1, route_2, inputs = darknet53(inputs)
                #route_1    Tensor: Tensor("detector/darknet-53/add_10:0", shape=(?, 52, 52, 256), dtype=float32)    
                #route_2    Tensor: Tensor("detector/darknet-53/add_18:0", shape=(?, 26, 26, 512), dtype=float32)    
                #inputs    Tensor: Tensor("detector/darknet-53/add_22:0", shape=(?, 13, 13, 1024), dtype=float32)

            with tf.variable_scope('yolo-v3'):
                with tf.name_scope("Predict_1"):
                    route, inputs = _yolo_block(inputs, 512)
                    detect_1 = _ratio_detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
                    detect_1 = tf.identity(detect_1, name='scale_1')#Tensor("detector/yolo-v3/Detection_0/detect_1:0", shape=(?, 507, 85), dtype=float32)

                with tf.name_scope("Predict_2"):
                    with tf.name_scope("upsample"):
                        inputs = _conv2d_fixed_padding(route, 256, 1)
                        upsample_size = route_2.get_shape().as_list()
                        inputs = _upsample(inputs, upsample_size, data_format)
                        inputs = tf.concat([inputs, route_2], axis=1 if data_format == 'NCHW' else 3)
    
                    route, inputs = _yolo_block(inputs, 256)
                    detect_2 = _ratio_detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                    detect_2 = tf.identity(detect_2, name='scale_2')#Tensor("detector/yolo-v3/Detection_2/detect_2:0", shape=(?, 2028, 85), dtype=float32)
                
                with tf.name_scope("Predict_3"):
                    with tf.name_scope("upsample"):
                        inputs = _conv2d_fixed_padding(route, 128, 1)
                        upsample_size = route_1.get_shape().as_list()
                        inputs = _upsample(inputs, upsample_size, data_format)
                        inputs = tf.concat([inputs, route_1], axis=1 if data_format == 'NCHW' else 3)
    
                    _, inputs = _yolo_block(inputs, 128)
                    detect_3 = _ratio_detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                    detect_3 = tf.identity(detect_3, name='scale_3')#Tensor("detector/yolo-v3/Detection_3/detect_3:0", shape=(?, 8112, 85), dtype=float32)
    
                #(xcenter, ycenter, w, h, confidence ,classes_probability)
                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                
                
                with tf.name_scope("coord_transform"):
                    center_x, center_y, width, height, confidence, class_prob = tf.split(detections, [1, 1, 1, 1, 1, -1], axis=-1)
                    w2 = width * 0.5
                    h2 = height * 0.5       
                    bboxes = tf.concat([center_x - w2, center_y - h2, center_x + w2, center_y + h2], axis=-1)
                    
                with tf.name_scope("select-threhold"):
                    # score filter
                    box_scores = confidence * class_prob      # (?, 20)
                    box_label = tf.to_int32(tf.argmax(box_scores, axis=-1))    # (?, )
                    box_scores_max = tf.reduce_max(box_scores, axis=-1)     # (?, )
             
                    pred_mask = box_scores_max > score_threshold
                    bboxes = tf.boolean_mask(bboxes, pred_mask)
                    scores = tf.boolean_mask(box_scores_max, pred_mask)
                    classes = tf.boolean_mask(box_label, pred_mask)
             
                with tf.name_scope("NMS"):
                    # non_max_suppression
                    xmin,ymin,xmax,ymax = tf.unstack(tf.transpose(bboxes))
                    _boxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
                    nms_indices = tf.image.non_max_suppression(_boxes, scores,
                                                           max_output_size=20,
                                                           iou_threshold=iou_threshold)
#                 bboxes = tf.gather(bboxes, idx_nms)
#                 scores = tf.gather(scores, idx_nms)
#                 classes = tf.to_int32(tf.gather(classes, idx_nms))
#                 print(bboxes,scores,classes) 
                
    
    bboxes = tf.identity(tf.gather(bboxes, nms_indices),name="detected_boxes")
    scores = tf.identity(tf.gather(scores, nms_indices), name="detected_scores")
    classes = tf.identity(tf.gather(classes, nms_indices),name="detected_classes")
    print(bboxes,scores,classes)

    return detections,bboxes,scores,classes#Tensor("detector/yolo-v3/concat:0", shape=(?, 10647, 85), dtype=float32)


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
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
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def detections_boxes(detections):
    """
    Converts center x, center y, width and height values to coordinates of top left and bottom right points.
    (xmin,ymin,xmax,ymax)
    :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
    :return: converted detections of same shape as input
    """
    with tf.name_scope("coord_transform"):
        center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
        w2 = width / 2
        h2 = height / 2
        x0 = center_x - w2
        y0 = center_y - h2
        x1 = center_x + w2
        y1 = center_y + h2
    
        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1)
    return detections


def _iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes
    
    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou


def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)#get the confidence of per bounding box
    predictions = predictions_with_boxes * conf_mask#shape    <type 'tuple'>: (1, 10647, 85)

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result

