from ctypes import *
from load import *
import random, os
import cv2
import numpy as np

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    return (ctype * len(values))(*values)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def object_detect(net, meta, image_path, image,  thresh=.2, hier_thresh=.5, nms=.45):
    im = load_image(image_path, 0, 0)
    #imag = cv2.imread(image_path)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)

    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)

    res = []
    box_point = []
    box_list = []
    box_size = 0

    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0 and meta.names[i] == 'person':
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
                box_point.append([boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h])		

    for i in range(len(box_point)):
        box_list.append(box_point[i][2] * box_point[i][3])
    box = np.sort(box_list)
    for i in range(len(box)):
        if box[len(box) - 1] == box_list[i]:

            box_size = box_point[i][2] * box_point[i][3]
            #cv2.rectangle(imag, (int(box_point[i][0]-box_point[i][2]/2), int(box_point[i][1]-box_point[i][3]/2)),
                          #(int(box_point[i][0]-box_point[i][2]/2 + box_point[i][2]), int(box_point[i][1]-box_point[i][3]/2 + box_point[i][3])), (0, 255, 0), 5)
        #image_name = os.path.join('test_image/', image)
        #cv2.imwrite(image_name, imag)

    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    
    return box_size