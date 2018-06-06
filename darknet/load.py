from ctypes import *

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("/media/mmlab/hdd/Rock/darknet/python/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int


load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p


load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

#set_gpu = lib.cuda_set_device
#set_gpu.argtypes = [c_int]

def load_darknet():
    #set_gpu(1)
    net = load_net("darknet/cfg/yolo.cfg", "darknet/yolo.weights", 0)
    meta = load_meta("darknet/cfg/coco.data")

    return net, meta


