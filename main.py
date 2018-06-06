import os
from detect import detect, remove_object_detect

input_drama = raw_input('drama name : ')
result_txt = input_drama + '.txt'

result_file = open(result_txt, 'w')
model_file =  ''#'../models/korea_221.pth.tar'
label_txt = ''#'../labels/221_categories.txt'
image_path = os.path.join('drama_image', input_drama)

detect(model_file, label_txt, image_path, result_file)
#remove_object_detect(model_file, label_txt, image_path, result_file)
