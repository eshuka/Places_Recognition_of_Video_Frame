import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os, cv2
import numpy as np
from PIL import Image
from darknet import load, darknet

def get_top_k_result(similar_list, k):
    result = (sorted(similar_list, key=lambda l: l[1], reverse=True))
    return result[0:k+1]


def accuracy(top5_lists, remove_image_num):

    fileGT = 'drama_image/high_kick.txt'
    #fileGT = 'drama_image/sound_of_mind_1.txt'

    with open(fileGT,'r') as f:
        lines = f.readlines()
    truthVector = []
    for line in lines:
        items = line.split()
        truthVector.append(int(items[0]))
    for i in range(len(remove_image_num), 0, -1):
        truthVector.pop(i)

    predictionVector = []
    predictionVector_top5 = []
    for line in top5_lists:
        predictionVector.append(int(line[0]))
        if len(line)==5:
            predictionVector_top5 = top5_lists

    n_classes = 221
    confusionMat = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predictionVector, truthVector):
        confusionMat[pred][exp] += 1
    t = sum(sum(l) for l in confusionMat)

    accuracy = sum(confusionMat[i][i] for i in range(len(confusionMat)))*1.0 / t


    top5error = 'NA'
    if len(predictionVector_top5) == len(truthVector):
        top5error = 0
        for i, curPredict in enumerate(predictionVector_top5):
            curTruth = truthVector[i]
            curHit = [1 for label in curPredict if label==curTruth]
            if len(curHit)==0:
                top5error = top5error+1
        top5error = top5error*1.0/len(truthVector)
            
    print ("accuracy:" + str(accuracy))
    print ("top 5 error rate:" + str(top5error))


def detect(model_file, label_txt, image_path, result_file):

    model = torch.load(model_file)

    model.eval()

    centre_crop = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classes = list()
    with open(label_txt) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][0:])
    classes = tuple(classes)

    image_list = os.listdir(image_path)
    image_list = np.sort(image_list)

    result_ids = []

    for image_name in image_list:
        image = os.path.join(image_path, image_name)
        img = Image.open(image)
        img = img.convert('RGB')
        cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = img.resize((224,224), Image.ANTIALIAS)
        input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        print('RESULT ON ' + image_name)
        a = []

        for i in range(0, 5):
            print(probs[i], classes[idx[i]])
            cv2.putText(cv2_image, str(classes[idx[i]]) + ' ' + str(probs[i]), (10, 28 * (i+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            a.append(idx[i])
        result_ids.append(a)
        #print('result' + image_path.split('/')[1])

        save_image_path = os.path.join('result',image_path.split('/')[1])
        save_image_name = os.path.join(save_image_path, image_name)
        cv2.imwrite(save_image_name, cv2_image)
    accuracy(result_ids, remove_image_num = [])


def remove_object_detect(model_file, label_txt, image_path, result_file):
    net, meta = load.load_darknet()    
    model = torch.load(model_file)

    model.eval()

    centre_crop = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classes = list()
    with open(label_txt) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][0:])
    classes = tuple(classes)

    image_list = os.listdir(image_path)
    image_list = np.sort(image_list)

    result_ids = []
    remove_image_num = []
    image_count = 0

    for image_name in image_list:
        image = os.path.join(image_path, image_name)
        img = Image.open(image)
        img = img.convert('RGB')
        cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        width, height, channel = np.shape(cv2_image)
        image_size = width * height
        img = img.resize((224,224), Image.ANTIALIAS)
        
        box_size = darknet.object_detect(net, meta, image, image_name)

        if box_size / image_size <= 0.2:
            input_img = V(centre_crop(img).unsqueeze(0), volatile=True)
 
            logit = model.forward(input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)

            print('RESULT ON ' + image_name)
            a = []

            for i in range(0, 5):
                print(classes[idx[i]])
                cv2.putText(cv2_image, str(classes[idx[i]]) + ' ' + str(probs[i]), (10, 28 * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                a.append(idx[i])

                save_image_path = os.path.join('remove_result', image_path.split('/')[1])
                save_image_name = os.path.join(save_image_path, image_name)
                cv2.imwrite(save_image_name, cv2_image)

            result_ids.append(a)
        else:
            remove_image_num.append(image_count)
	image_count += 1
    print('remove image : ', remove_image_num)
    accuracy(result_ids, remove_image_num)


