import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image, ImageFilter, ImageFile
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import sys

class Superbin(Dataset):

#classes = ['CAN', 'PET', 'RECYCLE', 'REUSE20', 'REUSE40', 'REUSE50', 'REUSE100', 'REUSE150', 'ETC', 'HAND']
    
    def __init__(self, is_train, transform=None):

        super(Superbin, self).__init__()
        self.data_path = r'./../../dataset/superbin_dataset/DET/'
        self.is_train = is_train
        self.transform = transform
# self.label_list = classes

        def _check_exist(self):
            print("Image Folder : {}".format(os.path.join(self.data_path, "Images")))
            print("Label Folder : {}".format(os.path.join(self.data_path, "Annotations")))

        if is_train == 1:
            with open(r'./../../dataset/superbin_dataset/DET/ImageSets/train.txt') as f:
                self.lines = f.readlines()
# print("lines:{}".format(lines))
            label_path = []
            img_path = []
            labels = []
            try:
                for line in self.lines:    
                    line = line.strip()
                    #print("line:{}\n".format(line))
                    label_path.append(os.path.join(r'./../../dataset/superbin_dataset/DET/Annotations', line+'.xml'))
                    img_path.append(os.path.join(r'./../../dataset/superbin_dataset/DET/Images', line + '.jpg'))
                    #pNameError: name 'root' is not definedrint("ann_path:{}\n".format(ann_path))
                    #doc = ET.parse(ann_path)
                    #print("doc:{}\n".format(doc))
                    #root = doc.getroot()
                    #print("root:{}\n".format(root))
            
                self.img_path = img_path
                labels=[]
                for ann_idx, ann_path in enumerate(label_path):
                    label = [0, 0, 0, 0]
                    xml_file = label_path[ann_idx]
                    doc = ET.parse(xml_file)
                    root = doc.getroot()
              
                    for object in root.findall("object"):
                        cls_name = object.find("name").text
                        cls_idx = cls_name.split('_')[0]
                       
                        if cls_idx == '0':
                            label[0] = 1	
                        elif cls_idx == '1':
                            label[1] = 1
                        elif cls_idx == '2':
                            label[2] = 1
                        elif cls_idx == '3':
                            label[3] = 1
                        else:
                            pass
                    labels.append(label)
                self.label = np.array(labels)
            except:
                pass
                
        else:
            with open(r'./../../dataset/superbin_dataset/DET/ImageSets/val.txt') as f:
                self.lines = f.readlines()
# print("lines:{}".format(lines))
            label_path = []
            img_path = []
            labels = []
            try:
                for line in self.lines:
                    line = line.strip()
                    #print("line:{}\n".format(line))
                    label_path.append(os.path.join(r'./../../dataset/superbin_dataset/DET/Annotations', line+'.xml'))
                    img_path.append(os.path.join(r'./../../dataset/superbin_dataset/DET/Images', line + '.jpg'))
                    #pNameError: name 'root' is not definedrint("ann_path:{}\n".format(ann_path))
                    #doc = ET.parse(ann_path)
                    #print("doc:{}\n".format(doc))
                    #root = doc.getroot()
                    #print("root:{}\n".format(root))

                self.img_path = img_path
                labels=[]
                for ann_idx, ann_path in enumerate(label_path):
                    label = [0, 0, 0, 0]
                    xml_file = label_path[ann_idx]
                    doc = ET.parse(xml_file)
                    root = doc.getroot()

                    for object in root.findall("object"):
                        cls_name = object.find("name").text
                        cls_idx = cls_name.split('_')[0]

                        if cls_idx == '0':
                            label[0] = 1
                        elif cls_idx == '1':
                            label[1] = 1
                        elif cls_idx == '2':
                            label[2] = 1
                        elif cls_idx == '3':
                            label[3] = 1
                        else:
                            pass
                    labels.append(label)
                self.label = np.array(labels)
            except:
                pass

    def __len__(self):
        return len(self.lines)


    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        path = self.img_path[index]
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = torch.from_numpy(self.label[index])
       # print(f"label_idx{label_idx}")
            
        return image, label_idx #, img_path

