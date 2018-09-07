#!/usr/bin/env python2
import os
import torch
import glob
import json
import torch.utils.data
from torchvision import transforms, utils
from PIL import Image



class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dirs,transform = None):

        self.transform = transform
        img_paths = []

        for data_dir in data_dirs:
            #find every image in the data directory
            img_paths.extend(glob.glob(os.path.join(data_dir,"*.png")))
        #sort file names in alphabetical order
        img_paths.sort()

        #create an empty list to hold the image
        self.img_annotation_path_pairs = []

        #for each image file get the annotation file
        for img_path in img_paths:
            annotation_path = img_path.replace("img.png","gt.txt")
            #if the annotation file exists then append the path pairs
            if os.path.exists(annotation_path):
                self.img_annotation_path_pairs.append((img_path,annotation_path))

        print("Data Dir: %s" % data_dir)
        print("Annotation pairs: %s" % len(self.img_annotation_path_pairs))

        #if the transform is None, compose this default one
        if self.transform == None:
            self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.img_annotation_path_pairs)

    def __getitem__(self,index):
        #get the path pairs at this index
        img_path,annotation_path = self.img_annotation_path_pairs[index]

        #load the image as a PIL image
        img_pil = Image.open(img_path)

        #use the composed transform to get the image tensor
        img_tensor = self.transform(img_pil)


        #load the annotation json file
        with open(annotation_path,"rb") as f:
            annotation_dict = json.load(f)


        #put annotation data into a list and convert into tensor
        force_tensor = torch.Tensor([
        annotation_dict["ft_x"],
        annotation_dict["ft_y"],
        annotation_dict["ft_z"],
        annotation_dict["ft_wx"],
        annotation_dict["ft_wy"],
        annotation_dict["ft_wz"]
        ])

        #put annotation data into a list and convert into tensor
        annotation_tensor = torch.Tensor([
        annotation_dict["gripper_open"]
        ])

        #create the sample dictionary
        sample = {"image": img_tensor, "force": force_tensor, "annotation": annotation_tensor}

        return sample



if __name__ == "__main__":
    d = Dataset(["/home/acrv/data/visual_servoing/handover"])
    print(d[4])
