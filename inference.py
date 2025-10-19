import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color

from model import UNET
from Data_Loader import ADE20KDataset 


### Load Model ###
model = UNET(skip_connection=True, num_classes=150)
model.load_state_dict(torch.load("work_dir/UNET_wo_skip_ADE20K_test/best_checkpoint/pytorch_model.bin"))

### Load Dataset ###
path_to_data = "archive/ADEChallengeData2016"
test_data = ADE20KDataset(path_to_data, train=False, inference_mode=True)
test_roots = test_data.file_roots



def inference_image(model, path_to_image, path_to_annotation, colors, image_size=256):

    ### Set Preprocessing Steps ###
    resize = transforms.Resize(size=(image_size, image_size))
    normalize = transforms.Normalize(mean=(0.48897059, 0.46548275, 0.4294), 
                                     std=(0.22861765, 0.22948039, 0.24054667))
    totensor = transforms.ToTensor()
    topil = transforms.ToPILImage()

    ### Load Images ###
    image = Image.open(path_to_image).convert("RGB")
    annot = Image.open(path_to_annotation)
    width, height = annot.size 
    annot = np.array(annot) # This goes from 0 to 150 by default 

    ### Prepare Image for Inference ###
    proc_image = normalize(totensor(resize(image))).unsqueeze(0)

    ### Inference Model ###
    with torch.no_grad():
        pred = model(proc_image)
        
    ### Process Inference ###
    pred = pred.argmax(axis=1).cpu().unsqueeze(0)
    pred = F.interpolate(pred.type(torch.FloatTensor), size=(height, width)).squeeze() 
    pred = np.array(pred, dtype=np.uint8)

    ### Predictions are from 0-149, but 0 is the background class in our data, so add one so it goes 1-150 instead ###
    pred = pred + 1

    ### Overlay Masks on Images ###
    annot_color = color.label2rgb(annot,np.array(image), colors)
    pred_color = color.label2rgb(pred,np.array(image), colors)

    ### Plot Results! ###
    fig, ax = plt.subplots(1,2, figsize=(15,8))
    ax[0].imshow(annot_color)
    ax[0].set_title("Ground Truth Segmentation")
    ax[0].axis("off")
    ax[1].imshow(pred_color)
    ax[1].set_title("Predicted Segmentation")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()



color_pallete = np.random.random((150, 3))

import random
for idx in list(range(10)):
    path_to_image = os.path.join(test_data.path_to_images, test_roots[idx]+".jpg")
    path_to_annotation = os.path.join(test_data.path_to_annotations, test_roots[idx]+".png")
    
    inference_image(model, path_to_image, path_to_annotation, color_pallete)