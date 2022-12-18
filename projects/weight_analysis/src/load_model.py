import logging
import torch
import os
import json
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import sys
from torchvision import datasets, transforms as T 
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_FILEDIR = '/scratch/data/TrojAI/image-classification-sep2022-train/models/'
CLEAN_SAMPLE_IMG_DIR = '/scratch/data/TrojAI/image-classification-sep2022-train/image-classification-sep2022-example-source-dataset'
METADATA_FILEPATH = '/scratch/data/TrojAI/image-classification-sep2022-train/METADATA.csv'
MODEL_ARCH = ['classification:' + arch for arch in ['resnet50', 'vit_base_patch32_224', 'mobilenet_v2']]
NUM_MODEL = 288
OUTPUT_FILEDIR = '/scratch/jialin/image-classification-sep2022/projects/trigger_inversion/extracted_source/'
EXTRACTED_FILEDIR = '/scratch/jialin/image-classification-sep2022/projects/trigger_inversion/extracted_source/'
COLOR_CHANNEL, RESOLUTION = 3, 256
METADATA = pd.read_csv(METADATA_FILEPATH)
with open(os.path.join(EXTRACTED_FILEDIR, 'class_to_model.json'), 'r') as outfile:
    CLASS_TO_MODEL = json.load(outfile)

def num_to_model_id(num):
    return 'id-' + str(100000000+num)[1:]

resize_transforms = T.Resize(size=(66, 66))
padding_transforms = T.Pad(padding=(256-66)//2, padding_mode='constant', fill=0)
augmentation_transforms = T.Compose([T.ConvertImageDtype(torch.float)])
# randomize on position and size of foreground, see if background matters

def process_img(img_filepath, resize=False):
    img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = torch.as_tensor(img)
    image = image.permute((2, 0, 1))
    if resize: 
        image = resize_transforms(image)
        image = padding_transforms(image)
    image = augmentation_transforms(image)
    return image

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("asr.py launched")

    model_id = num_to_model_id(2)

    logging.info(f"testing asr of model {model_id}")

    model = torch.load(os.path.join(MODEL_FILEDIR, model_id, 'model.pt'))
    model.eval()

    print(model_id)