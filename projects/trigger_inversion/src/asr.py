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

    model_id = num_to_model_id(35)

    logging.info(f"testing asr of model {model_id}")

    with open(os.path.join(EXTRACTED_FILEDIR, 'sample_image_trigger', f'trigger_dict_{model_id}.json'), 'r') as outfile:
        triggers = json.load(outfile)
    with open(os.path.join(MODEL_FILEDIR, model_id, 'fg_class_translation.json'), 'r') as outfile:
        model_fg_class_trans = json.load(outfile)
    clean_images_filedir = os.path.join(MODEL_FILEDIR, model_id, 'clean-example-data')
    label_to_clean_images = {}

    # for k, v in model_fg_class_trans.items():
    #     label_to_clean_images[k] = process_img(os.path.join(clean_images_filedir, v), resize=True)

    for img_fp in os.listdir(clean_images_filedir):
        if img_fp.endswith('.png'):
            img_id = img_fp[:-4]
            clean_image = process_img(os.path.join(clean_images_filedir, img_fp))
            with open(os.path.join(clean_images_filedir, f'{img_id}.json')) as outfile:
                label = json.load(outfile)
            if label in label_to_clean_images:
                label_to_clean_images[label].append(clean_image)
            else:
                label_to_clean_images[label] = [clean_image]
    for k, v in label_to_clean_images.items():
        label_to_clean_images[k] = torch.stack(v, dim=0)


    all_poisoned_images, all_target_labels, all_source_labels = [], [], []
    for source_class, clean_images in label_to_clean_images.items():
        # for k, (m, p) in triggers.items():
        #     src, tgt = k.split('-')[0], k.split('-')[1]
        #     if int(src) == source_class:
        #         m, p = torch.tensor(m), torch.tensor(p)
        #         poisoned_images = (1-m)*clean_images + m*p
        #         all_poisoned_images.append(poisoned_images)
        #         tgt = int(tgt)
        #         all_target_labels += [tgt]*clean_images.shape[0]
        #         all_source_labels += [source_class]*clean_images.shape[0]
        for k, t in triggers.items():
            src, tgt = k.split('-')[0], k.split('-')[1]
            if int(src) == source_class:
                t = torch.tensor(t)
                poisoned_images = t + clean_images*.018
                poisoned_images = torch.clamp(poisoned_images, 0, 1)
                all_poisoned_images.append(poisoned_images)
                tgt = int(tgt)
                all_target_labels += [tgt]*clean_images.shape[0]
                all_source_labels += [source_class]*clean_images.shape[0]

    all_poisoned_images = torch.concat(all_poisoned_images, dim=0)
    all_target_labels = torch.tensor(all_target_labels)

    model = torch.load(os.path.join(MODEL_FILEDIR, model_id, 'model.pt')).to(device)
    model.eval()

    batch_size, acc, count = 20, 0, 0
    for i in range(all_poisoned_images.shape[0]//batch_size):
        minibatch = all_poisoned_images[i*batch_size:(i+1)*batch_size, :].to(device)
        minilabel = all_target_labels[i*batch_size:(i+1)*batch_size].to(device)

        logits = model(minibatch)
        acc += (torch.argmax(logits, dim=1) == minilabel).float().sum(0).item()
        count += minibatch.shape[0]
    
    logging.info(f'the asr rate is {acc/count}')
