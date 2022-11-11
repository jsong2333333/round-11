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
EXTRACTED_FILEDIR = '/scratch/jialin/image-classification-sep2022/projects/trigger_inversion/extracted_source'
COLOR_CHANNEL, RESOLUTION = 3, 256
METADATA = pd.read_csv(METADATA_FILEPATH)
NUM_SAMPLE_IMGS = 100
# with open(os.path.join(EXTRACTED_FILEDIR, 'class_to_model.json'), 'r') as outfile:
#     CLASS_TO_MODEL = json.load(outfile)

def num_to_id(num, type='model'):
    if type == 'model':
        return 'id-' + str(100000000+num)[1:]
    if type == 'img':
        return 'img-' + str(100000000+num)[1:] + '.jpg'

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

def generate_trigger(epsilon_m=5, epsilon_p=.2, seed=1, type='generic'):
    rng = np.random.default_rng(seed)

    if type == 'generic':
        mask = rng.uniform(0, 1, size=[1, RESOLUTION, RESOLUTION]).astype(np.float32)
        mask = mask / np.linalg.norm(mask) * epsilon_m

        pattern = rng.uniform(0, 1, size=[COLOR_CHANNEL, RESOLUTION, RESOLUTION]).astype(np.float32)
        pattern = pattern / np.linalg.norm(pattern) * epsilon_p

        mask = torch.from_numpy(mask)
        pattern = torch.from_numpy(pattern)
    return mask, pattern

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("trigger_inversion.py launched")

    # curr_valid_model_nums = np.load(os.path.join(EXTRACTED_FILEDIR, 'valid_less.npy')).tolist()

    # for model_num in [35]: #curr_valid_model_nums:
    #     model_id = num_to_id(model_num)
    for model_id in ['id-00000035', 'id-00000069', 'id-00000094', 'id-00000098', 'id-00000110', 'id-00000124', 'id-00000126', 'id-00000139', 'id-00000156']:
            # ['id-00000179', 'id-00000200', 'id-00000226', 'id-00000238', 'id-00000245', 'id-00000251', 'id-00000256', 'id-00000262', 'id-00000287']:
        logging.info(f'analyzing model {model_id}')

        num_classes = METADATA[METADATA['model_name'] == model_id]['number_classes'].item()
        with open(os.path.join(MODEL_FILEDIR, model_id, 'fg_class_translation.json'), 'r') as outfile:
            model_fg_class_trans = json.load(outfile)
        FOREGROUND_FILEDIR = os.path.join(MODEL_FILEDIR, model_id, 'foregrounds')
        # label_to_clean_model_ids = {}
        label_to_clean_image = {}
        for k, v in model_fg_class_trans.items():
            class_num = v.split('.')[0]
            # label_to_clean_model_ids[k] = CLASS_TO_MODEL[class_num]['0']
            # if len(CLASS_TO_MODEL[class_num]['0']) == 0:
            #     print(k)
            # label_to_clean_image[k] = process_img(os.path.join(FOREGROUND_FILEDIR, v), resize=True)
            clean_images = []
            rand_img_nums = np.random.permutation(100)[:NUM_SAMPLE_IMGS]
            for img_num in rand_img_nums:
                clean_images.append(process_img(os.path.join(CLEAN_SAMPLE_IMG_DIR, class_num, num_to_id(img_num, type='img'))))
            label_to_clean_image[k] = torch.stack(clean_images, dim=0)
        # initial_triggers = {i:generate_trigger(epsilon_m=20, epsilon_p=1, seed=i) for i in range(num_classes**2)}

        model_filepath = os.path.join(MODEL_FILEDIR, model_id, 'model.pt')
        suspicious_model = torch.load(model_filepath)
        
        # clean_models, clean_model_class_label = {}, {}
        # for k, v in label_to_clean_model_ids.items():
        #     clean_model_id = v[0][0]
        #     clean_model_filepath = os.path.join(MODEL_FILEDIR, clean_model_id, 'model.pt')
        #     clean_models[k] = torch.load(clean_model_filepath)
        #     clean_model_class_label[k] = int(v[0][1])

        suspicious_model = suspicious_model.to(device)
        suspicious_model.eval()

        # loss_dict = {}
        trigger_dict = {}
        lambd = 1/1000
        # epoch = 100
        batch_size = 5
        for num in range(num_classes**2):
            src, tgt = num//num_classes, num%num_classes
            if num % 100 == 0:
                logging.info(f'running {num} out of {num_classes**2}')
            if src != tgt:
                # loss_dict[f'{str(src)}-{str(tgt)}'] = []

                mask, pattern = generate_trigger(epsilon_m=15, epsilon_p=.8, seed=num)
                mask, pattern = mask.requires_grad_(), pattern.requires_grad_()
                raw_input = label_to_clean_image[str(src)]
                raw_input = raw_input.requires_grad_()
                optimizer = torch.optim.Adam([pattern, mask], lr=.02, betas=(.5, .9))   #lr=.02

                # clean_model = clean_models[str(src)].to(device)
                # clean_model.eval()
                # clean_model_src_label =clean_model_class_label[str(src)]

                prev_loss = 1e8

                # for ind in range(epoch):
                for i in range(200):
                    optimizer.zero_grad()

                    ind = i % 20
                    input_with_trigger = (1-mask) * raw_input[ind*batch_size:(ind+1)*batch_size, :] + mask*pattern
                    # input_with_trigger = (1-mask) * raw_input + mask*pattern
                    input_with_trigger = torch.clamp(input_with_trigger, min=0, max=1).to(device)

                    suspicious_model_pred = suspicious_model(input_with_trigger)
                    suspicious_loss = F.cross_entropy(suspicious_model_pred, torch.tensor([tgt]*batch_size).to(device))
                    # suspicious_model_pred = suspicious_model(input_with_trigger.unsqueeze(0))
                    # suspicious_loss = F.cross_entropy(suspicious_model_pred, torch.tensor([tgt]).to(device))

                    loss_reg = torch.norm(mask, p=1)

                    # clean_model_pred = clean_model(input_with_trigger)
                    # clean_loss = F.cross_entropy(clean_model_pred, torch.tensor([clean_model_src_label]*batch_size).to(device))

                    total_loss = suspicious_loss + lambd*loss_reg #+ clean_loss
                    # loss_dict[f'{str(src)}-{str(tgt)}'].append(total_loss.item())

                    total_loss.backward()
                    optimizer.step()

                    del input_with_trigger, suspicious_model_pred #, clean_model_pred

                    with torch.no_grad():
                        mask[:] = torch.clamp(mask, min=0, max=1)
                        pattern[:] = torch.clamp(pattern, min=0, max=1)

                    # if np.abs(prev_loss - total_loss.item()) <= 1e-4:
                    #     logging.info(f'early stop as loss converged, at loss {total_loss.item()}')
                    #     break
                    # else:
                    #     prev_loss = total_loss.item()

                # trigger_dict[f'{str(src)}-{str(tgt)}'] = (mask.tolist(), pattern.tolist())
                trigger_dict[f'{str(src)}-{str(tgt)}'] = (mask*pattern).tolist()

                del mask, pattern, raw_input, optimizer #, clean_model
        del suspicious_model      

        logging.info('inversion process finished')

        # with open(os.path.join(EXTRACTED_FILEDIR, f'loss_dict_{model_id}.json'), 'w') as outfile:
        #     json.dump(loss_dict, outfile)
        with open(os.path.join(EXTRACTED_FILEDIR, 'sample_image_trigger',  f'trigger_dict_{model_id}.json'), 'w') as outfile:
            json.dump(trigger_dict, outfile)

        logging.info('saving process finished')