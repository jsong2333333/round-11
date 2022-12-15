import torch
import os
import numpy as np
import json
from itertools import product
import cv2
from torchvision import transforms as T
import torch.nn.functional as F
import joblib
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = '/scratch/data/TrojAI/image-classification-sep2022-train/models/'
MODEL_ARCH = ['resnet50', 'vit_base_patch32_224', 'mobilenet_v2']
WEIGHT_LENGTH_TO_MODEL_ARCH = {978: 'resnet50', 924: 'vit_base_patch32_224', 960:'mobilenet_v2'}
MODEL_ARCH_TO_FEATURE_LENGTH = {'resnet50': 1248, 'vit_base_patch32_224': 3184, 'mobilenet_v2': 3225}
MODEL_ARCH_FOR_LOSS_CALCULATION = MODEL_ARCH[1:]


def get_features_and_labels_by_model_class(dp, model_class, trigger_filedir, device):
    ret_dir = {'X': [], 'y': []}

    poly_trigger = torch.from_numpy(np.load(os.path.join(trigger_filedir, 'all_train_triggers.npy')))
    filter_trigger = torch.from_numpy(np.load(os.path.join(trigger_filedir, 'rand_num_for_filter.npy')))

    child_dirs = os.listdir(dp)
    for child_dir in child_dirs:
        model_filedir = os.path.join(dp, child_dir)
        if os.path.isdir(model_filedir):
            model_filepath = os.path.join(model_filedir, 'model.pt')
            json_filepath = os.path.join(model_filedir, 'config.json')
            example_dirpath = os.path.join(model_filedir, 'clean-example-data')
            model_arch = _get_model_arch(json_filepath)
            if model_class == model_arch:
                with torch.no_grad():
                    model = torch.load(model_filepath)
                model_features = _get_weight_features(model)
                model_features += _get_eigen_features(model)
                if model_class in MODEL_ARCH_FOR_LOSS_CALCULATION:
                    model_features += _get_loss_features(model, example_dirpath, poly_trigger, filter_trigger, device)
                model_label = _get_model_label(json_filepath)
                ret_dir['X'].append(model_features)
                ret_dir['y'].append(model_label)
    for k, v in ret_dir.items():
        ret_dir[k] = np.asarray(v)
    return ret_dir


def get_predict_model_features_and_class(model_filepath, example_filedir, trigger_filedir, device):
    with torch.no_grad():
        model = torch.load(model_filepath)
    features = _get_weight_features(model)
    model_class = WEIGHT_LENGTH_TO_MODEL_ARCH[len(features)]
    features += _get_eigen_features(model)
    if model_class in MODEL_ARCH_FOR_LOSS_CALCULATION:
        poly_trigger = torch.from_numpy(np.load(os.path.join(trigger_filedir, 'all_train_triggers.npy')))
        filter_trigger = torch.from_numpy(np.load(os.path.join(trigger_filedir, 'rand_num_for_filter.npy')))
        features += _get_loss_features(model, example_filedir, poly_trigger, filter_trigger, device)
    return model_class, features


def extract_stats(loss_arr, tri_size=164, loop_range=13):
    stats = []
    for i in range(loop_range):
        if i >= 9:
            start_from = 9*tri_size
            interval = (loss_arr.shape[-1] - start_from)//(loop_range-9)
            extracted_loss = loss_arr[:, start_from+(i-9)*interval:start_from+(i-9+1)*interval]
        else:
            ind = [i+9*j for j in range(tri_size)]
            extracted_loss = loss_arr[:, ind]
        stats.extend([np.mean(extracted_loss), np.median(extracted_loss), np.std(extracted_loss), np.max(extracted_loss)])
    return stats


def process_img(img_filepath, resize=False, resize_res=16, padding=False, padding_pos='middle', flat=True):
    img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = torch.as_tensor(img)
    image = image.permute((2, 0, 1))
    images = None
    if resize:
        resize_transforms = T.Resize(size=(resize_res, resize_res))
        image = resize_transforms(image)
    if padding:
        img_size = resize_res if resize else image.shape[0]
        if padding_pos == 'middle':
            p_trans = T.Pad(padding=(256-img_size)//2, padding_mode='constant', fill=0)
            image = p_trans(image)
        else:
            padding_transforms = []
            padding_slot = int((256-3*img_size)/6)
            mid = int((256-img_size)/2)
            positions = [mid-img_size-2*padding_slot, mid, mid+img_size+2*padding_slot]
            for left, top in list(product(positions, positions)):
                p_trans = T.Pad(padding=(left, top, 256-img_size-left, 256-img_size-top), padding_mode='constant', fill=0)
                padding_transforms.append(p_trans)
            if flat:
                canvas = torch.zeros(image.shape[0], 256, 256)
                for p_trans in padding_transforms:
                    canvas += p_trans(image)
                image = canvas
            else:
                images = []
                for p_trans in padding_transforms:
                    images.append(p_trans(image))
    augmentation_transforms = T.Compose([T.ConvertImageDtype(torch.float)]) 
    if images:
        return [augmentation_transforms(image) for image in images]
    return augmentation_transforms(image)


def _get_model_arch(json_filepath) -> bool:
    with open(json_filepath, 'r') as f:
        config = json.loads(f.read())
    return config['py/state']['model_architecture'][15:]


def _get_model_label(json_filepath) -> bool:
    with open(json_filepath, 'r') as f:
        config = json.loads(f.read())
    return config['py/state']['poisoned']


def _get_weight_features(model):
    params = []
    for param in model.parameters():
        if list(param.shape) in ([32, 3, 3, 3], [768, 3, 32, 32], [64, 3, 7, 7]):
            param_3d = torch.flatten(torch.permute(param, (1, 0, 2, 3)), start_dim=2)
            axis = (-1, -2)
            params += torch.amax(param_3d, dim=axis).tolist()
            params += torch.mean(param_3d, dim=axis).tolist()
            params += (torch.mean(param_3d, dim=axis) - torch.median(torch.flatten(param_3d, start_dim=1), dim=-1)[0]).tolist()
            params += torch.median(torch.flatten(param_3d, start_dim=1), dim=-1)[0].tolist()
            params += torch.sum(param_3d, dim=axis).tolist()
            params += (torch.linalg.norm(param_3d, ord='fro', dim=(-1, -2))**2/torch.linalg.norm(param_3d, ord=2, dim=(-1, -2))**2).tolist()
        else:
            params.append(param.max().tolist())
            params.append(param.mean().tolist())
            sub = param.mean() - torch.median(param)
            params.append(sub.tolist())
            params.append(torch.median(param).tolist())
            params.append(param.sum().tolist())
            params.append((torch.linalg.norm(param.reshape(param.shape[0], -1), ord='fro')**2/torch.linalg.norm(param.reshape(param.shape[0], -1), ord=2)**2).tolist())
    return params


def _get_eigen_features(model):
    min_shape, params = 1, []
    for param in model.parameters():
        if len(param.shape) > min_shape:
            reshaped_param = param.reshape(param.shape[0], -1)
            singular_values = torch.linalg.svd(reshaped_param, False).S
            ssv = torch.square(singular_values)
            params.append(ssv.max().tolist())
            params.append(ssv.mean().tolist())
            params.append((ssv.mean() - torch.median(ssv)).tolist())
            params.append(torch.median(ssv).tolist())
            params.append(ssv.sum().tolist())
    return params


def _get_loss_features(model, example_filedir, all_train_triggers, rand_num_for_filter, device, batch_size=25):
    model = model.to(device)
    model.eval()
    trigger_size, filter_size = all_train_triggers.shape[0], rand_num_for_filter.shape[0]
    losses = []
    for clean_image in os.listdir(example_filedir):
        if clean_image.endswith('png'):
            processed_image = process_img(os.path.join(example_filedir, clean_image))
            with open(os.path.join(example_filedir, f'{clean_image[:-4]}.json')) as outfile:
                label = json.load(outfile)
            expanded_image = processed_image.expand(batch_size, -1, -1, -1)
            loss_per_img = []
            for i in range((trigger_size+filter_size)//batch_size):
                s_ind, e_ind = batch_size*i, batch_size*(i+1)
                if e_ind < trigger_size:
                    polygon_triggers = all_train_triggers[s_ind:e_ind, :]
                    images = torch.where(polygon_triggers==0, expanded_image, polygon_triggers)
                elif s_ind < trigger_size and e_ind > trigger_size:
                    polygon_triggers = all_train_triggers[s_ind:, :]
                    poly_images = torch.where(polygon_triggers==0, expanded_image[polygon_triggers.shape[0], :], polygon_triggers)
                    filter_triggers = rand_num_for_filter[:e_ind-trigger_size, :]
                    filter_images = torch.clip(filter_triggers*expanded_image[batch_size-filter_triggers.shape[0], :], 0, 1)
                    images = torch.concat([poly_images, filter_images], dim=0)
                else:
                    filter_triggers = rand_num_for_filter[s_ind-trigger_size:e_ind-trigger_size, :]
                    images = torch.clip(filter_triggers*expanded_image, 0, 1)
                with torch.no_grad():
                    logits = model(images.to(device))
                    loss_per_batch = F.cross_entropy(logits, torch.tensor([label]*batch_size).to(device), reduction='none').tolist()
                    loss_per_img += loss_per_batch
            losses.append(loss_per_img)
    return np.concatenate([np.mean(losses, axis=0), np.median(losses, axis=0), np.std(losses, axis=0), np.max(losses, axis=0)]).tolist()


if __name__ =='__main__':
    EXTRACTED_DIR = '/scratch/jialin/image-classification-sep2022/projects/weight_analysis/for_container/learned_parameters/'
    clf_dict = {MODEL_ARCH[0]: GradientBoostingClassifier(learning_rate=0.015, n_estimators=900, max_depth=3, min_samples_split=24, min_samples_leaf=16, max_features=120),
                MODEL_ARCH[1]: GradientBoostingClassifier(learning_rate=0.01, n_estimators=750, max_depth=3, min_samples_split=40, min_samples_leaf=4, max_features=32),
                MODEL_ARCH[2]: GradientBoostingClassifier(learning_rate=0.011, n_estimators=500, max_depth=4, min_samples_split=34, min_samples_leaf=16, max_features=96)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    for model_arch in MODEL_ARCH:
        trainig_models_class_and_features = get_features_and_labels_by_model_class(DATA_PATH, model_arch, EXTRACTED_DIR, device)
        X = trainig_models_class_and_features['X']
        y = trainig_models_class_and_features['y']
        np.save(os.path.join(EXTRACTED_DIR, f'train_X_{model_arch}.npy'), X)
        np.save(os.path.join(EXTRACTED_DIR, f'train_y_{model_arch}.npy'), y)
        clf = clf_dict[model_arch].fit(X, y)
        joblib.dump(clf, os.path.join(EXTRACTED_DIR, f'original_{model_arch}_clf.joblib'))
        print(X.shape, y.shape, model_arch)