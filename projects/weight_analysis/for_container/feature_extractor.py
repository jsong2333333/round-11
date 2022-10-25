import torch
import os
import numpy as np
import json

DATA_PATH = '/scratch/data/TrojAI/image-classification-sep2022-train/models/'
MODEL_ARCH = ['resnet50', 'vit_base_patch32_224', 'mobilenet_v2']
WEIGHT_LENGTH_TO_MODEL_ARCH = {966: 'resnet50', 912: 'vit_base_patch32_224', 948:'mobilenet_v2'}
MODEL_ARCH_TO_FEATURE_LENGTH = {'resnet50': 1236, 'vit_base_patch32_224': 1172, 'mobilenet_v2': 1213}
# OPTIMAL_LAYERS = {'resnet50': [48, 49], 'vit_base_patch32_224': [2, 25], 'mobilenet_v2': [35, 7]}


def get_features_and_labels_by_model_class(dp, model_class):
    ret_dir = {'X': [], 'y': []}
    child_dirs = os.listdir(dp)
    for child_dir in child_dirs:
        model_filedir = os.path.join(dp, child_dir)
        if os.path.isdir(model_filedir):
            model_filepath = os.path.join(model_filedir, 'model.pt')
            json_filepath = os.path.join(model_filedir, 'config.json')
            model_arch = _get_model_arch(json_filepath)
            if model_class == model_arch:
                model_features = _get_weight_features(model_filepath)
                # if model_class in MODEL_ARCH[:2]:
                model_eigens = _get_eigen_features(model_filepath)
                model_features += model_eigens
                model_label = _get_model_label(json_filepath)
                ret_dir['X'].append(model_features)
                ret_dir['y'].append(model_label)
    for k, v in ret_dir.items():
        ret_dir[k] = np.asarray(v)
    return ret_dir


def get_predict_model_features_and_class(model_filepath):
    weight_features = _get_weight_features(model_filepath)
    model_class = WEIGHT_LENGTH_TO_MODEL_ARCH[len(weight_features)]
    # if model_class in MODEL_ARCH[:2]:
    eigen_features = _get_eigen_features(model_filepath)
    weight_features += eigen_features
    return model_class, weight_features


def _get_model_arch(json_filepath) -> bool:
    with open(json_filepath, 'r') as f:
        config = json.loads(f.read())
    return config['py/state']['model_architecture'][15:]


def _get_model_label(json_filepath) -> bool:
    with open(json_filepath, 'r') as f:
        config = json.loads(f.read())
    return config['py/state']['poisoned']


def _get_weight_features(model_filepath):
    with torch.no_grad():
        model = torch.load(model_filepath)
    params = []
    for name, param in model.named_parameters():
        if "3d" in name:
            axis = tuple(np.arange(len(param.shape)-1).tolist())
            params += torch.amax(param, dim=axis).flatten().tolist()
            params += torch.mean(param, dim=axis).flatten().tolist()
            sub = torch.mean(param, dim=axis).flatten() - torch.median(torch.flatten(param, end_dim=-2), dim=0).values.flatten()
            params += sub.tolist()
            params += torch.median(torch.flatten(param, end_dim=-2), dim=0).values.flatten().tolist()
            params += torch.sum(param, dim=axis).flatten().tolist()
            params += (torch.linalg.norm(param, ord='fro', dim=axis).flatten()**2/torch.linalg.norm(param, ord=2, dim=axis).flatten()**2).tolist()
        else:
            params.append(param.max().tolist())
            params.append(param.mean().tolist())
            sub = param.mean() - torch.median(param)
            params.append(sub.tolist())
            params.append(torch.median(param).tolist())
            params.append(param.sum().tolist())
            params.append((torch.linalg.norm(param.reshape(param.shape[0], -1), ord='fro')**2/torch.linalg.norm(param.reshape(param.shape[0], -1), ord=2)**2).tolist())
    return params


def _get_eigen_features(model_filepath):
    with torch.no_grad():
        model = torch.load(model_filepath)
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


def _get_optimal_eigens(model_filepath, optimal_layers=[]):
    with torch.no_grad():
        model = torch.load(model_filepath)
    eigens = []
    min_shape, layer_num = 1, 0
    for param in model.parameters():
        if len(param.shape) > min_shape:
            layer_num += 1
            if layer_num in optimal_layers:
                reshaped_param = param.reshape(param.shape[0], -1)
                singular_values = torch.linalg.svd(reshaped_param, False).S
                squared_singular_values = torch.square(singular_values)
                ssv = squared_singular_values.tolist()
                eigens += ssv
    eigens = np.array_split(eigens, 120)
    params = []
    for eig in eigens:
        if eig.shape[-1] != 0:
            params.append(eig.max())
            params.append(eig.mean())
            params.append((eig.mean() - np.median(eig)))
            params.append(np.median(eig))
            params.append(eig.sum())
    return params


if __name__ =='__main__':
    EXTRACTED_DIR = '/scratch/jialin/image-classification-sep2022/projects/weight_analysis/for_container/learned_parameters/'
    for model_arch in MODEL_ARCH:
        trainig_models_class_and_features = get_features_and_labels_by_model_class(DATA_PATH, model_arch)
        X = trainig_models_class_and_features['X']
        y = trainig_models_class_and_features['y']
        np.save(os.path.join(EXTRACTED_DIR, f'train_X_{model_arch}.npy'), X)
        np.save(os.path.join(EXTRACTED_DIR, f'train_y_{model_arch}.npy'), y)
        print(X.shape, y.shape, model_arch)