import torch
import os
import numpy as np
import json

DATA_PATH = '/scratch/data/TrojAI/image-classification-sep2022-train/models/'
MODEL_ARCH = ['resnet50', 'vit_base_patch32_224', 'mobilenet_v2']
WEIGHT_LENGTH_TO_MODEL_ARCH = {978: 'resnet50', 924: 'vit_base_patch32_224', 960:'mobilenet_v2'}
MODEL_ARCH_TO_FEATURE_LENGTH = {'resnet50': 1300, 'vit_base_patch32_224': 1236, 'mobilenet_v2': 1277}


def get_features_and_labels_by_model_class(dp, model_class):
    ret_dir = {'X': [], 'y': [], 'model_id': []}
    child_dirs = os.listdir(dp)
    for child_dir in child_dirs:
        model_filedir = os.path.join(dp, child_dir)
        if os.path.isdir(model_filedir):
            model_filepath = os.path.join(model_filedir, 'model.pt')
            json_filepath = os.path.join(model_filedir, 'config.json')
            model_arch = _get_model_arch(json_filepath)
            if model_class == model_arch:
                model_features = _get_weight_features(model_filepath)
                model_eigens = _get_eigen_features(model_filepath)
                model_features += model_eigens
                model_label = _get_model_label(json_filepath)
                ret_dir['X'].append(model_features)
                ret_dir['y'].append(model_label)
                ret_dir['model_id'].append(child_dir)
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
    EXTRACTED_DIR = '/scratch/jialin/image-classification-sep2022/projects/weight_analysis/extracted_source'
    for model_arch in [MODEL_ARCH[0]]:
        trainig_models_class_and_features = get_features_and_labels_by_model_class(DATA_PATH, model_arch)
        X = trainig_models_class_and_features['X']
        y = trainig_models_class_and_features['y']
        model_id = trainig_models_class_and_features['model_id']
        np.save(os.path.join(EXTRACTED_DIR, f'train_X_{model_arch}.npy'), X)
        np.save(os.path.join(EXTRACTED_DIR, f'train_y_{model_arch}.npy'), y)
        np.save(os.path.join(EXTRACTED_DIR, f'train_model_id_{model_arch}.npy'), model_id)
        print(X.shape, y.shape, model_arch)