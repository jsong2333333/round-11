from sklearn.ensemble import GradientBoostingClassifier
import joblib
import numpy as np
import os

MODEL_ARCH = ['classification:' + arch for arch in ['resnet50', 'vit_base_patch32_224', 'mobilenet_v2']]
clf_dict = {MODEL_ARCH[0]: GradientBoostingClassifier(learning_rate=0.015, n_estimators=900, max_depth=3, min_samples_split=24, min_samples_leaf=16, max_features=120),
            MODEL_ARCH[1]: GradientBoostingClassifier(learning_rate=0.01, n_estimators=750, max_depth=3, min_samples_split=40, min_samples_leaf=4, max_features=32),
            MODEL_ARCH[2]: GradientBoostingClassifier(learning_rate=0.011, n_estimators=500, max_depth=4, min_samples_split=34, min_samples_leaf=16, max_features=96)}

LEARNED_PARAM = '/scratch/jialin/image-classification-sep2022/projects/weight_analysis/for_container/learned_parameters'

# features = {}
# for model_arch in MODEL_ARCH:
#     features[model_arch] = (np.load(os.path.join(LEARNED_PARAM, f'train_X_{model_arch[15:]}.npy')), np.load(os.path.join(LEARNED_PARAM, f'train_y_{model_arch[15:]}.npy')))

# for model_arch in MODEL_ARCH:
#     clf = clf_dict[model_arch].fit(*features[model_arch])
#     joblib.dump(clf, os.path.join(OUTPUT_FILEDIR, f'original_{model_arch[15:]}_clf.joblib'))

for i in range(3):
    best_cen_X, best_cen_y = np.load(os.path.join(LEARNED_PARAM, f'train_X_{MODEL_ARCH[i][15:]}.npy')), np.load(os.path.join(LEARNED_PARAM, f'train_y_{MODEL_ARCH[i][15:]}.npy'))
    print(MODEL_ARCH[i][15:], best_cen_X.shape)
    # break
    # clf = clf_dict[MODEL_ARCH[i]].fit(best_cen_X, best_cen_y)
    # joblib.dump(clf, os.path.join(LEARNED_PARAM, f'original_{MODEL_ARCH[i][15:]}_clf.joblib'))