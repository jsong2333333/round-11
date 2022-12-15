import numpy as np
from jsonargparse import ArgumentParser, ActionConfigFile
import logging
import torch
import json
import jsonschema
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import feature_extractor as fe

# There's a difference in the filepath when loading data from learned_parameter folder in the container, deleting '.' in './learned_parameters'

MODEL_ARCH = ['resnet50', 'vit_base_patch32_224', 'mobilenet_v2']
ORIGINAL_LEARNED_PARAM_DIR = './learned_parameters'

TUNABLE_PARAMS = ['learning_rate', 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
param_grid = {'gbm__learning_rate':np.arange(.005, .0251, .005), 'gbm__n_estimators':range(300, 1001, 100), 'gbm__max_depth':range(2, 6), 'gbm__max_features':range(20, 121, 10)}


def weight_analysis_detector(model_filepath,
                             examples_dirpath,
                             result_filepath,
                             learned_parameters_dirpath):

    logging.info('model_filepath = {}'.format(model_filepath))
    logging.info('examples_dirpath = {}'.format(examples_dirpath))
    logging.info('result_filepath = {}'.format(result_filepath))
    logging.info('Using parameters_dirpath = {}'.format(learned_parameters_dirpath))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))

    # extract class and features from predict model
    logging.info("Starting feature extraction")
    predict_model_class, predict_model_features = fe.get_predict_model_features_and_class(model_filepath, examples_dirpath, ORIGINAL_LEARNED_PARAM_DIR, device)
    predict_model_features = np.asarray([predict_model_features])

    logging.info('Starting to load classifier')
    potential_reconfig_model_filepath = os.path.join(learned_parameters_dirpath, f'{predict_model_class}_clf.joblib')
    if os.path.exists(potential_reconfig_model_filepath):
        clf = joblib.load(potential_reconfig_model_filepath)
    else:
        logging.info('Using original classifier')
        clf = joblib.load(os.path.join(ORIGINAL_LEARNED_PARAM_DIR, f'original_{predict_model_class}_clf.joblib'))
    
    logging.info('Starting to detect trojan probability')
    try:
        trojan_probability = clf.predict_proba(predict_model_features)
    except:
        logging.warning('Not able to detect such model class')
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(0.50))
        return

    logging.info('Trojan Probability of this class {} model is: {}'.format(predict_model_class, trojan_probability[0, -1]))
    
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability[0, -1]))


def configure(output_parameters_dirpath,
              configure_models_dirpath,
              config_json_file):
    logging.info('Configuring detector parameters with models from ' + configure_models_dirpath)
    os.makedirs(output_parameters_dirpath, exist_ok=True)
    logging.info('Saving reconfigured models to ' + output_parameters_dirpath)

    logging.info('Starting to extract features from new training models')
    feature_dict = {}
    for ma in MODEL_ARCH:
        try:
            potential_features = fe.get_features_and_labels_by_model_class(configure_models_dirpath, ma, ORIGINAL_LEARNED_PARAM_DIR, device)
            X, y = potential_features['X'], potential_features['y']
            if X.shape[0] != 0 and X.shape[0] == y.shape[0] and X.shape[1] == fe.MODEL_ARCH_FOR_LOSS_CALCULATION[ma]:
                feature_dict[ma] = (X, y)
            else:
                logging.info(f'Mismatch or no data found for model architechture {ma}, not saved')
        except:
            logging.info(f'Problem encountered when extracting features from configure_models_dirpath for model architechture {ma}.')
            

    AUGMENT_TRAIN_DATA = config_json_file['augment_train_data']
    logging.info(f'Augment train data is {str(AUGMENT_TRAIN_DATA)}')
    if AUGMENT_TRAIN_DATA:
        for ma in MODEL_ARCH:
            try:
                train_X, train_y = np.load(os.path.join(ORIGINAL_LEARNED_PARAM_DIR, f'train_X_{ma}.npy')), np.load(os.path.join(ORIGINAL_LEARNED_PARAM_DIR, f'train_y_{ma}.npy'))
                new_X, new_y = feature_dict[ma]
                if train_X.shape[1] == new_X.shape[1] and train_X.shape[0] == train_y.shape[0]:
                    feature_dict[ma] = (np.concatenate((new_X, train_X), axis=0), np.concatenate((new_y, train_y), axis=0))
                else:
                    logging.info(f'Mismatch in number of features extracted for {ma}.')
            except:
                logging.info(f'Problem encountered when augmenting train data for {ma}.')


    logging.info('Starting to tune models')
    AUTOMATIC_TRAINING = config_json_file['automatic_training']
    logging.info(f'Auto-tuning mode is {str(AUTOMATIC_TRAINING)}')

    if AUTOMATIC_TRAINING:
        logging.info('Currently auto-tuning only learining_rate, n_estimator, max_depth and max_features.')

    output_metaparameter = {'augment_train_data': AUGMENT_TRAIN_DATA, 'automatic_training': AUTOMATIC_TRAINING}

    for ma, (X, y) in feature_dict.items():
        logging.info(f'Tuning classifier for {ma}')
        _, counts = np.unique(y, return_counts=True)
        try:
            recofig_clf, metaparams = None, None
            if AUTOMATIC_TRAINING:
                pipe = Pipeline(steps=[('gbm', GradientBoostingClassifier())])
                
                kfold = min(min(counts), 5)
                if kfold < 2 or len(counts) != 2:
                    logging.info(f'Not enough data points are given for auto-tuning the model for model architecture {ma}.')
                    continue
                gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold)
                gsearch.fit(X, y)
                
                recofig_clf = gsearch.best_estimator_
                metaparams = gsearch.best_params_
            else:
                if len(counts) != 2:
                    logging.info(f'Not enough classes are provided for fitting the classifier for model architecture {ma}.')
                    continue
                param_args = {k[len(ma)+1:]:v for k, v in config_json_file.items() if k.startswith(ma)}
                recofig_clf = GradientBoostingClassifier(**param_args)
                recofig_clf.fit(X, y)
                metaparams = param_args

            joblib.dump(recofig_clf, os.path.join(output_parameters_dirpath, f'{ma}_clf.joblib'))
            for k, v in metaparams.items():
                if AUTOMATIC_TRAINING:
                    k = k[5:]
                if k in TUNABLE_PARAMS:
                    output_metaparameter[f'{ma}_{k}'] = v
        except:
            logging.info(f'Problem encountered when parsing parameters or training classifier for model architecture {ma}.')
    
    logging.info(f'Saving new metaparameters to directory {output_parameters_dirpath}.')
    with open(os.path.join(output_parameters_dirpath, 'metaparameters.json'), 'w') as outfile:
        json.dump(output_metaparameter, outfile)


if __name__ == '__main__':
    parser = ArgumentParser(description='Weight Analysis Classifier')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--source_dataset_dirpath', type=str, help='File path to a directory containing the original clean dataset into which triggers were injected during training.', default=None)
    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', type=str, help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', default=None)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("classifier.py launched")


    # Validate config file against schema
    config_json = None
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)


    if not args.configure_mode:
        if (args.model_filepath is not None and
            args.examples_dirpath is not None and
            args.result_filepath is not None and
            args.learned_parameters_dirpath is not None):

            logging.info("Calling the trojan detector")
            logging.info('Calling the weight analysis classifier')

            weight_analysis_detector(args.model_filepath,
                                     args.examples_dirpath,
                                     args.result_filepath,
                                     args.learned_parameters_dirpath)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
            args.configure_models_dirpath is not None and
            config_json is not None):

            logging.info("Calling configuration mode")
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      config_json)
        else:
            logging.info("Required Configure-Mode parameters missing!")