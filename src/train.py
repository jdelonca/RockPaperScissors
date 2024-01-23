import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import joblib
import yaml
from configs.config import classes

def build_model(training_config):
    if training_config['model_type'] == 'mlp':
        model = MLPClassifier(**training_config['model_specific_params'])
    elif training_config['model_type'] == 'svm':
        model = SVC(**training_config['model_specific_params'])
    else:
        raise RuntimeError(f"{training_config['model_type']} is not a supported model type")
    return model

def main(path_to_training_config='./configs/training_config.yml', path_to_dataset_builder_config='./configs/dataset_builder_config.yml'):
    with open(path_to_training_config, 'r') as f:
        training_config = yaml.safe_load(f)
    with open(path_to_dataset_builder_config, 'r') as f:
        dataset_config = yaml.safe_load(f)
    datas = pd.read_csv(dataset_config['out_name'])
    # datas.drop('0Z', axis='columns', inplace=True) # because every z coord are distance to joints 0 so 0Z always equals 0
    model = build_model(training_config)
    grid_search_pipe = make_pipeline(StandardScaler(),
                         GridSearchCV(model, 
                                      param_grid=training_config['grid_search_ranges'], 
                                      verbose=1, 
                                      cv=training_config['cross_val_folds'], 
                                      n_jobs=-1,
                                      refit=True))

    classes_columns = list(classes.values())
    grid_search_pipe.fit(datas.loc[:,'0X':'20Z'], datas[classes_columns])

    print(grid_search_pipe['gridsearchcv'].best_params_)
    print(grid_search_pipe['gridsearchcv'].best_score_)
    
    joblib.dump(grid_search_pipe, training_config['model_file'])

if __name__ == '__main__':
    main()