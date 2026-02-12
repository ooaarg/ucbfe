import math
import random
import warnings
from typing import Optional, Union, List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import sklearn.metrics
from tqdm.std import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.utils.validation import check_random_state
from .ml import ML
from .xgboost import Xgboost
from .lightgbm import Lightgbm


def _calculate_permutation_scores(X_permuted, y_test, col_idx, model, n_repeats, baseline_score, random_state):
    random_state = check_random_state(random_state)
    scores = []
    shuffling_idx = np.arange(X_permuted.shape[0])
    for _ in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        col = X_permuted.iloc[shuffling_idx, col_idx]
        col.index = X_permuted.index
        X_permuted[X_permuted.columns[col_idx]] = col

        y_pred = model.predict(X_permuted)
        permuted_score = sklearn.metrics.roc_auc_score(y_test, y_pred)

        scores.append(permuted_score)

    scores = baseline_score - np.array(scores)
    return {"column": X_permuted.columns[col_idx], "importance_mean": scores.mean(), "importance_std": scores.std()}



def permutation(model, X_test, y_test, n_repeats=1, random_state=42):
    assert isinstance(X_test, pd.DataFrame), "X_test must be a pandas DataFrame"
    assert isinstance(model, ML), "model must be a ML object"

    X_permuted = X_test.copy()
    random_state = check_random_state(random_state)

    baseline_pred = model.predict(X_test)
    baseline_score = sklearn.metrics.roc_auc_score(y_test, baseline_pred)

    if isinstance(model, Xgboost) or isinstance(model, Lightgbm):
        scores = Parallel(n_jobs=-1)(
                delayed(_calculate_permutation_scores)(
                    X_permuted, y_test, col_idx, model, n_repeats, baseline_score, random_state
                )
                for col_idx in range(X_permuted.shape[1])
            )
    else:
        scores = [_calculate_permutation_scores(X_permuted, y_test, col_idx, model, n_repeats, baseline_score, random_state) for col_idx in range(X_permuted.shape[1])]
        

    return pd.DataFrame(scores)




# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_parquet('UCB-FE/data/train_int.parquet')
    X = df.drop(['target', 'request_id', 'item_id', 'item_imps', 'item_shows'], axis=1)

    y = df['target']
    categorical_features = ["mcat", "mcat_1", "mcat_2", "mcat_3", "mcat_4", "mcat_5", "cat_id", "item_region_id", "item_location_id"]
    
    # Calculate cardinalities for categorical features
    cat_cardinalities = [X[col].nunique() for col in categorical_features]
    
    # Test all models
    models = {
        # 'TabM': TabMModel(X, y, categorical_features),
        # 'LightGBM': Lightgbm(X, y, categorical_features),
        # 'XGBoost': Xgboost(X, y, categorical_features),
        # 'CatBoost': Catboost(X, y, categorical_features),
        # 'TabNet': Tab_net(X, y, categorical_features)
    }
    
    for name, model in models.items():
        print(f"\nTesting {name} model:")
        
        # model.fit()
        # model._log_model_()
        model._load_model_()
        print(model.loaded)
        predictions = model.predict(X[:100])
        print(f"{name} sample predictions:", predictions[:5])
