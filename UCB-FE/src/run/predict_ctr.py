import numpy  as np
import pandas as pd
import warnings

from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
import os

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/run/
src_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(src_dir)  # UCB-FE/

# Add to sys.path
sys.path.insert(0, src_dir)  # Add src directory
sys.path.insert(0, os.path.join(src_dir, 'models'))  # Add models directory

print("=" * 60)
print("Import Debug Information")
print("=" * 60)
print(f"Current directory: {current_dir}")
print(f"Source directory: {src_dir}")
print(f"Project root: {project_root}")
print(f"Python path (first 3): {sys.path[:3]}")

from models.ml import ML
from models.tabm import TabMModel
from models.lightgbm_ml import Lightgbm
from models.xgboost_ml import Xgboost
from models.catboost_ml import Catboost
from models.tab_net import Tab_net
from models.fuxiadapter import FuxiUniversalAdapter
from models.FinalNet import FinalNet
from models.DIN import DIN
from models.xDeepFM import xDeepFM
from models.wide_deep import WideDeep
from models.feature_transformer import FeatureTransformer

print("\n" + "=" * 60)
print("Import Summary")
print("=" * 60)
print(f"ML: {'✓' if ML else '✗'}")
print(f"TabMModel: {'✓' if TabMModel else '✗'}")
print(f"Lightgbm: {'✓' if Lightgbm else '✗'}")
print(f"Xgboost: {'✓' if Xgboost else '✗'}")
print(f"Catboost: {'✓' if Catboost else '✗'}")
print(f"Tab_net: {'✓' if Tab_net else '✗'}")
print(f"FinalNet: {'✓' if FinalNet else '✗'}")
print(f"FuxiUniversalAdapter: {'✓' if FuxiUniversalAdapter else '✗'}")

def ucb_ctr_task(df, delta, T):
    """
    ----------------------------------------------------------------------------------------
    delta : param of UCB
    T : cold-start period
    ----------------------------------------------------------------------------------------

    Create additional columns displaying the value of the !CTR feature! according to the UCB approach

    e.g., column 'ucb_ctr_T_100' shows UCB value of !CTR feature! that incorporates a cold-start period of T=100.

    e.g., column 'cold_100' indicates (1/0) if an item falls within the cold-start period of T=100.
    """

    N = np.clip(df["item_imps"], a_min=0.5, a_max=None)

    df[f"cold_{T}"] = 0 + (df['item_imps'] <= T)

    if T <= 1:
        df[f'ucb_ctr_T_{T}'] = df[f'cold_{T}'] + (1 - df[f'cold_{T}']) * df['ctr']
    else:
        df['ucb_ctr_T_' + str(T)] = np.clip(
            df['cold_' + str(T)] * (df['ctr'] + np.sqrt(delta * np.log(T) / N)) +
            (1 - df['cold_' + str(T)]) * df['ctr'],
            a_min=0, a_max=1
        )

def ctr_prediction_task(df: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, T: float, ml: ML, model_name: str, delta: float):
    """
    ----------------------------------------------------------------------------------------
    T : cold-start period
    model_name : name of the ML model, selected from a predefined list of pre-trained models
    ----------------------------------------------------------------------------------------

    Create additional columns displaying the value of the !CTR prediction!

    e.g., column 'base_ctr_pred_catboost' shows CatBoost baseline model's CTR predictions

    e.g., column 'ucb_ctr_pred_T_100_catboost' shows CatBoost model's CTR predictions that incorporates UCB-FE and a cold-start period of T=100
    """

    print(f"  Processing model: {model_name}, T: {T}")

    # For Fuxi models, we can pass CTR values directly to avoid rebuilding datasets
    if hasattr(ml, 'predict') and 'Fuxi' in str(type(ml)):
        print(f"    Using optimized Fuxi prediction for {model_name}")
        # Baseline CTR prediction
        df[f"base_ctr_pred_{model_name}"] = ml.predict(X_test, y_test, ctr_values=df['ctr'].values)

        # UCB-FE CTR prediction
        df[f"fe_ucb_ctr_pred_T_{T}_{model_name}"] = ml.predict(X_test, y_test, ctr_values=df[f"ucb_ctr_T_{T}"].values)
    else:
        print(f"    Using standard prediction for {model_name}")
        #baseline ctr prediction
        X_test_modified, _ = FeatureTransformer.prepare_test_data(X_test, y_test, ctr_values=df['ctr'].values)
        df[f"base_ctr_pred_{model_name}"] = ml.predict(X_test_modified, y_test)

        #fe-ucb ctr prediction
        X_test_modified, _ = FeatureTransformer.prepare_test_data(X_test, y_test, ctr_values=df[f"ucb_ctr_T_{T}"].values)
        df[f"fe_ucb_ctr_pred_T_{T}_{model_name}"] = ml.predict(X_test_modified, y_test)

    #Naive UCB ctr prediction
    df['ucb_ctr_pred_T_' + str(T) + '_' + model_name] = df[f"base_ctr_pred_{model_name}"].copy()
    mask = (df['item_imps'] <= T)
    df.loc[mask, 'ucb_ctr_pred_T_' + str(T) + '_' + model_name] = df['ucb_ctr_T_' + str(T)]

    ctr_diff = np.mean(np.abs(df[f"ucb_ctr_pred_T_{T}_{model_name}"].values - df[f"base_ctr_pred_{model_name}"].values))
    print(f"The real difference for the models are {ctr_diff}")

def add_position(df, model_names, cold_periods):
    """
    ----------------------------------------------------------------------------------------
    model_names : predefined list of names of pre-trained models [catboost, xgboost,...]
    cold_periods : list of values of cold-start periods [0, 10, 100,...]
    ----------------------------------------------------------------------------------------

    Create additional columns displaying SERP positions ranked by predicted CTR.

    e.g., column 'pos_old_catboost' shows rankings based on the CatBoost baseline model's predictions.

    e.g., column 'pos_new_T_100_catboost' shows rankings from a CatBoost model that incorporates UCB-FE and a cold-start period of T=100.
    """

    codes, _ = pd.factorize(df['request_id'])
    df['request_id'] = codes
    df = df.sort_values(['request_id'], ascending=[True])

    # Vectorized approach to calculate position information
    grouped = df.groupby('request_id')
    sizes = grouped.size()
    n_items = np.repeat(sizes.values, sizes.values)
    pos_serp = np.concatenate([np.arange(size) for size in sizes.values])

    # Convert to lists to maintain compatibility with existing code
    n_items = n_items.tolist()
    pos_serp = pos_serp.tolist()

    for name in tqdm(model_names, desc="Iterating over the models."):
        df = df.sort_values(['request_id', 'base_ctr_pred_' + name], ascending=[True, False])
        df['pos_old_' + name] = pos_serp
        df['n_items'] = n_items

        for T in cold_periods:
            df = df.sort_values(['request_id', 'fe_ucb_ctr_pred_T_' + str(T) + '_' + name], ascending=[True, False])
            df['pos_fe_ucb_T_' + str(T) + '_' + name] = pos_serp

            df = df.sort_values(['request_id', 'ucb_ctr_pred_T_' + str(T) + '_' + name], ascending=[True, False])
            df['pos_ucb_T_' + str(T) + '_' + name] = pos_serp

    return df

def run_prediction_pipeline(dataset_id='my_dataset', experiment_id='FinalNet_test', config_path=['src', 'models', 'config_finalnet'],
                             result_dir=None, cold_periods=None, delta=1.5, top_n=10, tail_m=30):
    """Run the prediction pipeline with specified dataset"""

    # Set up paths
    current_file_dir = os.path.dirname(__file__)  # src/run/
    src_dir = os.path.dirname(current_file_dir)    # src/
    project_root = os.path.dirname(src_dir)        # Project root (where data/ is)

    # Define data directory path
    WEIGHTS_DIR = os.path.join(project_root, 'weights')
    DATA_DIR = os.path.join(project_root, 'data')
    if result_dir is None:
        RESULT_DIR = os.path.join(DATA_DIR, 'results')
    else:
        RESULT_DIR = result_dir
    DATA_DIR = os.path.join(DATA_DIR, dataset_id)
    CONFIG_DIR = os.path.join(project_root, *config_path)

    TRAIN_DF = os.path.join(DATA_DIR, 'train.parquet')
    TEST_DF = os.path.join(DATA_DIR, 'test.parquet')

    # Read data
    train_df = pd.read_parquet(TRAIN_DF)
    test_df = pd.read_parquet(TEST_DF)
    train_df = train_df.loc[(train_df['item_imps'] > 0)]

    y_train = train_df.target
    X_train = train_df.drop(['target', 'request_id', 'item_id', 'item_imps', 'item_shows'], axis=1)

    y_test = test_df.target
    X_test = test_df.drop(['target', 'request_id', 'item_id', 'item_imps', 'item_shows'], axis=1)

    categorical_features = []
    df = test_df[['target', 'request_id', 'item_id', 'item_imps', 'ctr']]

    # Init models
    fuxi_WideDeep_ml =  FuxiUniversalAdapter(
            X_train=X_test,
            y_train=y_test,
            categorical_features=categorical_features,
            model_class=WideDeep,
            config_path=CONFIG_DIR,
            experiment_id=experiment_id,
    )

    model_dict = {
        # 'lightgbm': lightgbm_ml,
        # 'xgboost': xgboost_ml,
        # 'tabm': tabm_ml, 
        # 'catboost': catboost_ml, 
        # 'tabnet': tabnet_ml,
        # 'fuxi_finalnet': fuxi_finalnet_ml,
        # 'fuxi_xDeepFM': fuxi_xDeepFM_ml,
        'fuxi_WideDeep_ml': fuxi_WideDeep_ml,
    }
    model_names = model_dict.keys()
    SAVE_DF = os.path.join(RESULT_DIR, 'datasets', f'{dataset_id}', f'df_{"".join(model_names)}_.parquet')

    for model_name in model_names:
        ml = model_dict[model_name]
        try:
            ml._load_model_(prefix=WEIGHTS_DIR)
            print(f"Loaded pre-trained {model_name} model")
        except FileNotFoundError:
            print(f"No pre-trained model found for {model_name}. Fitting the model.")
            ml.fit()
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            print(f"Skipping {model_name} model.")
            continue

    # Use provided parameters or defaults
    if cold_periods is None:
        cold_periods = [0, 100]  # Limited for testing

    print("Starting prediction...")
    print(f"Models: {list(model_names)}")
    print(f"Cold periods: {cold_periods}")
    print(f"Delta: {delta}")
    print(f"Top N: {top_n}")
    print(f"Tail M: {tail_m}")

    # Reorder loops: iterate over models first, then over T values
    for model_name in tqdm(model_names, desc="Iteration over models"):
        print(f"Processing model: {model_name}")
        if model_name in ['fuxi_finalnet', 'fuxi_din', 'fuxi_xDeepFM', 'fuxi_WideDeep_ml']:
            X_test = test_df.drop(['target'], axis=1)

        ml = model_dict[model_name]
        for T in tqdm(cold_periods, desc=f"Iteration over cold periods for {model_name}"):
            print(f"  Processing T={T} for model {model_name}")
            ucb_ctr_task(df, delta, T)
            ctr_prediction_task(df, X_test, y_test, T, ml, model_name, delta)
    print("Call the add_position function")
    df = add_position(df, model_names, cold_periods)

    print(f"Start saving the results to parquet file.")
    df.to_parquet(SAVE_DF)
    print("Finished processing. Results saved to parquet file.")

    return model_names, SAVE_DF
