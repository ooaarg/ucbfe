import numpy  as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# class NDCG:
#     default_name = "NDCG"

#     def __init__(
#         self,
#         target_column: str,
#         discount_base: float,
#         top_k = None,
#     ):
#         self.target_column = target_column
#         self.discount_base = discount_base
#         self.top_k = top_k

#     def __call__(self, serp: pd.DataFrame, rank_column: str) -> float:
#         if serp[self.target_column].nunique() == 1:
#             return np.nan

#         if self.top_k is None:
#             top_k = len(serp)
#         else:
#             top_k = min(len(serp), self.top_k)

#         discount_weights = self.discount_base ** np.arange(top_k)
#         serp_sorted_by_rank = serp.sort_values(by=rank_column, ascending=False)[:top_k]
#         serp_sorted_by_target = serp.sort_values(by=self.target_column, ascending=False)[:top_k]

#         score = (discount_weights * serp_sorted_by_rank[self.target_column]).sum()
#         norm = (discount_weights * serp_sorted_by_target[self.target_column]).sum()
#         return score / (norm + 1e-5)

#     @property
#     def name(self):
#         name = self.default_name
#         if self.top_k is not None:
#             name += f"@{self.top_k}"
#         name += f"_base{self.discount_base}"
#         return name

#     @property
#     def agg_mode(self) -> str:
#         return "mean"



class NDCG:
    default_name = "NDCG"

    def __init__(
        self,
        target_column: str,
        discount_base: float,
        top_k=None,
    ):
        self.target_column = target_column
        self.discount_base = discount_base
        self.top_k = top_k

    def __call__(self, serp: pd.DataFrame, rank_column: str) -> float:
        y = serp[self.target_column].to_numpy()

        # if all labels identical → undefined
        if y.min() == y.max():
            return np.nan

        scores = serp[rank_column].to_numpy()
        n = len(y)

        if self.top_k is None:
            k = n
        else:
            k = min(n, self.top_k)

        # Precompute discount weights
        discount = self.discount_base ** np.arange(k, dtype=np.float64)

        # ---- Predicted ranking ----
        if k == n:
            idx_pred = np.argsort(-scores)
        else:
            idx_part = np.argpartition(-scores, k - 1)[:k]
            idx_pred = idx_part[np.argsort(-scores[idx_part])]

        y_pred = y[idx_pred]
        dcg = np.dot(discount, y_pred)

        # ---- Ideal ranking ----
        if k == n:
            idx_true = np.argsort(-y)
        else:
            idx_part = np.argpartition(-y, k - 1)[:k]
            idx_true = idx_part[np.argsort(-y[idx_part])]

        y_true = y[idx_true]
        idcg = np.dot(discount, y_true)

        return dcg / (idcg + 1e-5)

    @property
    def name(self):
        name = self.default_name
        if self.top_k is not None:
            name += f"@{self.top_k}"
        name += f"_base{self.discount_base}"
        return name

    @property
    def agg_mode(self) -> str:
        return "mean"



def run_ndcg_pipeline(dataset_id='my_dataset', model_names=None, df_path=None, result_dir=None,
                     cold_periods=None, delta=1.5, top_n=10, tail_m=30):
    """Run the NDCG pipeline with specified dataset"""
    
    current_file_dir = os.path.dirname(__file__)  # src/run/
    src_dir = os.path.dirname(current_file_dir)    # src/
    project_root = os.path.dirname(src_dir)        # Project root (where data/ is)

    # Define data directory path
    DATA_DIR = os.path.join(project_root, 'data')
    if result_dir is None:
        RESULT_DIR = os.path.join(DATA_DIR, 'results')
    else:
        RESULT_DIR = result_dir

    if df_path is None:
        DF_PATH = os.path.join(RESULT_DIR, 'datasets', f'df_{dataset_id}.parquet')
    else:
        DF_PATH = df_path
        
    SAVE_NDCG = os.path.join(RESULT_DIR, 'ndcg', f'ndcg_{dataset_id}.csv')

    df = pd.read_parquet(DF_PATH)
    
    # Use provided parameters or defaults
    if cold_periods is None:
        cold_periods = [0, 10, 100, 200, 500]

    if model_names is None:
        model_names = ['fuxi_finalnet']
        
    save_columns = ['target', 'request_id', 'item_id']

    for name in model_names:
        save_columns.append('base_ctr_pred_' + name)
        for T in cold_periods:
            save_columns.append('ucb_ctr_pred_T_'+str(T) + '_' + name)

    df = df[save_columns]

    ndcg_score = NDCG(target_column = 'target', discount_base = 0.8)
    df_grouped = df.groupby('request_id')
        

    ndcg_data = []

    columns_ndcg = ['NDCG base']


    for T in cold_periods:
        columns_ndcg.append('NDCG fe-ucb, T = ' + str(T))

    for name in model_names:
        ndcg = []

        ndcg_old =  df_grouped.apply(ndcg_score, rank_column= 'base_ctr_pred_' + name)

        ndcg.append(ndcg_old.mean())

        for T in cold_periods:
            print("period = ", T)    
            ndcg_new = df_grouped.apply(ndcg_score, rank_column= 'ucb_ctr_pred_T_'+str(T) + '_' + name)
        
            ndcg.append(ndcg_new.mean())
        
        ndcg_data.append(ndcg)

    result_ndcg = pd.DataFrame(
        data = ndcg_data, 
        columns= columns_ndcg, 
        index= model_names
    )

    result_ndcg.to_csv(SAVE_NDCG, index = True) # index is a name of the algorithm
    
    return SAVE_NDCG


if __name__ == "__main__":
    current_file_dir = os.path.dirname(__file__)  # src/run/
    src_dir = os.path.dirname(current_file_dir)    # src/
    project_root = os.path.dirname(src_dir)        # Project root (where data/ is)

    # Define data directory path
    DATA_DIR = os.path.join(project_root, 'data')
    RESULT_DIR = os.path.join(DATA_DIR, 'results')

    DF_PATH = os.path.join(RESULT_DIR, 'df_ucb_fe.parquet')
    SAVE_NDCG = os.path.join(RESULT_DIR, 'ndcg.csv')

    df =  pd.read_parquet(DF_PATH)
    cold_periods = [0, 10, 100, 200, 500]
    delta = 1.5
    top_n = 10
    tail_m = 30

    model_names = ['catboost','lightgbm', 'xgboost', 'tabnet', 'tabm']
    save_columns = ['target', 'request_id', 'item_id']

    for name in model_names:
        save_columns.append('base_ctr_pred_' + name)
        for T in cold_periods:
            save_columns.append('ucb_ctr_pred_T_'+str(T) + '_' + name)

    df = df[save_columns]

    ndcg_score = NDCG(target_column = 'target', discount_base = 0.8)
    df_grouped = df.groupby('request_id')
        

    ndcg_data = []

    columns_ndcg = ['NDCG base']


    for T in cold_periods:
        columns_ndcg.append('NDCG fe-ucb, T = ' + str(T))

    for name in model_names:
        ndcg = []

        ndcg_old =  df_grouped.apply(ndcg_score, rank_column= 'base_ctr_pred_' + name)

        ndcg.append(ndcg_old.mean())

        for T in cold_periods:
            print("period = ", T)    
            ndcg_new = df_grouped.apply(ndcg_score, rank_column= 'ucb_ctr_pred_T_'+str(T) + '_' + name)
        
            ndcg.append(ndcg_new.mean())
        
        ndcg_data.append(ndcg)

    result_ndcg = pd.DataFrame(
        data = ndcg_data, 
        columns= columns_ndcg, 
        index= model_names
    )

    result_ndcg.to_csv(SAVE_NDCG, index = True) # index is a name of the algorithm
