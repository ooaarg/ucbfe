import numpy  as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# class StratifiedAUC:
#     """
#     Weighted average of ROC-AUC metric by group, where weights are determined by the number of positive targets in each group. 
#     StratifiedAUC is taken from a Yahoo article https://arxiv.org/abs/2312.05052
#     """

#     default_name = "StratifiedAUC"

#     def __init__(
#         self,
#         target_column: str,
#         group_column: str,
#     ):
#         self.target_column = target_column
#         self.group_column = group_column

#     def __call__(self, serp: pd.DataFrame, rank_column: str) -> float:
#         if serp[self.target_column].sum() < 1:
#             return np.nan

#         def _fn_num(group_df: pd.DataFrame) -> float:
#             if group_df[self.target_column].nunique() == 1:
#                 return np.nan
#             roc_auc = roc_auc_score(group_df[self.target_column], group_df[rank_column])
#             return roc_auc * group_df[self.target_column].sum()

#         def _fn_den(group_df: pd.DataFrame) -> float:
#             if group_df[self.target_column].nunique() == 1:
#                 return 0
#             return group_df[self.target_column].sum()

#         num = serp.groupby(self.group_column).apply(_fn_num).sum()
#         den = serp.groupby(self.group_column).apply(_fn_den).sum()

#         return num / den

#     @property
#     def name(self):
#         return self.default_name

class StratifiedAUC:
    """
    Weighted average of ROC-AUC metric by group, where weights are determined by the number of positive targets in each group.
    StratifiedAUC is taken from a Yahoo article https://arxiv.org/abs/2312.05052
    """

    default_name = "StratifiedAUC"

    def __init__(
        self,
        target_column: str,
        group_column: str,
    ):
        self.target_column = target_column
        self.group_column = group_column

    def __call__(self, serp: pd.DataFrame, rank_column: str) -> float:
        y = serp[self.target_column].to_numpy()
        if y.sum() < 1:
            return np.nan

        scores = serp[rank_column].to_numpy()
        groups = serp[self.group_column].to_numpy()

        # Sort by (group, score)
        order = np.lexsort((scores, groups))
        y = y[order]
        scores = scores[order]
        groups = groups[order]

        # Find group boundaries
        unique_groups, group_starts = np.unique(groups, return_index=True)

        total_num = 0.0
        total_den = 0.0

        n = len(y)

        for i in range(len(group_starts)):
            start = group_starts[i]
            end = group_starts[i + 1] if i + 1 < len(group_starts) else n

            y_g = y[start:end]

            P = y_g.sum()
            if P == 0 or P == (end - start):
                continue  # skip single-class groups

            # Compute ranks inside the group
            # since already sorted by score, rank = 1..k
            k = end - start
            ranks = np.arange(1, k + 1, dtype=np.float64)

            pos_ranks_sum = ranks[y_g == 1].sum()
            N = k - P

            # Mann–Whitney U → AUC
            auc = (pos_ranks_sum - P * (P + 1) / 2.0) / (P * N)

            total_num += auc * P
            total_den += P

        if total_den == 0:
            return np.nan

        return total_num / total_den

    @property
    def name(self):
        return self.default_name



def run_roc_auc_pipeline(dataset_id='my_dataset', model_names=None, df_path=None, result_dir=None,
                        cold_periods=None, delta=1.5, top_n=10, tail_m=30):
    """Run the ROC AUC pipeline with specified dataset"""
    
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
        
    SAVE_AUC = os.path.join(RESULT_DIR, 'auc', f'auc_{dataset_id}.csv')
    SAVE_STRAT_AUC = os.path.join(RESULT_DIR, 'strat_auc', f'strat_auc_{dataset_id}.csv')

    df = pd.read_parquet(DF_PATH)
    
    # Use provided parameters or defaults
    if cold_periods is None:
        cold_periods = [0, 10, 100]  # 200, 500

    if model_names is None:
        model_names = ['fuxi_finalnet']
        
    save_columns = ['target', 'request_id', 'item_id']

    for name in model_names:
        save_columns.append('base_ctr_pred_' + name)
        for T in cold_periods:
            save_columns.append('ucb_ctr_pred_T_'+str(T) + '_' + name)

    df = df[save_columns]

    strat_auc_score = StratifiedAUC('target', 'request_id')
    df_grouped = df.groupby('request_id')

    auc_data = []
    strat_auc_data = []

    columns_auc = ['AUC base']
    columns_strat_auc = ['Stratified AUC base']

    for T in cold_periods:
        columns_auc.append('AUC fe-ucb, T = ' + str(T))
        columns_strat_auc.append('Stratified AUC fe-ucb, T = ' + str(T))

    for name in tqdm(model_names, desc="Iterating over models"):
        auc = []
        strat_auc= []

        auc_old = roc_auc_score(df['target'], df['base_ctr_pred_' + name])
        strat_auc_old = df_grouped.apply(strat_auc_score, rank_column= 'base_ctr_pred_' + name)

        auc.append(auc_old)
        strat_auc.append(strat_auc_old.mean())
        print(f"For model {name}, obtained the AUC: {auc_old} and Stratified AUC: {strat_auc_old}")

        for T in tqdm(cold_periods, desc="Iterating over periods with fixed model"):
            auc_new= roc_auc_score(df['target'], df['ucb_ctr_pred_T_'+str(T) + '_' + name])    
            strat_auc_new = df_grouped.apply(strat_auc_score, rank_column= 'ucb_ctr_pred_T_'+str(T) + '_' + name)
        
            auc.append(auc_new)
            strat_auc.append(strat_auc_new.mean())
            print(f"For model {name} with period T: {T}, obtained the AUC: {auc_old} and Stratified AUC: {strat_auc_old}")
        
        auc_data.append(auc)
        strat_auc_data.append(strat_auc)

    result_auc = pd.DataFrame(
        data = auc_data, 
        columns= columns_auc, 
        index= model_names
    )

    result_strat_auc = pd.DataFrame(
        data = strat_auc_data, 
        columns= columns_strat_auc, 
        index= model_names
    )


    result_auc.to_csv(SAVE_AUC, index = True) # index is a name of the algorithm
    result_strat_auc.to_csv(SAVE_STRAT_AUC, index = True)
    
    return SAVE_AUC, SAVE_STRAT_AUC


if __name__ == "__main__":
    current_file_dir = os.path.dirname(__file__)  # src/run/
    src_dir = os.path.dirname(current_file_dir)    # src/
    project_root = os.path.dirname(src_dir)        # Project root (where data/ is)

    # Define data directory path
    DATA_DIR = os.path.join(project_root, 'data')
    RESULT_DIR = os.path.join(DATA_DIR, 'results')

    DF_PATH = os.path.join(RESULT_DIR, 'df_ucb_fe.parquet')
    SAVE_AUC = os.path.join(RESULT_DIR, 'auc.csv')
    SAVE_STRAT_AUC = os.path.join(RESULT_DIR, 'strat_auc.csv')

    df =  pd.read_parquet(DF_PATH)
    cold_periods = [0, 10, 100, ] # 200, 500
    delta = 1.5
    top_n = 10
    tail_m = 30

    # model_names = ['catboost','lightgbm', 'xgboost', 'tabnet', 'tabm']
    model_names = ['fuxi_finalnet']
    save_columns = ['target', 'request_id', 'item_id']

    for name in model_names:
        save_columns.append('base_ctr_pred_' + name)
        for T in cold_periods:
            save_columns.append('ucb_ctr_pred_T_'+str(T) + '_' + name)

    df = df[save_columns]

    strat_auc_score = StratifiedAUC('target', 'request_id')
    df_grouped = df.groupby('request_id')

    auc_data = []
    strat_auc_data = []

    columns_auc = ['AUC base']
    columns_strat_auc = ['Stratified AUC base']

    for T in cold_periods:
        columns_auc.append('AUC fe-ucb, T = ' + str(T))
        columns_strat_auc.append('Stratified AUC fe-ucb, T = ' + str(T))

    for name in tqdm(model_names, desc="Iterating over models"):
        auc = []
        strat_auc= []

        auc_old = roc_auc_score(df['target'], df['base_ctr_pred_' + name])
        strat_auc_old = df_grouped.apply(strat_auc_score, rank_column= 'base_ctr_pred_' + name)

        auc.append(auc_old)
        strat_auc.append(strat_auc_old.mean())
        print(f"For model {name}, obtained the AUC: {auc_old} and Stratified AUC: {strat_auc_old}")

        for T in tqdm(cold_periods, desc="Iterating over periods with fixed model"):
            auc_new= roc_auc_score(df['target'], df['ucb_ctr_pred_T_'+str(T) + '_' + name])    
            strat_auc_new = df_grouped.apply(strat_auc_score, rank_column= 'ucb_ctr_pred_T_'+str(T) + '_' + name)
        
            auc.append(auc_new)
            strat_auc.append(strat_auc_new.mean())
            print(f"For model {name} with period T: {T}, obtained the AUC: {auc_old} and Stratified AUC: {strat_auc_old}")
        
        auc_data.append(auc)
        strat_auc_data.append(strat_auc)

    result_auc = pd.DataFrame(
        data = auc_data, 
        columns= columns_auc, 
        index= model_names
    )

    result_strat_auc = pd.DataFrame(
        data = strat_auc_data, 
        columns= columns_strat_auc, 
        index= model_names
    )


    result_auc.to_csv(SAVE_AUC, index = True) # index is a name of the algorithm
    result_strat_auc.to_csv(SAVE_STRAT_AUC, index = True)
