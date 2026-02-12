import numpy  as np
import pandas as pd
import os

def custom_metrics(df, model_names, cold_periods, dataset_name="Industrial", n=10, m=30, lam=0.98):
    """
    ----------------------------------------------------------------------------------------
    model_names : predefined list of names of pre-trained models ([catboost, xgboost,...])
    cold_periods : list of values of cold-start periods ([0, 10, 100,...])
    dataset_name : name of the dataset to determine the importance function
    n : lower level of top positions (...8, 9, 10)
    m : upper level of tail positions (30, 31, 32....)
    lam : basis of the degree for the position (0.98)
    ----------------------------------------------------------------------------------------

    Calculate the necessary values for calculating the 'clicks' metrics such as CBIQ in the chapter 3.4.3
    
    """    

    ad_df = pd.DataFrame()
    
    # Determine importance function based on dataset name
    if dataset_name == 'Industrial':
        def imp_pos(pos):
            return lam**pos
    else:
        def imp_pos(pos):
            return 1/(1+np.log(1+pos))

    for name in model_names:
        imps = np.sum(imp_pos(df['pos_old_' + name]))
        for T in cold_periods:
 
            imps_cold_base =  np.sum(imp_pos(df['pos_old_' + name]) * df['cold_'+str(T)])
            imps_cold_fe_ucb =  np.sum(imp_pos(df['pos_fe_ucb_T_'+str(T) + '_' + name]) * df['cold_'+str(T)])
            imps_cold_ucb =  np.sum(imp_pos(df['pos_ucb_T_'+str(T) + '_' + name]) * df['cold_'+str(T)])
            shows_cold = np.sum(df['cold_'+str(T)])


            clicks_cold_base = np.sum(imp_pos(df['pos_old_' + name]) * df['cold_'+str(T)] * df['target'])
            clicks_cold_ucb = np.sum(imp_pos(df['pos_ucb_T_'+str(T) + '_' + name]) * df['cold_'+str(T)] * df['target'])
            clicks_cold_fe_ucb = np.sum(imp_pos(df['pos_fe_ucb_T_'+str(T) + '_' + name]) * df['cold_'+str(T)] * df['target'])


            clicks_base =  np.sum(imp_pos(df['pos_old_' + name]) * df['target'])
            clicks_ucb =  np.sum(imp_pos(df['pos_ucb_T_'+str(T) + '_' + name]) * df['target'])
            clicks_fe_ucb =  np.sum(imp_pos(df['pos_fe_ucb_T_'+str(T) + '_' + name]) * df['target'])
    
            df_result= pd.DataFrame({'dataset': [dataset_name],'model': [name], 'cold_period' : [T], 'LTR': ['Baseline'],
                                    'ColdImps' : [imps_cold_base/shows_cold], 
                                    'ColdCtr': [clicks_cold_base/shows_cold], 
                                    'Ctr'  : [clicks_base/imps],
                                  })
    
            ad_df = pd.concat([ad_df, df_result])

            df_result= pd.DataFrame({'dataset': [dataset_name],'model': [name], 'cold_period' : [T], 'LTR': ['UCB-FE'],
                                    'ColdImps' : [imps_cold_fe_ucb/shows_cold], 
                                    'ColdCtr'  : [clicks_cold_fe_ucb/shows_cold],
                                    'Ctr' : [clicks_fe_ucb/imps],
                                  })
            ad_df = pd.concat([ad_df, df_result])

            df_result= pd.DataFrame({'dataset': [dataset_name],'model': [name], 'cold_period' : [T], 'LTR': ['UCB'],
                                    'ColdImps' : [imps_cold_ucb/shows_cold], 
                                    'ColdCtr'  : [clicks_cold_ucb/shows_cold],
                                    'Ctr' : [clicks_ucb/imps],
                                  })
            ad_df = pd.concat([ad_df, df_result])

    return ad_df


def run_custom_metrics_pipeline(dataset_id='my_dataset', model_names=None, df_path=None, result_dir=None,
                               cold_periods=None, delta=1.5, top_n=10, tail_m=30):
    """Run the custom metrics pipeline with specified dataset"""
    
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
        
    SAVE_98 = os.path.join(RESULT_DIR, 'custom_metrics', f'custom_metrics_0.98_{dataset_id}.csv')
    SAVE_90 = os.path.join(RESULT_DIR, 'custom_metrics', f'custom_metrics_0.90_{dataset_id}.csv')
    SAVE_80 = os.path.join(RESULT_DIR, 'custom_metrics', f'custom_metrics_0.80_{dataset_id}.csv')

    df = pd.read_parquet(DF_PATH)

    if model_names is None:
        model_names = ['fuxi_finalnet']
        
    # Use provided parameters or defaults
    if cold_periods is None:
        cold_periods = [0, 10, 100, 200, 500]

    result = custom_metrics(df, model_names, cold_periods, n = top_n, m = tail_m, lam = 0.98)
    result.to_csv(SAVE_98, index = False)

    result = custom_metrics(df, model_names, cold_periods, n = top_n, m = tail_m, lam = 0.9)
    result.to_csv(SAVE_90, index = False)

    result = custom_metrics(df, model_names, cold_periods, n = top_n, m = tail_m, lam = 0.8)
    result.to_csv(SAVE_80, index = False)
    
    return SAVE_98, SAVE_90, SAVE_80


if __name__ == "__main__":
    current_file_dir = os.path.dirname(__file__)  # src/run/
    src_dir = os.path.dirname(current_file_dir)    # src/
    project_root = os.path.dirname(src_dir)        # Project root (where data/ is)


    # Define data directory path
    DATA_DIR = os.path.join(project_root, 'data')
    RESULT_DIR = os.path.join(DATA_DIR, 'results')

    DF_PATH = os.path.join(RESULT_DIR, 'df_ucb_fe.parquet')
    SAVE_98 = os.path.join(RESULT_DIR, 'custom_metrics_0.98.csv')
    SAVE_90 = os.path.join(RESULT_DIR, 'custom_metrics_0.90.csv')
    SAVE_80 = os.path.join(RESULT_DIR, 'custom_metrics_0.80.csv')


    df =  pd.read_parquet(DF_PATH)

    model_names = ['catboost','lightgbm', 'xgboost', 'tabnet', 'tabm']
    cold_periods = [0, 10, 100, 200, 500]
    top_n = 10
    tail_m = 30

    result = custom_metrics(df, model_names, cold_periods, n = top_n, m = tail_m, lam = 0.98)
    result.to_csv(SAVE_98, index = False)

    result = custom_metrics(df, model_names, cold_periods, n = top_n, m = tail_m, lam = 0.9)
    result.to_csv(SAVE_90, index = False)

    result = custom_metrics(df, model_names, cold_periods, n = top_n, m = tail_m, lam = 0.8)
    result.to_csv(SAVE_80, index = False)
