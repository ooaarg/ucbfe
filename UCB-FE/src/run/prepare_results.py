#!/usr/bin/env python3
"""
Script to prepare final results for a specified dataset by combining all metrics
and calculating relative improvements.
"""
import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/run/
src_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(src_dir)  # UCB-FE/

# Add to sys.path
sys.path.insert(0, src_dir)  # Add src directory
sys.path.insert(0, os.path.join(src_dir, 'models'))  # Add models directory

# Import custom classes
from roc_auc import StratifiedAUC
from ndcg import NDCG

def load_metrics_data(df_path):
    """
    Load all metrics data for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset
        result_dir (str): Directory where results are stored

    Returns:
        tuple: (df, model_names, cold_periods)
    """
    # Load main dataframe
    # df_path = os.path.join(result_dir, 'datasets', f'df_{dataset_name}.parquet')
    print(f"Reading results from {df_path}")
    df = pd.read_parquet(df_path)

    # Extract model names from columns
    model_columns = [col for col in df.columns if col.startswith('base_ctr_pred_')]
    model_names = [col.replace('base_ctr_pred_', '') for col in model_columns]

    # Extract cold periods from columns
    cold_columns = [col for col in df.columns if col.startswith('cold_') and col != 'cold_period']
    cold_periods = sorted(list(set([int(col.replace('cold_', '')) for col in cold_columns])))

    print(f"Found models: {model_names}")
    print(f"Found cold periods: {cold_periods}")

    return df, model_names, cold_periods

def calculate_metrics(df, model_names, cold_periods, dataset_name):
    """
    Calculate all metrics (NDCG, AUC_ROC, stratified AUC, custom metrics) for the dataset.

    Args:
        df (pd.DataFrame): Main dataframe with predictions
        model_names (list): List of model names
        cold_periods (list): List of cold periods
        dataset_name (str): Name of the dataset

    Returns:
        pd.DataFrame: Combined results with all metrics
    """
    warnings.filterwarnings("ignore")

    result2 = pd.DataFrame()

    for dataset_name in [dataset_name]:
        print('Dataset: ' + dataset_name)
        # if dataset_name == 'Industrial':
        #     def imp_pos(pos):
        #         return 0.98 ** pos
        # else:
        def imp_pos(pos):
            return 1/(1 + np.log(1 + pos))

        # Load custom metrics from saved files
        # For now, we'll calculate them directly from the dataframe
        ad_df = pd.DataFrame()
        print(df.columns)

        for name in tqdm(model_names, desc="Processing models"):
            imps = np.sum(imp_pos(df['pos_old_' + name]))
            for T in tqdm(cold_periods, desc=f"Processing cold periods for {name}", leave=False):
                imps_cold_base = np.sum(imp_pos(df['pos_old_' + name]) * df['cold_'+str(T)])
                imps_cold_fe_ucb = np.sum(imp_pos(df['pos_fe_ucb_T_'+str(T) + '_' + name]) * df['cold_'+str(T)])
                imps_cold_ucb = np.sum(imp_pos(df['pos_ucb_T_'+str(T) + '_' + name]) * df['cold_'+str(T)])
                shows_cold = np.sum(df['cold_'+str(T)])

                clicks_cold_base = np.sum(imp_pos(df['pos_old_' + name]) * df['cold_'+str(T)] * df['target'])
                clicks_cold_ucb = np.sum(imp_pos(df['pos_ucb_T_'+str(T) + '_' + name]) * df['cold_'+str(T)] * df['target'])
                clicks_cold_fe_ucb = np.sum(imp_pos(df['pos_fe_ucb_T_'+str(T) + '_' + name]) * df['cold_'+str(T)] * df['target'])

                clicks_base = np.sum(imp_pos(df['pos_old_' + name]) * df['target'])
                clicks_ucb = np.sum(imp_pos(df['pos_ucb_T_'+str(T) + '_' + name]) * df['target'])
                clicks_fe_ucb = np.sum(imp_pos(df['pos_fe_ucb_T_'+str(T) + '_' + name]) * df['target'])

                df_result = pd.DataFrame({'dataset': [dataset_name], 'model': [name], 'cold_period' : [T], 'LTR': ['Baseline'],
                                        'ColdImps' : [imps_cold_base/shows_cold], 
                                        'ColdCtr': [clicks_cold_base/shows_cold], 
                                        'Ctr'  : [clicks_base/imps],
                                      })
                ad_df = pd.concat([ad_df, df_result])

                df_result = pd.DataFrame({'dataset': [dataset_name], 'model': [name], 'cold_period' : [T], 'LTR': ['UCB-FE'],
                                        'ColdImps' : [imps_cold_fe_ucb/shows_cold], 
                                        'ColdCtr'  : [clicks_cold_fe_ucb/shows_cold],
                                        'Ctr' : [clicks_fe_ucb/imps],
                                      })
                ad_df = pd.concat([ad_df, df_result])

                df_result = pd.DataFrame({'dataset': [dataset_name], 'model': [name], 'cold_period' : [T], 'LTR': ['UCB'],
                                        'ColdImps' : [imps_cold_ucb/shows_cold], 
                                        'ColdCtr'  : [clicks_cold_ucb/shows_cold],
                                        'Ctr' : [clicks_ucb/imps],
                                      })
                ad_df = pd.concat([ad_df, df_result])

        result = ad_df

        # Calculate ROC AUC and Stratified AUC
        df_grouped = df.groupby('request_id')
        strat_auc_score = StratifiedAUC('target', 'request_id')
        ndcg_score = NDCG(target_column = 'target', discount_base = 0.8)

        for name in tqdm(model_names, desc="Calculating ROC AUC metrics for models"):
            ltr = 'baseline'
            print(f"Call roc_auc, strat roc auc, ndcg for df with shape {df['target'].shape} \n")
            auc_old = roc_auc_score(df['target'], df['base_ctr_pred_' + name])
            # print(df['target'].shape)
            strat_auc_old = df_grouped.apply(strat_auc_score, rank_column= 'base_ctr_pred_' + name)
            # print(df['target'].shape)
            ndcg_old = df_grouped.apply(ndcg_score, rank_column= 'base_ctr_pred_' + name)

            for T in tqdm(cold_periods, desc=f"Processing cold periods for ROC AUC, Strat ROC AUC, NDCG calculation", leave=False):
                print("period = ", T)

                # Calculate coldNDCG for Baseline
                df['cold_target'] = df['target'] * (1 + df['cold_' + str(T)])
                cold_ndcg_score = NDCG(target_column='cold_target', discount_base=0.8)
                cold_ndcg_old = df_grouped.apply(cold_ndcg_score, rank_column='base_ctr_pred_' + name)

                df_result = pd.DataFrame({'dataset': [dataset_name], 'model': [name], 'cold_period' : [T], 'LTR': ['Baseline'],
                                    'RocAuc' : [auc_old], 
                                    'StrAuc' : [strat_auc_old.mean()],
                                    'NDCG' : [ndcg_old.mean()],
                                    'coldNDCG' : [cold_ndcg_old.mean()]})
                result2 = pd.concat([result2, df_result])

                auc = roc_auc_score(df['target'], df['ucb_ctr_pred_T_'+str(T) + '_' + name])
                strat_auc = df_grouped.apply(strat_auc_score, rank_column= 'ucb_ctr_pred_T_'+str(T) + '_' + name)
                ndcg = df_grouped.apply(ndcg_score, rank_column= 'ucb_ctr_pred_T_'+str(T) + '_' + name)

                # Calculate coldNDCG for UCB
                cold_ndcg = df_grouped.apply(cold_ndcg_score, rank_column='ucb_ctr_pred_T_'+str(T) + '_' + name)

                df_result = pd.DataFrame({'dataset': [dataset_name], 'model': [name], 'cold_period' : [T], 'LTR': ['UCB'],
                                        'RocAuc' : [auc], 
                                        'StrAuc' : [strat_auc.mean()],
                                        'NDCG' : [ndcg.mean()],
                                        'coldNDCG' : [cold_ndcg.mean()]})
                result2 = pd.concat([result2, df_result])

                auc = roc_auc_score(df['target'], df['fe_ucb_ctr_pred_T_'+str(T) + '_' + name])
                strat_auc = df_grouped.apply(strat_auc_score, rank_column= 'fe_ucb_ctr_pred_T_'+str(T) + '_' + name)
                ndcg = df_grouped.apply(ndcg_score, rank_column= 'fe_ucb_ctr_pred_T_'+str(T) + '_' + name)

                # Calculate coldNDCG for UCB-FE
                cold_ndcg = df_grouped.apply(cold_ndcg_score, rank_column='fe_ucb_ctr_pred_T_'+str(T) + '_' + name)

                df_result = pd.DataFrame({'dataset': [dataset_name], 'model': [name], 'cold_period' : [T], 'LTR': ['UCB-FE'],
                                        'RocAuc' : [auc], 
                                        'StrAuc' : [strat_auc.mean()],
                                        'NDCG' : [ndcg.mean()],
                                        'coldNDCG' : [cold_ndcg.mean()]})
                result2 = pd.concat([result2, df_result])

    result = result.merge(result2, on = ['dataset', 'model', 'cold_period', 'LTR'], how = 'left')
    return result

def calculate_relative_improvements(result):
    """
    Calculate relative improvements in metrics for LTR methods relative to Baseline.

    Args:
        result (pd.DataFrame): Combined results with all metrics

    Returns:
        pd.DataFrame: Results with relative improvements
    """
    # Calculate relative improvement in metrics for LTR methods relative to Baseline
    metrics = ['ColdImps', 'ColdCtr', 'Ctr', 'RocAuc', 'StrAuc', 'NDCG', 'coldNDCG']

    # Get Baseline values
    baseline = result[result['LTR'] == 'Baseline'].copy()

    # Get LTR values (UCB and UCB-FE)
    ltr_methods = result[result['LTR'] != 'Baseline'].copy()

    # Identify key columns for merging
    key_cols = ['model', 'cold_period']
    if 'dataset' in result.columns:
        key_cols.append('dataset')

    # Merge baseline with LTR methods on key columns
    merged = ltr_methods.merge(
        baseline[key_cols + metrics],
        on=key_cols,
        suffixes=('_ltr', '_baseline')
    )

    result_new = pd.DataFrame()

    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="Calculating relative improvements"):
        improvements = {}
        for col in key_cols:
            improvements[col] = row[col]
        improvements['LTR'] = row['LTR']

        # Calculate percentage improvement for each metric
        for metric in metrics:
            baseline_col = f'{metric}_baseline'
            ltr_col = f'{metric}_ltr'

            if baseline_col in row and ltr_col in row:
                baseline_val = row[baseline_col]
                ltr_val = row[ltr_col]

                # Calculate percentage improvement: (LTR - Baseline) / Baseline * 100
                if baseline_val != 0 and not pd.isna(baseline_val) and not pd.isna(ltr_val):
                    improvement = ((ltr_val - baseline_val) / baseline_val) * 100
                else:
                    improvement = np.nan

                improvements[metric] = improvement

        result_new = pd.concat([result_new, pd.DataFrame([improvements])], ignore_index=True)

    # Add Baseline rows from result (with 0% improvement)
    baseline_rows = baseline[key_cols + ['LTR'] + metrics].copy()
    # For Baseline, all improvements are 0%
    result_new = pd.concat([result_new, baseline_rows], ignore_index=True)
    result_new = result_new[['model', 'cold_period', 'dataset', 'LTR','StrAuc','RocAuc','NDCG', 'Ctr', 'ColdCtr', 'ColdImps', 'coldNDCG']]
    result_new = result_new.sort_values(by = ['model', 'cold_period', 'dataset', 'LTR'])

    # Round values: Baseline to 3 decimal places, others to 1 decimal place
    metrics_to_round = ['StrAuc', 'RocAuc', 'NDCG', 'coldNDCG', 'Ctr', 'ColdCtr', 'ColdImps']

    # Create a copy to avoid modifying the original
    result_new_rounded = result_new.copy()

    # Round Baseline rows to 3 decimal places
    baseline_mask = result_new_rounded['LTR'] == 'Baseline'
    for metric in metrics_to_round:
        if metric in result_new_rounded.columns:
            result_new_rounded.loc[baseline_mask, metric] = result_new_rounded.loc[baseline_mask, metric].round(3)

    # Round other rows to 1 decimal place
    other_mask = result_new_rounded['LTR'] != 'Baseline'
    for metric in metrics_to_round:
        if metric in result_new_rounded.columns:
            result_new_rounded.loc[other_mask, metric] = result_new_rounded.loc[other_mask, metric].round(1)

    # Convert rounded values to strings
    metrics_to_round = ['StrAuc', 'RocAuc', 'NDCG', 'coldNDCG', 'Ctr', 'ColdCtr', 'ColdImps']

    # Convert Baseline rows to strings (without %)
    baseline_mask = result_new_rounded['LTR'] == 'Baseline'
    for metric in metrics_to_round:
        if metric in result_new_rounded.columns:
            result_new_rounded.loc[baseline_mask, metric] = result_new_rounded.loc[baseline_mask, metric].astype(str)

    # Convert other rows to strings and add '%' at the end
    other_mask = result_new_rounded['LTR'] != 'Baseline'
    for metric in metrics_to_round:
        if metric in result_new_rounded.columns:
            result_new_rounded.loc[other_mask, metric] = result_new_rounded.loc[other_mask, metric].astype(str) + '%'

    return result_new_rounded

def save_final_results(result_new_rounded, dataset_name, result_dir, loaded_model_names):
    """
    Save the final results to a CSV file.

    Args:
        result_new_rounded (pd.DataFrame): Final results with improvements
        dataset_name (str): Name of the dataset
        result_dir (str): Directory where results are stored
    """
    final_result_path = os.path.join(result_dir, 'final_results', f'final_results_{dataset_name}_{loaded_model_names}.csv')
    result_new_rounded.to_csv(final_result_path, index=False)
    print(f"Final results saved to: {final_result_path}")
    return final_result_path

def run_prepare_results_pipeline(dataset_id='my_dataset', model_names=None, df_path=None, result_dir=None,
                                cold_periods=None, delta=1.5, top_n=10, tail_m=30):
    """
    Run the prepare results pipeline with specified dataset.
    This function can be called directly from the pipeline without argument parsing.

    Args:
        dataset_id (str): Identifier for the dataset to use
        model_names (list): List of model names
        df_path (str): Path to the dataframe file
        result_dir (str): Directory where results are stored
        cold_periods (list): List of cold start periods to test
        delta (float): Delta parameter for UCB
        top_n (int): Top N positions for metrics
        tail_m (int): Tail M positions for metrics

    Returns:
        str: Path to the final results file
    """
    # Set up paths
    current_file_dir = os.path.dirname(__file__)  # src/run/
    src_dir = os.path.dirname(current_file_dir)    # src/
    project_root = os.path.dirname(src_dir)        # Project root (where data/ is)

    if result_dir is None:
        result_dir = os.path.join(project_root, 'data', 'results')

    print(f"Processing results for dataset: {dataset_id}")
    print(f"Results directory: {result_dir}")

    try:
        # Load data
        df, loaded_model_names, loaded_cold_periods = load_metrics_data(df_path)

        # Use provided parameters or loaded ones
        if model_names is None:
            model_names = loaded_model_names
        if cold_periods is None:
            cold_periods = loaded_cold_periods

        # Calculate all metrics
        result = calculate_metrics(df, model_names, cold_periods, dataset_id)

        # Calculate relative improvements
        result_new_rounded = calculate_relative_improvements(result)

        # Save final results
        final_path = save_final_results(result_new_rounded, dataset_id, result_dir, loaded_model_names)

        print("\n" + "=" * 60)
        print("✓ Final results processing completed successfully!")
        print(f"✓ Results saved to: {final_path}")
        print("=" * 60)

        return final_path
    except Exception as e:
        print(f"\n✗ Error processing results: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main function to process results for a specified dataset."""
    parser = argparse.ArgumentParser(description='Prepare final results for a specified dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name to process')
    parser.add_argument('--result-dir', type=str, default=None,
                        help='Directory where results are stored')

    args = parser.parse_args()

    # Set up paths
    current_file_dir = os.path.dirname(__file__)  # src/run/
    src_dir = os.path.dirname(current_file_dir)    # src/
    project_root = os.path.dirname(src_dir)        # Project root (where data/ is)

    if args.result_dir is None:
        result_dir = os.path.join(project_root, 'data', 'results')
    else:
        result_dir = args.result_dir

    print(f"Processing results for dataset: {args.dataset}")
    print(f"Results directory: {result_dir}")

    try:
        # Load data
        df, model_names, cold_periods = load_metrics_data(args.dataset, result_dir)

        # Calculate all metrics
        result = calculate_metrics(df, model_names, cold_periods, args.dataset)

        # Calculate relative improvements
        result_new_rounded = calculate_relative_improvements(result)

        # Save final results
        final_path = save_final_results(result_new_rounded, args.dataset, result_dir)

        print("\n" + "=" * 60)
        print("✓ Final results processing completed successfully!")
        print(f"✓ Results saved to: {final_path}")
        print("=" * 60)

        return 0
    except Exception as e:
        print(f"\n✗ Error processing results: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
