#!/usr/bin/env python3
"""
Pipeline controller for UCB-FE experiments with different datasets
"""
import sys
import os
import argparse
from tqdm import tqdm

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/run/
src_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(src_dir)  # UCB-FE/

# Add to sys.path
sys.path.insert(0, src_dir)  # Add src directory
sys.path.insert(0, os.path.join(src_dir, 'models'))  # Add models directory

    # Import pipeline components
from build import build_for_pipeline
from predict_ctr import run_prediction_pipeline
from prepare_results import run_prepare_results_pipeline


def run_pipeline(dataset_id='my_dataset', experiment_id='FinalNet_test', 
                 config_path=['src', 'models', 'config_finalnet'], result_dir=None,
                 cold_periods=None, delta=1.5, top_n=10, tail_m=30):
    """
    Run the complete UCB-FE pipeline for a specified dataset
    
    Args:
        dataset_id (str): Identifier for the dataset to use
        experiment_id (str): Experiment identifier for model configuration
        config_path (list): Path to configuration directory
        result_dir (str): Directory to save results (optional)
        cold_periods (list): List of cold start periods to test
        delta (float): Delta parameter for UCB
        top_n (int): Top N positions for metrics
        tail_m (int): Tail M positions for metrics
    """
    print(f"Starting UCB-FE pipeline for dataset: {dataset_id}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Cold periods: {cold_periods}")
    print(f"Delta: {delta}")
    print(f"Top N: {top_n}")
    print(f"Tail M: {tail_m}")
    print("=" * 60)
    
    # Step 1: Build dataset
    print("\nStep 1: Building dataset...")
    try:
        success = build_for_pipeline(
            experiment_id=experiment_id,
            config_path=config_path,
            dataset_id=dataset_id
        )
        if not success:
            print("✗ Dataset building failed")
            return False
        print("✓ Dataset building completed")
    except Exception as e:
        print(f"✗ Dataset building failed with error: {str(e)}")
        return False
    
    # Step 2: Run prediction pipeline
    print("\nStep 2: Running prediction pipeline...")
    try:
        model_names, df_path = run_prediction_pipeline(
            dataset_id=dataset_id,
            result_dir=result_dir,
            cold_periods=cold_periods,
            delta=delta,
            top_n=top_n,
            tail_m=tail_m,
            experiment_id=experiment_id,
            config_path=config_path,
        )
        print("✓ Prediction pipeline completed")
        print(f"  Generated results file: {df_path}")
    except Exception as e:
        print(f"✗ Prediction pipeline failed with error: {e}")
        print(e)
        return False
    
    # Step 3-5: Run prepare results pipeline (combines ROC AUC, NDCG, and custom metrics evaluation)
    print("\nStep 3-5: Running prepare results pipeline...")
    try:
        final_results_path = run_prepare_results_pipeline(
            dataset_id=dataset_id,
            model_names=model_names,
            df_path=df_path,
            result_dir=result_dir,
            cold_periods=cold_periods,
            delta=delta,
            top_n=top_n,
            tail_m=tail_m
        )
        print("✓ Prepare results pipeline completed")
        print(f"  Final results saved to: {final_results_path}")
    except Exception as e:
        print(f"✗ Prepare results pipeline failed with error: {str(e)}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All pipeline steps completed successfully!")
    print("=" * 60)
    return True


def main():
    """Main function to parse arguments and run pipeline"""
    parser = argparse.ArgumentParser(description='Run UCB-FE pipeline for specified dataset')
    parser.add_argument('--dataset', type=str, default='my_dataset',
                        help='Dataset identifier (default: my_dataset)')
    parser.add_argument('--experiment', type=str, default='FinalNet_test',
                        help='Experiment ID (default: FinalNet_test)')
    parser.add_argument('--config-path', type=str, nargs='+', 
                        default=['src', 'models', 'config_finalnet'],
                        help='Path to config directory (default: src/models/config_finalnet)')
    parser.add_argument('--result-dir', type=str, default=None,
                        help='Directory to save results (default: data/results)')
    parser.add_argument('--cold-periods', type=int, nargs='+', default=None,
                        help='Cold start periods to test (default: [0, 10, 100])')
    parser.add_argument('--delta', type=float, default=1.5,
                        help='Delta parameter for UCB (default: 1.5)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Top N positions for metrics (default: 10)')
    parser.add_argument('--tail-m', type=int, default=30,
                        help='Tail M positions for metrics (default: 30)')
    
    args = parser.parse_args()
    
    success = run_pipeline(
        dataset_id=args.dataset,
        experiment_id=args.experiment,
        config_path=args.config_path,
        result_dir=args.result_dir,
        cold_periods=args.cold_periods,
        delta=args.delta,
        top_n=args.top_n,
        tail_m=args.tail_m
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
