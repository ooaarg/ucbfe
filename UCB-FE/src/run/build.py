#!/usr/bin/env python3
"""
Test script to verify the fast Fuxi optimization implementation
"""
import numpy as np
import pandas as pd
import warnings
import sys
import os
import time


current_dir = os.path.dirname(os.path.abspath(__file__))  # src/run/
src_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(src_dir)  # UCB-FE/

# Add to sys.path
sys.path.insert(0, src_dir)  # Add src directory
sys.path.insert(0, os.path.join(src_dir, 'models'))  # Add models directory


from models.fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from models.fuxictr.pytorch.torch_utils import seed_everything
from models.fuxictr.preprocess import FeatureProcessor, build_dataset


def build_for_pipeline(experiment_id='FinalNet_test', config_path=['src', 'models', 'config_finalnet'], dataset_id=None):
    """The main function used to build dataset before the main inference"""    
    
    # Load config
    CONFIG_DIR = os.path.join(project_root, *config_path)
    
    print(CONFIG_DIR)
    params = load_config(CONFIG_DIR, experiment_id)
    
    # Override dataset_id if provided
    if dataset_id is not None:
        params['dataset_id'] = dataset_id

    set_logger(params)
    seed_everything(seed=params['seed'])

    # Build feature_map and transform data
    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(feature_encoder, **params)
    success = params["train_data"] is not None
    
    return success

def main():
    """Main test function"""
    warnings.filterwarnings("ignore")
    
    try:
        success = build_for_pipeline()

        if success:
            print("\n✓ Successful build of the dataset!")
        else:
            print("\n✗ Build of the dataset test failed")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
