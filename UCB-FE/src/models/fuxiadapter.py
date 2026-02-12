import numpy as np
import pandas as pd
import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"DEBUG: Current dir for fuxiadapter: {current_dir}")
print(f"DEBUG: Python path: {sys.path[:2]}")

from .ml import ML
from .feature_transformer import FeatureTransformer
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset


class FuxiUniversalAdapter(ML):
    """Working adapter for Fuxi Library"""
    
    def __init__(self, X_train, y_train, categorical_features, model_class, config_path=None,
                  experiment_id='FinalNet_test'):
        print(f"\nDEBUG: Initializing FuxiUniversalAdapter...")
        print(f"DEBUG: model_class = {model_class}")
        print(f"DEBUG: model_class type = {type(model_class)}")
        
        if model_class is None:
            raise ValueError("model_class cannot be None. Fuxi model might not be imported correctly.")
        
        # Call parent constructor
        super().__init__(X_train, y_train, categorical_features, model_name=experiment_id)
        
        self.config_path = config_path
        params = load_config(self.config_path, experiment_id)
        print(params['data_root'], params['dataset_id'], self.config_path, experiment_id)


        set_logger(params)
        seed_everything(seed=params['seed'])

        print(params['data_root'], params['dataset_id'])

        data_dir = os.path.join(params['data_root'], params['dataset_id'])
        feature_map_json = os.path.join(data_dir, "feature_map.json")
        if params["data_format"]:
            # Build feature_map and transform data
            feature_encoder = FeatureProcessor(**params)
            params["train_data"], params["valid_data"], params["test_data"] = \
                build_dataset(feature_encoder, **params)
        feature_map = FeatureMap(params['dataset_id'], data_dir)
        feature_map.load(feature_map_json, params)
        
        self.model_class = model_class
        self.model = model_class(feature_map, **params)
        print(f"MODEL {experiment_id, model_class} IS ON THE {self.model.device}")
        self.model.count_parameters() # print number of parameters used in model

        self.train_gen, self.valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
        self.test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        self.params = params
        self.feature_map = feature_map
        
        # Try to load model weights, but don't fail if not found
        # try:
        #     self.model.load_weights(self.model.checkpoint)
        #     print(f"✓ Loaded model weights from {self.model.checkpoint}")
        # except FileNotFoundError:
        #     print(f"⚠ Warning: Model checkpoint not found at {self.model.checkpoint}")
        #     print("  Model will need to be trained or weights need to be placed in the correct location")
        
        print("✓ FuxiUniversalAdapter initialized successfully")
    
    def fit(self):
        """Train the model - simplified version"""
        print("FuxiUniversalAdapter.fit() - Training model...")
        
        # For now, create a dummy model
        self.model.fit(self.train_gen, validation_data=self.valid_gen, **self.params)
        print("✓ Model 'trained'")
        
    
    def predict(self, X_test, y_test, ctr_values=None):
        """
        Make predictions with optional CTR value modifications - optimized version
        
        Args:
            X_test: Test features DataFrame
            y_test: Test labels
            ctr_values: Optional CTR values to use instead of existing ones
        """
        print(f"FuxiUniversalAdapter.predict() called for {len(X_test)} samples")

        # print(f"Temp path is {self.config_path['valid_data']}\n\n {self.config_path['dataset_id']}")
        # NEW_PATH = self.config_path['valid_data'] #'/home/adpudovikov/project/UCB-FE/data/my_dataset/valid.parquet'
        NEW_PATH = f'/home/adpudovikov/project/UCB-FE/data/{self.params["dataset_id"]}/valid.parquet'

        # del self.train_gen
        # del self.valid_gen
        # Prepare test data with potential CTR modifications
        if ctr_values is not None:
            X_test_modified = X_test.copy()
            # print(ctr_values, X_test_modified)
            X_test_modified['ctr'] = ctr_values
        else:
            X_test_modified = X_test.copy()  # Ensure we always work with a copy

        # Add target column and save to temporary parquet file
        X_test_modified['target'] = y_test
        X_test_modified.to_parquet(NEW_PATH, index=False)
        # print(X_test_modified.columns)
        
        # Create a new DataLoader that uses our modified test data
        # We only need the test generator since we're just predicting
        tmp_gen = RankDataLoader(
            self.feature_map, 
            stage='test', 
            test_data=NEW_PATH,#self.params['valid_data'],  # Use our modified parquet file
            batch_size=self.params.get('batch_size', 1024),  # Use batch_size from params or default
            shuffle=False,
            data_format=self.params.get('data_format', 'parquet')  # Use the same data format as training
        ).make_iterator()

        # Make predictions using the modified data
        print(len(tmp_gen))
        answer = self.model.predict(tmp_gen)
        print(answer.shape, answer, np.sum(answer), np.unique(answer))
        return answer
    
    def _load_model_(self, prefix=None):
        """Load pre-trained model"""
        print("FuxiUniversalAdapter._load_model_() called")
        
        # Try to load model weights, but don't fail if not found
        try:
            self.model.load_weights(self.model.checkpoint)
            print(f"✓ Loaded model weights from {self.model.checkpoint}")
            self.loaded = True
        except FileNotFoundError:
            self.fit()
            print(f"⚠ Warning: Model checkpoint not found at {self.model.checkpoint}")
            print("  Model will need to be trained or weights need to be placed in the correct location")
            self.loaded = False
