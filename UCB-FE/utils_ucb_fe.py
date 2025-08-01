import math
import random
import warnings
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import os

warnings.simplefilter('ignore')
from tabm_reference import Model, make_parameter_groups

warnings.resetwarnings()
from abc import ABC, abstractmethod

def log_model_operation(func):
    def wrapper(self, *args, **kwargs):
        # Print before execution
        print(f"🏁 Starting {func.__name__} on model: {getattr(self, 'model_name', 'Unknown')}")
        # print(f"   Args: {args}, Kwargs: {kwargs}")
        
        # Execute the function
        result = func(self, *args, **kwargs)
        
        # Print after execution
        print(f"✅ Finished {func.__name__} on model: {getattr(self, 'model_name', 'Unknown')}")
        print(f"   Returned: {result}")
        
        return result
    return wrapper

def log_model_weights(func):
    def wrapper(self, *args, **kwargs):
        # Execute the function
        result = func(self, *args, **kwargs)
        
        # Log model weights after execution
        self._log_model_()
        
        return result
    return wrapper



class ML(ABC):
    """
    Base class for ML algorithms
    """
    def __init__(self, X_train, y_train, categorical_features, model_name="ML"):
        print(f"Initializing {model_name} model")
        self.model_name = model_name
        
        # Handle both pandas DataFrame and numpy array for X_train
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.values
        else:
            self.X_train = X_train
            
        # Handle both pandas Series and numpy array for y_train
        if isinstance(y_train, pd.Series):
            self.y_train = y_train.values
        else:
            self.y_train = y_train

        self.categorical_features = categorical_features
        self.model = None
        self.ctr = None
        self.X_test = None
        self.loaded = False


    @abstractmethod
    def fit(self):
        """
        fit a model
        """
        print(f"Fitting {self.model_name} model")
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """
        predict_proba
        """
        print(f"Predicting {self.model_name} model")
        # Handle both pandas DataFrame and numpy array for X_test
        if isinstance(X_test, pd.DataFrame):
            self.X_test = X_test
        else:
            self.X_test = pd.DataFrame(X_test)

    @log_model_operation
    def _log_model_(self):
        """Save model weights"""
        if not os.path.exists('weights'):
            os.makedirs('weights')
        
        if isinstance(self.model, TabNetClassifier):
            # Remove any existing file to avoid conflicts
            if os.path.exists('weights/tabnet.pth.zip'):
                os.remove('weights/tabnet.pth.zip')
            self.model.save_model('weights/tabnet.pth.zip')
            print(f"Saved TabNet model to weights/tabnet.pth.zip")
        elif isinstance(self.model, LGBMClassifier):
            self.model.booster_.save_model('weights/lightgbm.txt')
            print(f"Saved LightGBM model to weights/lightgbm.txt")
        elif isinstance(self.model, XGBClassifier):
            self.model.save_model('weights/xgboost.json')
            print(f"Saved XGBoost model to weights/xgboost.json")
        elif isinstance(self.model, CatBoostClassifier):
            self.model.save_model('weights/catboost.cbm')
            print(f"Saved CatBoost model to weights/catboost.cbm")
        else:
            torch.save(self.model.state_dict(), 'weights/tabm.pth')
            print(f"Saved TabM model to weights/tabm.pth")
    
    @log_model_operation
    def _load_model_(self):
        """Load model weights"""
        if isinstance(self.model, TabNetClassifier):
            if os.path.exists('weights/tabnet.pth.zip'):
                self.model.load_model('weights/tabnet.pth.zip')
                print(f"Loaded TabNet model from weights/tabnet.pth.zip")
            else:
                print("No saved TabNet model found")
        elif isinstance(self.model, LGBMClassifier):
            if os.path.exists('weights/lightgbm.txt'):
                # Load the booster model
                with open('weights/lightgbm.txt', 'r') as f:
                    model_str = f.read()
                    print(f"\nSuccessfully read model file, size: {len(model_str)} bytes")
                
                # Create basic parameters
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'verbose': -1
                }
                                
                print("\nTrying to create Booster with model string...")
                self.model = lgb.Booster(params=params, model_str=model_str)

                print(f"Loaded LightGBM model from weights/lightgbm.txt")
            else:
                print("No saved LightGBM model found")
        elif isinstance(self.model, XGBClassifier):
            if os.path.exists('weights/xgboost.json'):
                # Create a new booster from the saved model
                booster = xgb.Booster()
                booster.load_model('weights/xgboost.json')
                # Update the XGBClassifier with the loaded booster
                self.model._Booster = booster
                print(f"Loaded XGBoost model from weights/xgboost.json")
            else:
                print("No saved XGBoost model found")
        elif isinstance(self.model, CatBoostClassifier):
            if os.path.exists('weights/catboost.cbm'):
                self.model.load_model('weights/catboost.cbm')
                print(f"Loaded CatBoost model from weights/catboost.cbm")
            else:
                print("No saved CatBoost model found")
        else:
            if os.path.exists('weights/tabm.pth'):
                self.model.load_state_dict(torch.load('weights/tabm.pth', map_location=torch.device("cpu")))
                print(f"Loaded TabM model from weights/tabm.pth")
            else:
                print("No saved TabM model found")
        self.loaded = True



class TabMModel(ML):
    """
    TabM model implementation for classification tasks
    """
    @log_model_operation
    def __init__(self, X_train, y_train, categorical_features=None):
        categorical_features = categorical_features or []  # Convert None to empty list
        super().__init__(X_train, y_train, categorical_features, model_name="TabM")
        
        # Set random seeds for reproducibility
        seed = 0
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        
        # Convert data to float32, handling both pandas and numpy inputs
        if isinstance(X_train, pd.DataFrame):
            # Separate numerical and categorical features
            numerical_features = [col for col in X_train.columns if col not in categorical_features]
            self.X_train_num = X_train[numerical_features].values.astype(np.float32)
            if categorical_features:
                self.X_train_cat = X_train[categorical_features].values.astype(np.int64)
                self.X_train_cat_tensor = None  # Will be set after preprocessing
                # Store categorical feature cardinalities
                self.cat_cardinalities = [X_train[col].nunique() for col in categorical_features]
            else:
                self.X_train_cat = None
                self.X_train_cat_tensor = None
                self.cat_cardinalities = []
        else:
            self.X_train_num = X_train.astype(np.float32)
            self.X_train_cat = None
            self.X_train_cat_tensor = None
            self.cat_cardinalities = []
            
        if isinstance(y_train, pd.Series):
            self.y_train = y_train.values.astype(np.float32)
        else:
            self.y_train = y_train.astype(np.float32)
        
        # Feature preprocessing for numerical features
        noise = (
            np.random.default_rng(0)
            .normal(0.0, 1e-5, self.X_train_num.shape)
            .astype(self.X_train_num.dtype)
        )
        self.preprocessing = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(len(self.X_train_num) // 30, 1000), 10),
            output_distribution='normal',
            subsample=10**9,
        ).fit(self.X_train_num + noise)
        
        # Transform training data
        self.X_train_num = self.preprocessing.transform(self.X_train_num)
                
        # PyTorch settings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        
        # Convert data to tensors with explicit dtype
        self.X_train_num_tensor = torch.as_tensor(self.X_train_num, dtype=self.dtype, device=self.device)
        if self.X_train_cat is not None:
            self.X_train_cat_tensor = torch.as_tensor(self.X_train_cat, dtype=torch.int64, device=self.device)
        self.y_train_tensor = torch.as_tensor(self.y_train, dtype=self.dtype, device=self.device)
        
        # Model configuration
        self.model = Model(
            n_num_features=self.X_train_num.shape[1],
            cat_cardinalities=self.cat_cardinalities,
            n_classes=2,  
            backbone={
                'type': 'MLP',
                'n_blocks': 3,
                'd_block': 512,
                'dropout': 0.1,
            },
            bins=None,
            num_embeddings=None,
            arch_type='tabm',
            k=32,
            share_training_batches=True,
        ).to(device=self.device, dtype=self.dtype)
        
        self.optimizer = torch.optim.AdamW(make_parameter_groups(self.model), lr=2e-3, weight_decay=3e-4)
        
        # Training parameters
        self.n_epochs = 100
        self.patience = 16
        self.batch_size = 256
        self.best_model_state = None
        self.best_val_score = - math.inf
        
    def _apply_model(self, X_num: Tensor, X_cat: Optional[Tensor] = None) -> Tensor:
        """Apply the model to input data"""
        output = self.model(X_num, X_cat)
        if len(output.shape) == 3:  # If output has shape [batch, k, 2]
            output = output.mean(1)  # Average over k predictions
        return output.float()
    
    def _loss_fn(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the loss function for binary classification"""
        if len(y_pred.shape) == 3:  # If predictions have shape [batch, k, 2]
            y_pred = y_pred.mean(1)  # Average over k predictions
        if len(y_pred.shape) == 2:  # If predictions have shape [batch, 2]
            y_pred = y_pred[:, 1]  # Take probability of class 1
        y_true = y_true.float()
        return F.binary_cross_entropy_with_logits(y_pred, y_true)
    
    def _evaluate(self, X_num: Tensor, X_cat: Optional[Tensor] = None, y: Optional[Tensor] = None) -> float:
        """Evaluate the model on given data"""
        self.model.eval()
        
        eval_batch_size = 8096
        with torch.no_grad():
            y_pred = torch.cat([
                self._apply_model(
                    batch_num,
                    batch_cat if X_cat is not None else None
                )
                for batch_num, batch_cat in tqdm(zip(
                    X_num.split(eval_batch_size),
                    [None] * len(X_num.split(eval_batch_size)) if X_cat is None else X_cat.split(eval_batch_size)
                ), desc="Evaluating model")
            ])
            
            if len(y_pred.shape) == 2:  # If predictions have shape [batch, 2]
                y_pred = y_pred[:, 1]  # Take probability of class 1
            y_pred = torch.sigmoid(y_pred).cpu().numpy()
        
        if y is not None:
            y_true = y.cpu().numpy()
            # Use AUC-ROC score for binary classification
            score = sklearn.metrics.roc_auc_score(y_true, y_pred)
            return float(score)
        return y_pred
    
    @log_model_weights
    @log_model_operation
    def fit(self):
        """Train the model"""
        train_size = len(self.X_train_num_tensor)
        best = {
            'val': -math.inf,
            'test': -math.inf,
            'epoch': -1,
        }
        remaining_patience = self.patience
        
        # Create validation split
        train_idx, val_idx = sklearn.model_selection.train_test_split(
            np.arange(train_size), train_size=0.8
        )
        X_val_num = self.X_train_num_tensor[val_idx]
        X_val_cat = self.X_train_cat_tensor[val_idx] if self.X_train_cat_tensor is not None else None
        y_val = self.y_train_tensor[val_idx]
        
        print('-' * 88 + '\n')
        for epoch in range(self.n_epochs): # self.n_epochs
            # Create batches
            batches = (
                torch.randperm(train_size, device=self.device).split(self.batch_size)
                if self.model.share_training_batches
                else [
                    x.transpose(0, 1).flatten()
                    for x in torch.rand((self.model.k, train_size), device=self.device)
                    .argsort(dim=1)
                    .split(self.batch_size, dim=1)
                ]
            )
            
            # Training loop
            for batch_idx in tqdm(batches, desc=f'Epoch {epoch}'):
                self.model.train()
                self.optimizer.zero_grad()
                loss = self._loss_fn(
                    self._apply_model(
                        self.X_train_num_tensor[batch_idx],
                        self.X_train_cat_tensor[batch_idx] if self.X_train_cat_tensor is not None else None
                    ),
                    self.y_train_tensor[batch_idx]
                )
                loss.backward()
                self.optimizer.step()
            
            # Evaluation
            val_score = self._evaluate(X_val_num, X_val_cat, y_val)
            print(f'(val) {val_score:.4f}')
            
            if val_score > best['val']:
                print('🌸 New best epoch! 🌸')
                best = {'val': val_score, 'epoch': epoch}
                self.best_model_state = self.model.state_dict().copy()
                remaining_patience = self.patience
            else:
                remaining_patience -= 1
            
            if remaining_patience < 0:
                break
            
            print()
        
        # Load best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
    
    @log_model_operation
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features
        
        Returns:
            Predicted probabilities
        """
        self.X_test = X_test
        
        # Convert data to appropriate format
        if isinstance(self.X_test, pd.DataFrame):
            # Separate numerical and categorical features
            numerical_features = [col for col in self.X_test.columns if col not in self.categorical_features]
            X_test_num = self.X_test[numerical_features].values.astype(np.float32)
            if self.categorical_features:
                X_test_cat = self.X_test[self.categorical_features].values.astype(np.int64)
                # Clip categorical values to match training cardinalities
                for i, cardinality in enumerate(self.cat_cardinalities):
                    X_test_cat[:, i] = np.clip(X_test_cat[:, i], 0, cardinality - 1)
            else:
                X_test_cat = None
        else:
            X_test_num = self.X_test.astype(np.float32)
            X_test_cat = None

        # Preprocess numerical features
        X_test_num = self.preprocessing.transform(X_test_num)
        X_test_num = torch.tensor(X_test_num, device=self.device)

        # Preprocess categorical features if they exist
        if X_test_cat is not None:
            X_test_cat = torch.tensor(X_test_cat, device=self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self._evaluate(X_test_num, X_test_cat)
        
        return predictions



class Xgboost(ML):
    """
    XGBClassifier
    """
    @log_model_operation
    def __init__(self, X_train, y_train, categorical_features):
        super().__init__(X_train, y_train, categorical_features, "XGBoost")

        #prepare cat_features
        self.categorical_features = categorical_features
        for name in categorical_features:
            X_train.loc[:, name] = X_train[name].astype('category')
        
        #prepare train data
        self.X_train = X_train
        self.y_train = y_train

        #init model 
        self.model = XGBClassifier(enable_categorical=True, seed=42, eval_metric='logloss')

    @log_model_weights
    @log_model_operation
    def fit(self):
        super().fit()
        self.model.fit(self.X_train, self.y_train)

    @log_model_operation
    def predict(self, X_test):
        super().predict(X_test)
        
        #prepare test data
        self.X_test = X_test.copy()  # Create a copy to avoid SettingWithCopyWarning
        for name in self.categorical_features:
            self.X_test.loc[:, name] = self.X_test[name].astype('category')

        #predict proba
        if not self.loaded:
            self.ctr = self.model.predict_proba(self.X_test)[:,1]
        else:
            # self.ctr = self.model.predict(self.X_test)
            dtest = xgb.DMatrix(self.X_test)
            # Get raw predictions
            raw_preds = self.model._Booster.predict(dtest)
            # Convert to probabilities using sigmoid function
            self.ctr = 1.0 / (1.0 + np.exp(-raw_preds))

        return self.ctr

class Lightgbm(ML):
    """
    LGBMClassifier
    """
    @log_model_operation
    def __init__(self, X_train, y_train, categorical_features):
        super().__init__(X_train, y_train, categorical_features, "LightGBM")
        
        #prepare cat_features
        self.categorical_features = categorical_features
        for name in categorical_features:
            X_train.loc[:, name] = X_train[name].astype('category')
        
        #prepare train data
        self.X_train = X_train
        self.y_train = y_train
        
        # Get categorical feature indices
        cat_indices = [X_train.columns.get_loc(name) for name in categorical_features]
        
        #init model
        self.model = LGBMClassifier(random_state=42, categorical_feature=cat_indices)

    @log_model_weights
    @log_model_operation
    def fit(self):
        super().fit()
        self.model.fit(self.X_train, self.y_train)
        
    @log_model_operation
    def predict(self, X_test):
        super().predict(X_test)

        #prepare test data
        self.X_test = X_test.copy()  # Create a copy to avoid SettingWithCopyWarning
        for name in self.categorical_features:
            self.X_test.loc[:, name] = self.X_test[name].astype('category')

        #predict    
        if isinstance(self.model, lgb.Booster):
            # If model is a Booster (loaded from file), use predict directly
            self.ctr = self.model.predict(self.X_test)
        else:
            # If model is LGBMClassifier, use predict_proba
            self.ctr = self.model.predict_proba(self.X_test)[:, 1]
        return self.ctr
    
class Catboost(ML):
    """
    CatBoost model implementation for classification tasks
    """
    @log_model_operation
    def __init__(self, X_train, y_train, categorical_features=None):
        categorical_features = categorical_features or []  # Convert None to empty list
        super().__init__(X_train, y_train, categorical_features, model_name="CatBoost")
        
        # Convert data to pandas DataFrame if it's not already
        if not isinstance(X_train, pd.DataFrame):
            self.X_train = pd.DataFrame(X_train)
        else:
            self.X_train = X_train.copy()  # Make a copy to avoid modifying the original
        
        # Initialize CatBoost model with appropriate parameters
        self.model = CatBoostClassifier(
            loss_function='Logloss',
            random_seed=42,
            verbose=False
        )
    
    @log_model_operation
    def fit(self, train_data=None, train_labels=None, **kwargs):
        if train_data is None:
            train_data = self.X_train
        if train_labels is None:
            train_labels = self.y_train

        # Convert to DataFrame if numpy array
        if isinstance(train_data, np.ndarray):
            train_data = pd.DataFrame(train_data, columns=[f'feature_{i}' for i in range(train_data.shape[1])])
        else:
            train_data = train_data.copy()  # Make a copy to avoid modifying the original
            
        # Convert categorical columns to string type
        if self.categorical_features:
            for col in self.categorical_features:
                train_data[col] = train_data[col].astype(str)

        self.model.fit(
            train_data,
            train_labels,
            cat_features=self.categorical_features,
            verbose=False,
            plot=False
        )

    @log_model_operation
    def predict(self, test_data):
        # Convert to DataFrame if numpy array
        if isinstance(test_data, np.ndarray):
            test_data = pd.DataFrame(test_data, columns=[f'feature_{i}' for i in range(test_data.shape[1])])
        else:
            test_data = test_data.copy()  # Make a copy to avoid modifying the original
            
        # Convert categorical columns to string type
        if self.categorical_features:
            print(self.categorical_features)
            for col in self.categorical_features:
                test_data[col] = test_data[col].astype(str)

        return self.model.predict_proba(test_data)[:, 1]

class Tab_net(ML):
    """
    TabNetClassifier
    """
    # Model parameters that match the training configuration
    MODEL_PARAMS = {
        'n_d': 8,
        'n_a': 8,
        'n_steps': 3,
        'gamma': 1.3,
        'n_independent': 2,
        'n_shared': 2,
        'lambda_sparse': 1e-3,
        'momentum': 0.02,
        'clip_value': None,
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=2e-2),
        'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_params': dict(mode='min', patience=5, min_lr=1e-5, factor=0.5),
        'mask_type': 'sparsemax',
        'seed': 42,
        'verbose': 0
    }
    
    @log_model_operation
    def __init__(self, X_train, y_train, categorical_features):
        super().__init__(X_train, y_train, categorical_features, "TabNet")
        
        #prepare cat_features
        self.categorical_features = categorical_features
        self.categorical_idx = [X_train.columns.get_loc(column) for column in categorical_features]
        
        # Calculate cardinalities for categorical features
        self.cat_dims = [X_train[col].nunique() for col in categorical_features]
        
        #prepare train data
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.copy()
        else:
            self.X_train = pd.DataFrame(X_train)

        if isinstance(y_train, pd.Series):
            self.y_train = y_train.values.squeeze()
        else:
            self.y_train = y_train.squeeze()
        
        #init model
        self.model = TabNetClassifier(
            cat_idxs=self.categorical_idx,
            cat_dims=self.cat_dims,
            **self.MODEL_PARAMS
        )

    def _process_categorical_features(self, X):
        """Process categorical features to match training cardinalities"""
        X_processed = X.copy()
        for cat_feat, max_val in zip(self.categorical_features, self.cat_dims):
            if cat_feat in X_processed.columns:
                # Clip values to match training cardinalities
                X_processed[cat_feat] = X_processed[cat_feat].clip(0, max_val - 1)
        return X_processed

    @log_model_weights
    @log_model_operation
    def fit(self):
        super().fit()
        # Process categorical features
        X_train_processed = self._process_categorical_features(self.X_train)
        
        self.model.fit(
            X_train=X_train_processed.values,
            y_train=self.y_train,
            max_epochs=10,
            eval_metric=['auc'],
            compute_importance=False,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

    @log_model_operation
    def predict(self, X_test):
        super().predict(X_test)

        try:
            # Convert input to DataFrame if needed
            if not isinstance(X_test, pd.DataFrame):
                if hasattr(self, 'feature_names'):
                    X_test = pd.DataFrame(X_test, columns=self.feature_names)
                else:
                    X_test = pd.DataFrame(X_test)
            
            # Process categorical features
            X_test_processed = self._process_categorical_features(X_test)
            
            # Make prediction
            self.ctr = self.model.predict_proba(X_test_processed.values)[:,1]
            return self.ctr
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(f"Model categorical features: {self.categorical_features}")
            print(f"Model categorical dimensions: {self.cat_dims}")
            print(f"Input data shape: {X_test.shape}")
            if hasattr(self, 'feature_names'):
                print(f"Expected features: {self.feature_names}")
            raise

    @log_model_operation
    def _load_model_(self):
        """Load model weights"""
        if os.path.exists('weights/tabnet.pth.zip'):
            try:
                # Load the model first to get its configuration
                temp_model = TabNetClassifier()
                temp_model.load_model('weights/tabnet.pth.zip')
                
                # Get the model's configuration
                self.input_dim = temp_model.input_dim
                num_embeddings = len(temp_model.network.embedder.embeddings)
                
                # Adjust our configuration to match the model
                self.categorical_features = self.categorical_features[:num_embeddings]
                self.categorical_idx = self.categorical_idx[:num_embeddings]
                self.cat_dims = [
                    temp_model.network.embedder.embeddings[i].num_embeddings 
                    for i in range(num_embeddings)
                ]
                
                # Store feature names if available
                if hasattr(self, 'X_train'):
                    self.feature_names = self.X_train.columns.tolist()
                    
                    # Determine which features the model expects
                    num_features = self.input_dim
                    cat_features = self.categorical_features
                    
                    # Get numerical features (all features except categorical ones)
                    num_features_list = [col for col in self.feature_names if col not in cat_features]
                    
                    # Combine numerical and categorical features in the correct order
                    self.model_features = num_features_list[:num_features - len(cat_features)] + cat_features
                    
                    print(f"Model features: {self.model_features}")
                
                # Initialize our model with the correct configuration
                self.model = TabNetClassifier(
                    cat_idxs=self.categorical_idx,
                    cat_dims=self.cat_dims,
                    **self.MODEL_PARAMS
                )
                
                # Load the weights
                self.model.load_model('weights/tabnet.pth.zip')
                print(f"Loaded TabNet model from weights/tabnet.pth.zip")
                print(f"Model expects {self.input_dim} features and {num_embeddings} categorical features")
                print(f"Categorical features used: {self.categorical_features}")
                print(f"Categorical dimensions: {self.cat_dims}")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        else:
            print("No saved TabNet model found")




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
        'XGBoost': Xgboost(X, y, categorical_features),
        # 'CatBoost': Catboost(X, y, categorical_features),
        # 'TabNet': Tab_net(X, y, categorical_features)
    }
    
    for name, model in models.items():
        print(f"\nTesting {name} model:")
        
        # model.fit()
        # model._log_model_()
        # model._load_model_()
        # print(model.loaded)
        # predictions = model.predict(X[:100])
        # print(f"{name} sample predictions:", predictions[:5])
