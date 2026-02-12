import math
import random
from typing import Optional, Union, List
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import os
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm

from .ml import ML
from .ml import log_model_operation, log_model_weights
from .tabm_reference import Model, make_parameter_groups

class TabMModel(ML):
    """
    TabM model implementation for classification tasks
    """
    @log_model_operation
    def __init__(self, X_train, y_train, categorical_features=None):
        '''
        Provide a full train dataset for the model or a small sample with fitted=True
        '''
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
                self.cat_cardinalities = [X_train[col].max() for col in categorical_features]
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
        if torch.cuda.is_available():
            # Use the first GPU as default
            self.device = torch.device('cuda:0')
            self.n_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {self.n_gpus}")
        else:
            self.device = torch.device('cpu')
            self.n_gpus = 0

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
        )

        if self.n_gpus > 1:
            print(f"Wrapping model with DataParallel on {self.n_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        else:
            print(f"Using single device: {self.device}")
        
        # Move to device AFTER wrapping
        self.model = self.model.to(device=self.device, dtype=self.dtype)

        
        self.optimizer = torch.optim.AdamW(make_parameter_groups(self.model), lr=3.2e-2, weight_decay=3e-4) #2e-3
        
        # Training parameters
        self.n_epochs = 45
        self.patience = 16
        self.batch_size = 4096 ########----------------------return to 256---------------------
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
    
    def get_model_attr(self, attr_name, default=None):
        """
        Universal method to get attributes from model.
        Works with both DataParallel wrapped and regular models.
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        return getattr(model, attr_name, default)
        
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
        for epoch in range(self.n_epochs):
            # Use the universal accessor
            share_training_batches = self.get_model_attr('share_training_batches')
            k_value = self.get_model_attr('k')
            
            batches = (
                torch.randperm(train_size, device=self.device).split(self.batch_size)
                if share_training_batches
                else [
                    x.transpose(0, 1).flatten()
                    for x in torch.rand((k_value, train_size), device=self.device)
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
    def predict(self, X_test, y_test=None):
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
                    # assert X_test_cat[:, i].max() < cardinality
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
        
        self.ctr = predictions
        return predictions

    def _load_model_(self, prefix=None):
        """
        Load model weights
        args:
            - prefix: str -is used for theprefix of the path
        """
        TABM_PATH = 'tabm.pth'

        if prefix is not None:
            TABM_PATH = os.path.join(prefix, TABM_PATH)


        if os.path.exists(TABM_PATH):
            self.model.load_state_dict(torch.load(TABM_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            print(f"Loaded TabM model from weights/tabm.pth")
        else:
            self.fit()
            print("No saved TabM model found")
        self.loaded = True

    def _log_model_(self):
        super()._log_model_()
        torch.save(self.model.state_dict(), 'weights/tabm.pth')
        print(f"Saved TabM model to weights/tabm.pth")
