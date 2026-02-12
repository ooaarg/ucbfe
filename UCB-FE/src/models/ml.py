# ml_base.py - Clean base class without circular imports
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import os
import warnings

warnings.simplefilter('ignore')

def log_model_operation(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
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
        
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.values
            self.feature_names = X_train.columns.tolist()
        else:
            self.X_train = X_train
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            
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
        """fit a model"""
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """predict_proba"""
        pass
    
    def _log_model_(self):
        """Save model weights"""
        if not os.path.exists('weights'):
            os.makedirs('weights')

        pass
    
    def _load_model_(self, prefix=None):
        """Load model weights"""
        pass
