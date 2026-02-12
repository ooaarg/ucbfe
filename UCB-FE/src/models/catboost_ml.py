import numpy as np
import pandas as pd
import os
from catboost import CatBoostClassifier

from .ml import ML
from .ml import log_model_operation, log_model_weights



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
        # self.model = CatBoostClassifier(
        #     cat_features=categorical_features,
        #     border_count = 128,
        #     learning_rate = 0.2,
        #     loss_function =  "Logloss",
        #     min_data_in_leaf = 32,
        #     model_size_reg = 1,
        #     num_trees =  200,
        #     random_seed = 0,
        #     task_type="GPU",
        #     devices='0'
        # )
        self.model = CatBoostClassifier(
        cat_features=categorical_features,
        # border_count = 128,
        # learning_rate = 0.2,
        loss_function =  "Logloss",
        # min_data_in_leaf = 32,
        # model_size_reg = 1,
        # num_trees =  200,
        random_seed = 0,
        task_type="GPU",
        devices=['0']
        )

    @log_model_weights
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
            verbose=True,
            plot=False,
            early_stopping_rounds = 50
        )

    @log_model_operation
    def predict(self, test_data, y_test=None):
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
        print(test_data.columns)
        self.ctr = self.model.predict_proba(test_data)[:, 1]
        return self.ctr
    
    def _load_model_(self, prefix=None):
        """
        Load model weights
        args:
            - prefix: str -is used for theprefix of the path
        """
        CATBOOST_PATH = 'catboost.cbm'

        if prefix is not None:
            CATBOOST_PATH = os.path.join(prefix, CATBOOST_PATH)


        # if isinstance(self.model, CatBoostClassifier):
        if os.path.exists(CATBOOST_PATH):
            self.model.load_model(CATBOOST_PATH)
            print(f"Loaded CatBoost model from weights/catboost.cbm")
        else:
            self.fit()
            print("No saved CatBoost model found")

        self.loaded = True

    def _log_model_(self):
        super()._log_model_()
        self.model.save_model('weights/catboost.cbm')
        print(f"Saved CatBoost model to weights/catboost.cbm")


