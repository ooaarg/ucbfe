import numpy as np
import xgboost as xgb
import os
from xgboost import XGBClassifier

from .ml import ML
from .ml import log_model_operation, log_model_weights


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
    def predict(self, X_test, y_test=None):
        super().predict(X_test)
        
        #prepare test data
        self.X_test = X_test.copy()  # Create a copy to avoid SettingWithCopyWarning
        for name in self.categorical_features:
            self.X_test.loc[:, name] = self.X_test[name].astype('category')

        expected_features = self.model.get_booster().feature_names
        self.X_test = self.X_test[expected_features]


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

    def _load_model_(self, prefix=None):
        """
        Load model weights
        args:
            - prefix: str -is used for theprefix of the path
        """
        XGBOOST_PATH = 'xgboost.json'

        if prefix is not None:
            XGBOOST_PATH = os.path.join(prefix, XGBOOST_PATH)

        if os.path.exists(XGBOOST_PATH):
            # Create a new booster from the saved model
            booster = xgb.Booster()
            booster.load_model(XGBOOST_PATH)
            # Update the XGBClassifier with the loaded booster
            self.model._Booster = booster
            print(f"Loaded XGBoost model from weights/xgboost.json")
        else:
            self.fit()
            print("No saved XGBoost model found")
        self.loaded = True

    def _log_model_(self):
        super()._log_model_()
        self.model.save_model('weights/xgboost.json')
        print(f"Saved XGBoost model to weights/xgboost.json")
