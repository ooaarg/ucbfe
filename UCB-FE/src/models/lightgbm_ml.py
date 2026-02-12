import sys
import os
import lightgbm as lgb

# Add the src directory to path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.dirname(current_dir)
# if src_dir not in sys.path:
#     sys.path.insert(0, src_dir)

# # Import ML from the clean base
# try:
#     from models.ml_base import ML  # Import from the clean base
# except ImportError:
#     # Fallback
#     from ml_base import ML

from ml import ML, log_model_operation, log_model_weights


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
        self.model = lgb.LGBMClassifier(random_state=42, categorical_feature=cat_indices)

    @log_model_weights
    @log_model_operation
    def fit(self):
        super().fit()
        self.model.fit(self.X_train, self.y_train)
        
    @log_model_operation
    def predict(self, X_test, y_test=None):
        # Import lightgbm here, not at top level
        # try:
        #     import lightgbm as lgb
        #     lgb = lgb
        # except ImportError:
        #     raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        super().predict(X_test)

        #prepare test data
        self.X_test = X_test.copy()  # Create a copy to avoid SettingWithCopyWarning
        for name in self.categorical_features:
            self.X_test.loc[:, name] = self.X_test[name].astype('category')
        #predict    
        if isinstance(self.model, lgb.Booster):
            # If model is a Booster (loaded from file), use predict directly
            self.ctr = self.model.predict(self.X_test, predict_disable_shape_check=True)
        else:
            # If model is LGBMClassifier, use predict_proba
            self.ctr = self.model.predict_proba(self.X_test, predict_disable_shape_check=True)[:, 1]
        return self.ctr

    def _load_model_(self, prefix=None):
        """
        Load model weights
        args:
            - prefix: str -is used for theprefix of the path
        """
        LGBM_PATH = 'lightgbm.txt'

        if prefix is not None:
            LGBM_PATH = os.path.join(prefix, LGBM_PATH)


        # elif isinstance(self.model, lgb.LGBMClassifier):
        if os.path.exists(LGBM_PATH):
            # Load the booster model
            with open(LGBM_PATH, 'r') as f:
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
            self.fit()
        self.loaded = True

    def _log_model_(self):
        super()._log_model_()
        self.model.booster_.save_model('weights/lightgbm.txt')
        print(f"Saved LightGBM model to weights/lightgbm.txt")

