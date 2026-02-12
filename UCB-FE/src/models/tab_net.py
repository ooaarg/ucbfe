import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim
from pytorch_tabnet.tab_model import TabNetClassifier
import os

from .ml import ML
from .ml import log_model_operation, log_model_weights


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
        'verbose': 2,
        'device_name': 'cuda:3'
    }
    
    @log_model_operation
    def __init__(self, X_train, y_train, categorical_features):
        super().__init__(X_train, y_train, categorical_features, "TabNet")
        
        #prepare cat_features
        self.categorical_features = categorical_features
        self.categorical_idx = [X_train.columns.get_loc(column) for column in categorical_features]
        
        # Calculate cardinalities for categorical features
        self.cat_dims = [X_train[col].max() for col in categorical_features]
        
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
        # super().fit()
        # Process categorical features
        print("Processing cat features", self.model.device)
        X_train_processed = self._process_categorical_features(self.X_train)
        
        print("Starting fit", self.model.device, self.y_train.shape)
        self.model.fit(
            X_train=X_train_processed.values,
            y_train=self.y_train,
            max_epochs=100,
            eval_metric=['auc'],
            compute_importance=False,
            batch_size=131_072,
            virtual_batch_size=16_384,
            num_workers=3,
            drop_last=False,
            warm_start=False
        )


    @log_model_operation
    def predict(self, X_test, y_test=None):
        super().predict(X_test)
        # print(self.model.network.device)

        try:
            # Convert input to DataFrame if needed
            if not isinstance(X_test, pd.DataFrame):
                if hasattr(self, 'feature_names'):
                    X_test = pd.DataFrame(X_test, columns=self.feature_names)
                else:
                    X_test = pd.DataFrame(X_test)
            
            # Process categorical features
            X_test_processed = self._process_categorical_features(X_test)
            
            print('Bfore', hasattr(self.model, "network"))
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

    def _load_model_(self, prefix=None):
        """
        Load model weights
        args:
            - prefix: str -is used for theprefix of the path
        """
        TABNET_PATH = 'tabnet.pth.zip'
        if prefix is not None:
            TABNET_PATH = os.path.join(prefix, TABNET_PATH)

        # if os.path.exists(TABNET_PATH):
        try:    
            # Load the model first to get its configuration
            temp_model = TabNetClassifier()
            print("Loaded model temp")
            temp_model.load_model(TABNET_PATH)
            print("Loaded model temp")
            
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
            self.model.load_model(TABNET_PATH)
            print(f"Loaded TabNet model from weights/tabnet.pth.zip")
            print(f"Model expects {self.input_dim} features and {num_embeddings} categorical features")
            print(f"Categorical features used: {self.categorical_features}")
            print(f"Categorical dimensions: {self.cat_dims}")
            
        except Exception as e:
            # self.fit()
            print(f"Error loading model: {(e)}")
            raise
        self.loaded = True
    def _load_model_(self, prefix=None):
        TABNET_PATH = 'tabnet.pth.zip'
        if prefix is not None:
            TABNET_PATH = os.path.join(prefix, TABNET_PATH)

        # self.model = TabNetClassifier()
        self.model.load_model(TABNET_PATH)

        print("Loaded TabNet model")
        print("Input dim:", self.model.input_dim)
        print("Cat idxs:", self.model.cat_idxs)
        print("Cat dims:", self.model.cat_dims)

        self.loaded = True

    def _log_model_(self):
        super()._log_model_()
        if os.path.exists('weights/tabnet.pth.zip'):
            os.remove('weights/tabnet.pth.zip')
        self.model.save_model('weights/tabnet.pth')


    # @log_model_operation
    # def _load_model_(self, prefix=None):
    #     """Load model weights"""
    #     if os.path.exists('weights/tabnet.pth.zip'):
    #         try:
    #             # Load the model first to get its configuration
    #             temp_model = TabNetClassifier()
    #             temp_model.load_model('weights/tabnet.pth.zip')
                
    #             # Get the model's configuration
    #             self.input_dim = temp_model.input_dim
    #             num_embeddings = len(temp_model.network.embedder.embeddings)
                
    #             # Adjust our configuration to match the model
    #             self.categorical_features = self.categorical_features[:num_embeddings]
    #             self.categorical_idx = self.categorical_idx[:num_embeddings]
    #             self.cat_dims = [
    #                 temp_model.network.embedder.embeddings[i].num_embeddings 
    #                 for i in range(num_embeddings)
    #             ]
                
    #             # Store feature names if available
    #             if hasattr(self, 'X_train'):
    #                 self.feature_names = self.X_train.columns.tolist()
                    
    #                 # Determine which features the model expects
    #                 num_features = self.input_dim
    #                 cat_features = self.categorical_features
                    
    #                 # Get numerical features (all features except categorical ones)
    #                 num_features_list = [col for col in self.feature_names if col not in cat_features]
                    
    #                 # Combine numerical and categorical features in the correct order
    #                 self.model_features = num_features_list[:num_features - len(cat_features)] + cat_features
                    
    #                 print(f"Model features: {self.model_features}")
                
    #             # Initialize our model with the correct configuration
    #             self.model = TabNetClassifier(
    #                 cat_idxs=self.categorical_idx,
    #                 cat_dims=self.cat_dims,
    #                 **self.MODEL_PARAMS
    #             )
                
    #             # Load the weights
    #             self.model.load_model('weights/tabnet.pth.zip')
    #             print(f"Loaded TabNet model from weights/tabnet.pth.zip")
    #             print(f"Model expects {self.input_dim} features and {num_embeddings} categorical features")
    #             print(f"Categorical features used: {self.categorical_features}")
    #             print(f"Categorical dimensions: {self.cat_dims}")
                
    #         except Exception as e:
    #             print(f"Error loading model: {str(e)}")
    #             raise
    #     else:
    #         print("No saved TabNet model found")
