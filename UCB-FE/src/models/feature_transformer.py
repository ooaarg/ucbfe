import numpy as np
import pandas as pd


class FeatureTransformer:
    """Centralized feature transformation logic for UCB-FE experiments"""
    
    @staticmethod
    def apply_ucb_transformation(df, delta, T):
        """
        Apply UCB transformation to CTR feature
        
        Args:
            df: DataFrame with item data
            delta: UCB parameter
            T: cold-start period
            
        Returns:
            Modified DataFrame with UCB-transformed CTR features
        """
        df_copy = df.copy()
        
        # Clip item impressions to avoid division by zero
        N = np.clip(df_copy["item_imps"], a_min=0.5, a_max=None)
        
        # Create cold start indicator
        df_copy[f"cold_{T}"] = 0 + (df_copy['item_imps'] <= T)
        
        # Apply UCB transformation
        if T <= 1:
            df_copy[f'ucb_ctr_T_{T}'] = df_copy[f'cold_{T}'] + (1 - df_copy[f'cold_{T}']) * df_copy['ctr']
        else:
            df_copy[f'ucb_ctr_T_{T}'] = np.clip(
                df_copy[f'cold_{T}'] * (df_copy['ctr'] + np.sqrt(delta * np.log(T) / N)) + 
                (1 - df_copy[f'cold_{T}']) * df_copy['ctr'], 
                a_min=0, a_max=1
            )
        
        return df_copy
    
    @staticmethod
    def prepare_test_data(X_test, y_test, ctr_values=None):
        """
        Prepare test data for prediction
        
        Args:
            X_test: Test features DataFrame
            y_test: Test labels
            ctr_values: Optional CTR values to use instead of existing ones
            
        Returns:
            Prepared DataFrame for prediction
        """
        X_test_copy = X_test.copy()
        
        # If ctr_values provided, replace the CTR column
        if ctr_values is not None:
            X_test_copy['ctr'] = ctr_values
            
        return X_test_copy, y_test
    
    @staticmethod
    def get_cold_periods():
        """Return standard cold periods for experiments"""
        return [0, 10, 100, 200, 500]
    
    @staticmethod
    def get_default_delta():
        """Return default delta parameter for UCB"""
        return 1.5
