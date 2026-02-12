import sys
import importlib

__all__ = [
    'ML', 'FinalNet', 'FuxiUniversalAdapter', 
    'TabMModel', 'Lightgbm', 'Xgboost', 'Catboost', 'Tab_net', 'permutation'
]

# Define lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import of modules."""
    if name == 'ML':
        from .ml_base import ML
        return ML
    elif name == 'Lightgbm':
        from .lightgbm import Lightgbm
        return Lightgbm
    elif name == 'Xgboost':
        from .xgboost import Xgboost
        return Xgboost
    elif name == 'Catboost':
        from .catboost import Catboost
        return Catboost
    elif name == 'TabMModel':
        from .tabm import TabMModel
        return TabMModel
    elif name == 'Tab_net':
        from .tab_net import Tab_net
        return Tab_net
    elif name == 'FinalNet':
        from .FinalNet import FinalNet
        return FinalNet
    elif name == 'FuxiUniversalAdapter':
        from .fuxiadapter import FuxiUniversalAdapter
        return FuxiUniversalAdapter
    elif name == 'permutation':
        from .utils import permutation
        return permutation
    else:
        raise AttributeError(f"module 'models' has no attribute '{name}'")

# Also provide explicit imports for backward compatibility
try:
    from .ml_base import ML
except ImportError:
    ML = None

try:
    from .FinalNet import FinalNet
except ImportError:
    FinalNet = None

try:
    from .fuxiadapter import FuxiUniversalAdapter
except ImportError:
    FuxiUniversalAdapter = None
