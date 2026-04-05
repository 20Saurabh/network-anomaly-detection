"""Model registry and imports."""
from typing import Dict, Type

_MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> Type:
    """Retrieve a registered model class by name."""
    if name not in _MODEL_REGISTRY:
        # Lazy import all models to populate registry
        _import_all_models()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not registered. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def list_models():
    """List all available model names."""
    _import_all_models()
    return list(_MODEL_REGISTRY.keys())


def _import_all_models():
    """Lazy import all model modules."""
    try:
        from . import vanilla_autoencoder
        from . import variational_autoencoder
        from . import cnn1d
        from . import bilstm_attention
        from . import cnn_lstm_hybrid
        from . import ft_transformer
        from . import contrastive_ssl
        from . import isolation_forest_wrapper
        from . import ensemble_stacking
    except ImportError:
        pass
    # GNN models require PyTorch Geometric — optional
    try:
        from . import e_graphsage
        from . import gnn_transformer_hybrid
    except ImportError:
        pass
