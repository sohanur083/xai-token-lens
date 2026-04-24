"""xai-token-lens — unified token-level explanations for transformer models."""
from .core import TokenLens, QALens, Visualization, explain
from .methods import lime_explain, shap_explain, attention_explain

__version__ = "0.1.0"
__all__ = [
    "TokenLens", "QALens", "Visualization", "explain",
    "lime_explain", "shap_explain", "attention_explain",
]
