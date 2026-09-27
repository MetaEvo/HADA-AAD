"""Global configuration module for HADA.

This module contains global settings that can be imported by other modules.
Change the values here to affect all code that uses these settings.
"""

import numpy as np
import random

# =============================================================================
# GLOBAL SETTINGS - Modify these values to change behavior across all modules
# =============================================================================

# Random seed for reproducibility
SEED = 42

# Model name for LLM calls (used by all agents)
# AIHubMix models (recommended):
#   - "claude-3-5-sonnet"
#   - "gpt-4o"
#   - "gpt-4o-mini"
#   - "gemini-2.5-pro"
#   - "gemini-2.5-flash"
# Aliyun models:
#   - "openai/qwen-turbo"
#   - "openai/qwen-flash"
#   - "openai/qwen-plus"
#   - "openai/deepseek-v3"
# Kimi models:
#   - "kimi-k3"
#   - "kimi-k2.7-code"
#   - "kimi-k2.6"
#   - "kimi-k2.5"
#   - "moonshot-v1-8k"
#   - "moonshot-v1-32k"
#   - "moonshot-v1-128k"
# DeepSeek models:
#   - "deepseek-v4-flash"
#   - "deepseek-v4-pro"
MODEL_NAME = "deepseek-v4-pro"

# =============================================================================
# Initialization function - Call this at the start of your script
# =============================================================================

def set_global_seed(seed=None):
    """Set the random seed for all random number generators.
    
    Args:
        seed: Random seed value. If None, uses the global SEED from config.
    """
    if seed is None:
        seed = SEED
    
    # NumPy random seed
    np.random.seed(seed)
    
    # Python random seed
    random.seed(seed)
    
    # Try to set torch seed if available

    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed


def get_model_name():
    """Get the global model name.
    
    Returns:
        str: The model name to use for LLM calls.
    """
    return MODEL_NAME


# Auto-initialize seed when module is imported
set_global_seed()