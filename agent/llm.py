import backoff
import os
import sys
from typing import Tuple
import requests
import litellm
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Try to import global config, fallback to local definitions if not available
try:
    sys.path.insert(0, '/hada')
    from config import MODEL_NAME, set_global_seed
    # Set global seed
    set_global_seed()
    GLOBAL_MODEL_NAME = MODEL_NAME
except ImportError:
    GLOBAL_MODEL_NAME = None


MAX_TOKENS = 20000

# AIHubMix API configuration
AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
AIHUBMIX_BASE_URL = "https://aihubmix.com/v1"
AIHUBMIX_GEMINI_BASE_URL = "https://aihubmix.com/gemini"
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Model definitions for AIHubMix - using simplified model names
# AIHubMix models (no provider prefix needed)
CLAUDE_MODEL = "claude-3-5-sonnet"
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini-2.5-flash"

# Aliyun models (need "openai/" prefix for LiteLLM routing)
QWEN_MODEL = "openai/qwen-flash"
QWEN_PLUS_MODEL = "openai/qwen-plus"
DOUBAO_MODEL = "openai/doubao-seed-2-0-mini"
DEEPSEEK_MODEL = "openai/deepseek-v3"

# Kimi models (use OpenAI SDK directly, compatible with OpenAI protocol)
KIMI_K3 = "kimi-k3"
KIMI_K2_7_CODE = "kimi-k2.7-code"
KIMI_K2_6 = "kimi-k2.6"
KIMI_K2_5 = "kimi-k2.5"
KIMI_MOONSHOT_V1_8K = "moonshot-v1-8k"
KIMI_MOONSHOT_V1_32K = "moonshot-v1-32k"
KIMI_MOONSHOT_V1_128K = "moonshot-v1-128k"

# DeepSeek models (use OpenAI SDK directly, compatible with OpenAI protocol)
DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"
DEEPSEEK_V4_PRO = "deepseek/deepseek-v4-pro"

# Default model for cost optimization (use cheaper model by default)
# Use global config if available, otherwise fallback to AIHubMix Claude
DEFAULT_MODEL = GLOBAL_MODEL_NAME if GLOBAL_MODEL_NAME else CLAUDE_MODEL

litellm.drop_params=True

# Configure LiteLLM for AIHubMix with retry and timeout settings
litellm.set_verbose=False  # Disable verbose logging

# Configure LiteLLM for AIHubMix
def configure_litellm_for_aihubmix():
    """Configure LiteLLM to use AIHubMix with correct API base URLs"""
    # Set default API key for all models
    litellm.api_key = AIHUBMIX_API_KEY
    
    # Configure model-specific API base URLs
    # OpenAI and Claude models use standard endpoint
    litellm.api_base = AIHUBMIX_BASE_URL
    
    # Configure timeout and retry settings
    litellm.timeout = 180.0  # 180 seconds timeout (increased for complex tasks)
    litellm.max_retries = 3  # Retry up to 3 times
    litellm.fallback = ["gpt-4o-mini"]  # Fallback to cheaper model
    
    # Gemini models need special configuration
    # We'll handle this in the completion call

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
) -> Tuple[str, list, dict]:
    if msg_history is None:
        msg_history = []

    # Convert text to content, compatible with LITELLM API
    msg_history = [
        {**msg, "content": msg["text"]} if "text" in msg else msg
        for msg in msg_history
    ]

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    

    # Build kwargs - handle model-specific requirements
    completion_kwargs = {
        "model": model,
        "messages": new_msg_history,
        "api_key": AIHUBMIX_API_KEY,
    }

    # Set correct API base URL based on model type
    # AIHubMix models (no prefix): claude, gpt-4o, gemini
    if model.startswith("gemini"):
        completion_kwargs["api_base"] = AIHUBMIX_GEMINI_BASE_URL
    elif model.startswith("openai/"):
        # Aliyun models
        completion_kwargs["api_base"] = ALIYUN_BASE_URL
        completion_kwargs["api_key"] = ALIYUN_API_KEY
    elif model.startswith("kimi-") or model.startswith("moonshot-"):
        # Kimi models - use OpenAI SDK directly (bypass LiteLLM)
        client = OpenAI(
            api_key=KIMI_API_KEY,
            base_url=KIMI_BASE_URL
        )
        # Kimi K3 uses reasoning_effort (top-level parameter)
        # K2.6 uses temperature=1 (fixed, do not pass explicitly)
        # Other models: temperature is configurable
        kimi_kwargs = {
            "model": model,
            "messages": new_msg_history,
            "max_tokens": max_tokens,
        }
        if model == "kimi-k3":
            kimi_kwargs["reasoning_effort"] = "high"
        elif model == "kimi-k2.6":
            # K2.6 requires temperature=1, but do not pass explicitly
            pass
        else:
            kimi_kwargs["temperature"] = temperature
        
        response = client.chat.completions.create(**kimi_kwargs)
        response_text = response.choices[0].message.content or ""
        if not response_text:
            # Debug: print response structure for empty responses
            import logging
            logging.warning(f"Kimi {model} returned empty response. Finish reason: {response.choices[0].finish_reason}")
        
        new_msg_history.append({"role": "assistant", "content": response_text})
        
        # Convert content to text, compatible with MetaGen API
        new_msg_history = [
            {**msg, "text": msg["content"]} if "content" in msg else msg
            for msg in new_msg_history
        ]
        
        return response_text, new_msg_history, {}
    elif model.startswith("deepseek-flash") or model.startswith("deepseek-v4") or model.startswith("deepseek/deepseek-"):
        # DeepSeek models - use OpenAI SDK directly (bypass LiteLLM)
        deepseek_model_name = model.replace("deepseek/", "")
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        response = client.chat.completions.create(
            model=deepseek_model_name,
            messages=new_msg_history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response_text = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": response_text})
        
        # Convert content to text, compatible with MetaGen API
        new_msg_history = [
            {**msg, "text": msg["content"]} if "content" in msg else msg
            for msg in new_msg_history
        ]
        
        return response_text, new_msg_history, {}
    else:
        # AIHubMix models (claude, gpt-4o, etc.)
        completion_kwargs["api_base"] = AIHUBMIX_BASE_URL

    # GPT-5 and GPT-5-mini only support default temperature (1), skip it
    # GPT-5.2 supports temperature
    if model in ["openai/gpt-5", "openai/gpt-5-mini"]:
        pass  # Don't set temperature
    else:
        completion_kwargs["temperature"] = temperature

    # GPT-5 models require max_completion_tokens instead of max_tokens
    if "gpt-5" in model:
        completion_kwargs["max_completion_tokens"] = max_tokens
    else:
        # Claude Haiku has a 4096 token limit
        if "claude-3-haiku" in model:
            completion_kwargs["max_tokens"] = min(max_tokens, 4096)
        else:
            completion_kwargs["max_tokens"] = max_tokens

    response = litellm.completion(**completion_kwargs)
    response_text = response['choices'][0]['message']['content']  # pyright: ignore
    new_msg_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

    # Convert content to text, compatible with MetaGen API
    new_msg_history = [
        {**msg, "text": msg["content"]} if "content" in msg else msg
        for msg in new_msg_history
    ]

    return response_text, new_msg_history, {}


if __name__ == "__main__":
    msg = 'Hello there!'
    models = [
        # ("CLAUDE_MODEL", CLAUDE_MODEL),
        # ("CLAUDE_HAIKU_MODEL", CLAUDE_HAIKU_MODEL),
        # ("OPENAI_MODEL", OPENAI_MODEL),
        # ("OPENAI_MINI_MODEL", OPENAI_MINI_MODEL),
        # ("GEMINI_MODEL", GEMINI_MODEL),
        # ("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL),
        ("QWEN_MODEL",QWEN_MODEL)
    ]
    for name, model in models:
        print(f"\n{'='*50}")
        print(f"Testing {name}: {model}")
        print('='*50)
        try:
            output_msg, msg_history, info = get_response_from_llm(msg, model=model)
            print(f"OK: {output_msg[:100]}...")
        except Exception as e:
            print(f"FAIL: {str(e)[:200]}")