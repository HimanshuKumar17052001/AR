import os
from typing import Optional

def get_config(key: str, default: str = None) -> str:
    """
    Get configuration from Streamlit secrets or environment variables.
    This function works both in Streamlit and standalone Python scripts.
    """
    try:
        # Try to import streamlit and check if we're in a Streamlit context
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets.get(key):
            return st.secrets[key]
    except (ImportError, AttributeError):
        pass
    
    # Fallback to environment variables
    return os.getenv(key, default)

# Configuration constants
QDRANT_URL = get_config("QDRANT_URL")
QDRANT_API_KEY = get_config("QDRANT_API_KEY")
AWS_ACCESS_KEY_ID = get_config("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_config("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = get_config("AWS_REGION_NAME", "us-east-1")
CLAUDE_MODEL_ID = get_config("CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
ANTHROPIC_VERSION = get_config("ANTHROPIC_VERSION", "bedrock-2023-05-31")
OCR_SERVICE_URL = get_config("OCR_SERVICE_URL", "http://52.7.81.94:8000/ocr_image")
SPARSE_EMBEDDING_URL = get_config("SPARSE_EMBEDDING_URL", "http://52.7.81.94:8010/embed")
LOG_LEVEL = get_config("LOG_LEVEL", "INFO") 