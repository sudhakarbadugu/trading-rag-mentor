import os
import streamlit as st

def get_secret(key: str, default: str = "") -> str:
    """
    Helper function to safely fetch secrets, checking Streamlit secrets first,
    then falling back to environment variables.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    return os.environ.get(key, default)
