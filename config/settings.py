"""
Configuration settings for StartupTN Evaluation and Monitoring System.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "StartupTN AI Proposal Evaluation System"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Model configuration
    BASE_MODEL_NAME: str = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen3-0.6B")
    MODEL_CACHE_DIR: Path = BASE_DIR / "models"
    
    # Data paths
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    
    # Financial ML Config
    DEFAULT_RISK_THRESHOLD_HIGH: float = 0.70
    DEFAULT_RISK_THRESHOLD_MEDIUM: float = 0.40
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/startuptn_db"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
