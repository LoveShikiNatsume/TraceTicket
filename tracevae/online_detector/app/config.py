#!/usr/bin/env python3
"""
在线异常检测服务配置管理
"""

import os
from pathlib import Path
from typing import Optional

class Config:
    """服务配置类"""
    
    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # 模型配置
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../results/train/models/final.pt")
    CONFIG_DIR: str = os.getenv("CONFIG_DIR", "../TT_Dataset/TT_Dataset/convert_data_time_corrected")
    
    # 设备配置
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    # 检测阈值
    ANOMALY_THRESHOLD: float = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
    
    # 性能配置
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "1000"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_absolute_path(cls, relative_path: str) -> str:
        """获取绝对路径"""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / relative_path)
    
    @classmethod
    def validate_paths(cls) -> bool:
        """验证路径是否存在"""
        model_path = cls.get_absolute_path(cls.MODEL_PATH)
        config_dir = cls.get_absolute_path(cls.CONFIG_DIR)
        
        if not Path(model_path).exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
            
        if not Path(config_dir).exists():
            print(f"❌ 配置目录不存在: {config_dir}")
            return False
            
        return True

config = Config()