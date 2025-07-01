# -*- coding: utf-8 -*-

import os
from pathlib import Path

class Config:
    def __init__(self):
        # Jaeger配置
        self.JAEGER_HOST = os.getenv("JAEGER_HOST", "localhost")
        self.JAEGER_PORT = os.getenv("JAEGER_PORT", "16686")
        
        # Prometheus配置（新增）
        self.PROMETHEUS_HOST = os.getenv("PROMETHEUS_HOST", "localhost")
        self.PROMETHEUS_PORT = os.getenv("PROMETHEUS_PORT", "9090")
        
        # 采集配置
        self.DEFAULT_COLLECTION_INTERVAL = 60  # 默认采集间隔（秒）
        self.LOOKBACK_PERIOD = "5m"           # 回看时间窗口
        
        self.SCRIPT_DIR = Path(__file__).resolve().parent
        self.PROJECT_ROOT = self.SCRIPT_DIR.parent
        self.OUTPUT_DIR = str(self.PROJECT_ROOT / "trace")
        
    def ensure_output_dir(self) -> str:
        """确保输出目录存在并返回路径"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        return self.OUTPUT_DIR
    
    def get_project_info(self) -> dict:
        """获取项目路径信息（用于调试）"""
        return {
            "script_dir": str(self.SCRIPT_DIR),
            "project_root": str(self.PROJECT_ROOT),
            "output_dir": self.OUTPUT_DIR,
            "current_working_dir": os.getcwd()
        }

# 单例配置实例
config = Config()

# 调试信息
if __name__ == "__main__":
    info = config.get_project_info()
    print("=== 路径配置信息 ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print(f"输出目录: {config.OUTPUT_DIR}")