# -*- coding: utf-8 -*-

import os
from pathlib import Path

class Config:
    def __init__(self):
        # Jaeger配置
        self.JAEGER_HOST = os.getenv("JAEGER_HOST", "192.168.1.102")
        self.JAEGER_PORT = os.getenv("JAEGER_PORT", "16686")
        
        # Prometheus配置
        self.PROMETHEUS_HOST = os.getenv("PROMETHEUS_HOST", "192.168.1.102")
        self.PROMETHEUS_PORT = os.getenv("PROMETHEUS_PORT", "9090")
        
        # 请求超时配置
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # 采集配置
        self.DEFAULT_COLLECTION_INTERVAL = int(os.getenv("COLLECTION_INTERVAL", "60"))
        
        # 输出目录配置
        self.OUTPUT_DIR = os.getenv("OUTPUT_DIR", "trace")
        
    def ensure_output_dir(self) -> str:
        """确保输出目录存在"""
        output_path = Path(self.OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        return str(output_path)
    
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