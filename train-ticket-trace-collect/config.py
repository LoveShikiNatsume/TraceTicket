# -*- coding: utf-8 -*-

import os
from pathlib import Path

class Config:
    # Jaeger 连接配置
    JAEGER_HOST = "192.168.1.102"
    JAEGER_PORT = 31686
    REQUEST_TIMEOUT = 30
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    OUTPUT_DIR = str(PROJECT_ROOT / "trace_output")
    
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
    
    print(f"\n输出目录是否存在: {os.path.exists(config.OUTPUT_DIR)}")
    print(f"输出目录: {config.OUTPUT_DIR}")