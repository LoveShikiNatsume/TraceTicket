#!/usr/bin/env python3
"""
服务启动脚本
"""

import os
import sys
from pathlib import Path

# 添加项目路径到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from app.config import config

def main():
    """主函数"""
    print("🚀 启动TraceVAE在线异常检测服务...")
    
    # 检查配置
    if not config.validate_paths():
        print("❌ 配置验证失败，请检查模型文件和配置目录路径")
        sys.exit(1)
    
    # 启动服务
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,  # 使用单进程，因为模型需要共享
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()