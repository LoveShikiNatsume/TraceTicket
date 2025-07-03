from normal_request_manager import main_thread
import logging
import signal
import sys
import os

# 导入配置模块
try:
    from config import get_pressure_config, DEFAULT_PRESSURE_LEVEL
except ImportError:
    # 如果config.py不可用时的后备配置
    print("警告: 未找到config.py，使用默认配置")
    def get_pressure_config(level=None):
        return {"thread_count": 5, "request_interval": 1.0, "name": "默认配置"}
    DEFAULT_PRESSURE_LEVEL = "medium"

# 配置日志
logging.basicConfig(level=logging.INFO)

# 全局标志控制运行状态
running = True

def signal_handler(signum, frame):
    """处理终止信号"""
    global running
    print(f"\n收到终止信号 ({signum})，正在停止压测...")
    running = False

if __name__ == '__main__':
    # 注册信号处理器
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 加载压测配置
    pressure_level = os.getenv('PRESSURE_TEST_LEVEL', DEFAULT_PRESSURE_LEVEL)
    
    try:
        config = get_pressure_config(pressure_level)
        thread_count = config["thread_count"]
        request_interval = config["request_interval"]
        config_name = config["name"]
        
        # 检查是否在安静模式下运行（由主控制器调用）
        quiet_mode = 'PRESSURE_TEST_QUIET' in os.environ
        
        if not quiet_mode:
            print("独立运行模式 - 详细日志输出已启用")
            print(f"压测配置: {config_name}")
            print(f"线程数量: {thread_count}, 请求间隔: {request_interval}秒")
        
    except Exception as e:
        print(f"配置加载失败: {e}")
        print("使用后备配置")
        thread_count = 5
        request_interval = 1.0
    
    try:
        # 使用配置的参数启动压测
        main_thread(thread_count=thread_count, request_interval=request_interval, running_flag=lambda: running)
    except KeyboardInterrupt:
        print("用户中断压测")
    finally:
        print("压测已终止")