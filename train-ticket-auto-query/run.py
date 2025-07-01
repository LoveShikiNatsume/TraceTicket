from normal_request_manager import main_thread
import logging
import signal
import sys
import time

# 配置日志
logging.basicConfig(level=logging.INFO)

# 全局标志控制运行状态
running = True

def signal_handler(signum, frame):
    """处理停止信号"""
    global running
    print(f"\n收到停止信号 ({signum})，正在停止压测...")
    running = False

def run_load_test():
    """运行压测，支持优雅停止"""
    global running
    
    # 注册信号处理器
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("启动Train Ticket压测...")
        # 修改main_thread函数，传入running标志
        main_thread(running_flag=lambda: running)
        
    except KeyboardInterrupt:
        print("用户中断压测")
    except Exception as e:
        print(f"压测异常: {e}")
    finally:
        print("压测已停止")

if __name__ == '__main__':
    run_load_test()