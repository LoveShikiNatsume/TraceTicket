from normal_request_manager import main_thread
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # 启动5个线程进行压测
    main_thread()