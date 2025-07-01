from query_and_preserve import query_and_preserve
from query_order_and_pay import query_order_and_pay
from query_and_collect_ticket import query_and_collect_ticket
from query_and_enter_station import query_and_enter_station
from query_and_cancel import query_one_and_cancel

from atomic_queries import _login, _query_orders, _query_high_speed_ticket

from utils import random_boolean
import time

from threading import Thread

def main(duration_minutes=60):  # 运行指定分钟数，默认60分钟
    pairs = [('Shang Hai', 'Su Zhou'), ('Shang Hai', 'Nan Jing')]
    headers = {
        "Cookie": "JSESSIONID=21A0370861087E0831E5D25D56BC9ABB; YsbCaptcha=BE12EE0295F548569DCC1D5B07FDBA55",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZHNlX21pY3Jvc2VydmljZSIsInJvbGVzIjpbIlJPTEVfVVNFUiJdLCJpZCI6IjRkMmE0NmM3LTcxY2ItNGNmMS1iNWJiLWI2ODQwNmQ5ZGE2ZiIsImlhdCI6MTYyNzI2MzE4NywiZXhwIjoxNjI3MjY2Nzg3fQ.xOXWi3QpTYL1OZqXaAHmpifyPc_lMX9smtOPTUveO9M",
        "Content-Type": "application/json"
    }

    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    i = 0
    
    while time.time() < end_time:  # 运行指定时间
        try:
            now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"now_time:{now_time}")

            if i % 20 == 0:
                uid, token = _login()
                if uid is not None and token is not None:
                    headers['Authorization'] = "Bearer " + token

            print(f"idx:{i}")
            query_and_preserve(headers)

            if random_boolean() and random_boolean():
                query_one_and_cancel(headers)
            else:
                query_order_and_pay(headers, pairs)
                query_and_collect_ticket(headers)
                query_and_enter_station(headers)
            
            i += 1
            time.sleep(1)
        except Exception as e:
            print(f"main loop error at index {i}: {e}")
            i += 1


def main_thread(duration_minutes=60, running_flag=None):
    """主压测线程，支持外部停止控制"""
    import threading
    import time
    
    if running_flag is None:
        running_flag = lambda: True
    
    def worker_thread(thread_id):
        """工作线程"""
        print(f"压测线程 {thread_id} 启动")
        
        while running_flag():
            try:
                # 这里放置实际的压测逻辑
                # 例如：发送HTTP请求到Train Ticket系统
                
                # 模拟压测请求
                time.sleep(1)  # 1秒一次请求
                
                # 检查是否需要停止
                if not running_flag():
                    break
                    
            except Exception as e:
                print(f"压测线程 {thread_id} 异常: {e}")
                if not running_flag():
                    break
                time.sleep(5)  # 异常后等待5秒
        
        print(f"压测线程 {thread_id} 已停止")
    
    # 启动5个压测线程
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_thread, args=(i+1,))
        thread.daemon = True  # 设置为守护线程
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成或收到停止信号
    try:
        while running_flag() and any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("收到中断信号，停止压测")
    
    print("所有压测线程已停止")


def query_order():
    headers = {
        "Cookie": "JSESSIONID=21A0370861087E0831E5D25D56BC9ABB; YsbCaptcha=BE12EE0295F548569DCC1D5B07FDBA55",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZHNlX21pY3Jvc2VydmljZSIsInJvbGVzIjpbIlJPTEVfVVNFUiJdLCJpZCI6IjRkMmE0NmM3LTcxY2ItNGNmMS1iNWJiLWI2ODQwNmQ5ZGE2ZiIsImlhdCI6MTYyNzI2MzE4NywiZXhwIjoxNjI3MjY2Nzg3fQ.xOXWi3QpTYL1OZqXaAHmpifyPc_lMX9smtOPTUveO9M",
        "Content-Type": "application/json"
    }
    uid, token = _login()
    if uid is not None and token is not None:
        headers['Authorization'] = "Bearer " + token

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"start:{start_time}")

    for i in range(50):
        pairs = _query_orders(headers=headers, types=tuple([0, 1]), query_other=False)
        print(pairs)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"start:{start_time} end:{end_time}")


def query_tickets():
    headers = {
        "Cookie": "JSESSIONID=21A0370861087E0831E5D25D56BC9ABB; YsbCaptcha=BE12EE0295F548569DCC1D5B07FDBA55",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZHNlX21pY3Jvc2VydmljZSIsInJvbGVzIjpbIlJPTEVfVVNFUiJdLCJpZCI6IjRkMmE0NmM3LTcxY2ItNGNmMS1iNWJiLWI2ODQwNmQ5ZGE2ZiIsImlhdCI6MTYyNzI2MzE4NywiZXhwIjoxNjI3MjY2Nzg3fQ.xOXWi3QpTYL1OZqXaAHmpifyPc_lMX9smtOPTUveO9M",
        "Content-Type": "application/json"
    }
    uid, token = _login()
    if uid is not None and token is not None:
        headers['Authorization'] = "Bearer " + token


    date = time.strftime("%Y-%m-%d", time.localtime())

    start = "Shang Hai"
    end = "Su Zhou"
    high_speed_place_pair = (start, end)

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"start:{start_time}")

    for i in range(50):
        trip_ids = _query_high_speed_ticket(place_pair=high_speed_place_pair, headers=headers, time=date)
        print(trip_ids)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"start:{start_time} end:{end_time}")


if __name__ == '__main__':
    main_thread()