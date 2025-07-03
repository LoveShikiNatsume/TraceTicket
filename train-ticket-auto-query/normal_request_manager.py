from query_and_preserve import query_and_preserve
from query_order_and_pay import query_order_and_pay
from query_and_collect_ticket import query_and_collect_ticket
from query_and_enter_station import query_and_enter_station
from query_and_cancel import query_one_and_cancel

from atomic_queries import _login, _query_orders, _query_high_speed_ticket

from utils import random_boolean
import time
import os
import sys

from threading import Thread

def main(request_interval=1.0, running_flag=None):  
    if running_flag is None:
        running_flag = lambda: True
    
    # 检查是否由主控制器调用（通过环境变量）
    quiet_mode = os.getenv('PRESSURE_TEST_QUIET', 'false').lower() == 'true'
    
    pairs = [('Shang Hai', 'Su Zhou'), ('Shang Hai', 'Nan Jing')]
    headers = {
        "Cookie": "JSESSIONID=21A0370861087E0831E5D25D56BC9ABB; YsbCaptcha=BE12EE0295F548569DCC1D5B07FDBA55",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZHNlX21pY3Jvc2VydmljZSIsInJvbGVzIjpbIlJPTEVfVVNFUiJdLCJpZCI6IjRkMmE0NmM3LTcxY2ItNGNmMS1iNWJiLWI2ODQwNmQ5ZGE2ZiIsImlhdCI6MTYyNzI2MzE4NywiZXhwIjoxNjI3MjY2Nzg3fQ.xOXWi3QpTYL1OZqXaAHmpifyPc_lMX9smtOPTUveO9M",
        "Content-Type": "application/json"
    }

    i = 0
    success_count = 0
    
    while running_flag():
        try:
            if not quiet_mode:
                now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"当前时间: {now_time}")

            if i % 20 == 0:
                uid, token = _login()
                if uid is not None and token is not None:
                    headers['Authorization'] = "Bearer " + token

            if not quiet_mode:
                print(f"请求索引: {i}")
            
            try:
                query_and_preserve(headers)
                success_count += 1
                
                if random_boolean() and random_boolean():
                    query_one_and_cancel(headers)
                else:
                    query_order_and_pay(headers, pairs)
                    query_and_collect_ticket(headers)
                    query_and_enter_station(headers)
                    
            except Exception as query_error:
                if not quiet_mode:
                    print(f"查询操作失败: {query_error}")
            
            i += 1
            
            # 安静模式下每50次操作报告一次状态
            if quiet_mode and i % 50 == 0:
                success_rate = (success_count / i) * 100
                print(f"[压测状态] 完成请求: {i}, 成功率: {success_rate:.1f}%")
            
            time.sleep(request_interval)
            
            if not running_flag():
                if quiet_mode:
                    print(f"[压测] 收到停止信号，退出测试循环 (完成 {i} 次请求)")
                else:
                    print("收到停止信号，退出测试循环")
                break
                
        except Exception as e:
            if not quiet_mode:
                print(f"主循环在索引 {i} 处发生错误: {e}")
            i += 1
            if not running_flag():
                break

    # 最终统计
    if quiet_mode:
        success_rate = (success_count / max(i, 1)) * 100
        print(f"[压测完成] 总请求: {i}, 成功: {success_count}, 成功率: {success_rate:.1f}%")


def main_thread(thread_count=5, request_interval=1.0, running_flag=None):
    if running_flag is None:
        running_flag = lambda: True
    
    # 检查是否由主控制器调用
    quiet_mode = os.getenv('PRESSURE_TEST_QUIET', 'false').lower() == 'true'
    
    threads = []

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if quiet_mode:
        print(f"[压测启动] 时间: {start_time}, 线程数: {thread_count}")
    else:
        print(f"测试开始: {start_time}, 线程数: {thread_count}, 请求间隔: {request_interval}秒")

    for i in range(thread_count):
        t = Thread(name="thread" + str(i), target=main, args=(request_interval, running_flag))
        time.sleep(1)
        t.start()
        threads.append(t)

    # 等待所有线程完成或收到停止信号
    for t in threads:
        while t.is_alive() and running_flag():
            time.sleep(0.1)
        if not running_flag():
            if quiet_mode:
                print("[压测] 主线程收到停止信号")
            else:
                print("主线程收到停止信号")
            break

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if quiet_mode:
        print(f"[压测结束] 时间: {end_time}")
    else:
        print(f"测试开始: {start_time}, 测试结束: {end_time}")


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