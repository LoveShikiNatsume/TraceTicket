import subprocess
import json
import os
from datetime import datetime

# --- Configuration ---
REMOTE_HOST = "192.168.1.102"
REMOTE_PORT = 22
REMOTE_USER = "aiops_admin"
REMOTE_SCRIPT_PATH = "/home/aiops_admin/chaos_injection_master.sh"

# --- Internal Paths ---
REMOTE_TEMP_LOG = "/tmp/last_chaos_injection.log"
LOCAL_RECORDS_DIR = "fault_injection_records"

def run_remote_script():
    """
    使用SSH远程连接并以交互模式运行脚本。
    """
    print(f"你好, liningshuai!")
    print(f"🚀 准备启动远程故障注入脚本于 {REMOTE_USER}@{REMOTE_HOST}...")
    print("---------------------------------------------------------")
    
    command = [
        "ssh", "-t", f"{REMOTE_USER}@{REMOTE_HOST}",
        f"bash {REMOTE_SCRIPT_PATH}"
    ]
    
    try:
        process = subprocess.run(" ".join(command), shell=True, check=True)
        print("---------------------------------------------------------")
        print("✅ 远程脚本执行完毕。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 远程脚本执行出错: {e}")
        return False
    except FileNotFoundError:
        print("❌ 错误: 'ssh' 命令未找到。请确认OpenSSH客户端已安装并在系统PATH中。")
        return False


def fetch_and_parse_log():
    """
    使用SCP从远程服务器下载临时日志文件，并解析其内容。
    """
    print(f"⬇️ 正在从服务器下载结果日志...")
    local_temp_log = "temp_chaos_log.txt"
    remote_source = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_TEMP_LOG}"
    
    scp_command = ["scp", remote_source, local_temp_log]

    try:
        # --- MODIFICATION FOR OLDER PYTHON VERSIONS ---
        # 将 capture_output=True 替换为功能等价的 stdout 和 stderr 参数
        subprocess.run(
            " ".join(scp_command), 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        with open(local_temp_log, 'r', encoding='utf-8') as f:
            line = f.read().strip()
        
        if not line:
            print(f"❌ 错误: 下载的日志文件是空的。远程脚本可能未成功写入日志。")
            return None
            
        parts = line.split(';')
        if len(parts) != 4:
            print(f"❌ 错误: 日志文件格式不正确: {line}")
            return None
        
        os.remove(local_temp_log)
        cleanup_remote_log()
        
        print("✅ 成功获取并解析结果日志。")
        return {
            "start_time": parts[0], "end_time": parts[1],
            "fault_type": parts[2], "description": parts[3]
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 使用 SCP 下载日志文件时出错: {e.stderr.decode()}")
        print("   请确认您已经配置了到 192.168.1.102 的SSH免密登录。")
        return None
    except FileNotFoundError:
        print("❌ 错误: 'scp' 命令未找到。")
        return None

def cleanup_remote_log():
    """
    远程删除临时日志文件，保持服务器整洁。
    """
    print("🧹 正在清理服务器上的临时日志文件...")
    cleanup_command = [
        "ssh", f"{REMOTE_USER}@{REMOTE_HOST}",
        f"rm {REMOTE_TEMP_LOG}"
    ]
    subprocess.run(" ".join(cleanup_command), shell=True)


def generate_json_record(fault_data):
    """
    根据获取的数据生成并更新当天的JSON故障记录文件。
    """
    print("✍️ 正在生成 JSON 故障记录文件...")
    
    try:
        start_dt = datetime.strptime(fault_data["start_time"], "%Y-%m-%d %H:%M:%S")
        date_str = start_dt.strftime("%Y%m%d")
        minute_key = start_dt.strftime("%H_%M")
    except (ValueError, KeyError):
        print("❌ 错误: 无法从获取的数据中解析时间。")
        return

    new_record = {
        "start_time": fault_data["start_time"], "end_time": fault_data["end_time"],
        "minute_key": minute_key, "fault_type": fault_data["fault_type"],
        "description": fault_data["description"]
    }

    os.makedirs(LOCAL_RECORDS_DIR, exist_ok=True)
    json_file_path = os.path.join(LOCAL_RECORDS_DIR, f"fault_records_{date_str}.json")

    records = []
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            try: records = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ 警告: 无法解析已存在的JSON文件 {json_file_path}。将覆盖此文件。")
                records = []

    records.append(new_record)

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ 成功更新故障记录文件: {json_file_path}")


if __name__ == "__main__":
    if run_remote_script():
        fault_data = fetch_and_parse_log()
        if fault_data:
            generate_json_record(fault_data)
            print("\n🎉 liningshuai，所有步骤已成功完成！")
