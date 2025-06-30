# -*- coding: utf-8 -*-
"""
Train Ticket 异常检测系统主控制器
一键启动：压测 + 数据采集 + 实时异常检测

Author: LoveShikiNatsume
Date: 2025-06-18
Version: 1.0 - 简化版，默认实时监控模式
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# 添加子目录到路径
sys.path.append(str(Path(__file__).parent / "train-ticket-trace-collect"))

class TrainTicketAnomalyDetectionController:
    """Train Ticket 异常检测系统主控制器 - 一键启动模式"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config = self._load_default_config()
        self.logger = self._setup_logging()
        
        # 各组件的状态
        self.component_status = {
            "load_test": "未开始",
            "data_collection": "未开始",
            "anomaly_detection": "未开始"
        }
        
        # 运行结果存储
        self.results = {
            "start_time": None,
            "end_time": None,
            "real_time_detections": []
        }
        
        # 实时监控相关
        self.monitoring_active = False
        self.last_processed_minute = None
        
        self.logger.info("🚀 Train Ticket 异常检测系统启动")
        self.logger.info("一键启动模式：自动压测 + 数据采集 + 实时异常检测")
        self.logger.info("按 Ctrl+C 停止运行")

    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        return {
            "real_time_mode": {
                "enabled": True,
                "check_interval_seconds": 30,
                "auto_process_delay_seconds": 65,
                "detection_threshold": 0.15
            },
            "data_collection": {
                "interval_seconds": 60,
                "lookback_period": "5m"
            }
        }

    def _setup_logging(self):
        """设置日志"""
        logger = logging.getLogger('MainController')
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - [主控制器] - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"main_controller_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def _run_subprocess(self, cmd: List[str], component_name: str, 
                       timeout: Optional[int] = None, 
                       background: bool = False) -> subprocess.Popen:
        """运行子进程"""
        self.logger.info(f"启动 {component_name}")
        
        try:
            if background:
                # 后台运行
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.project_root
                )
                self.component_status[component_name.lower().replace(' ', '_')] = "运行中"
                return process
            else:
                # 前台运行并等待完成
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.project_root
                )
                
                if result.returncode == 0:
                    self.logger.info(f"{component_name} 执行成功")
                    self.component_status[component_name.lower().replace(' ', '_')] = "完成"
                else:
                    self.logger.error(f"{component_name} 执行失败: {result.stderr}")
                    self.component_status[component_name.lower().replace(' ', '_')] = "失败"
                
                return result
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"{component_name} 执行超时")
            self.component_status[component_name.lower().replace(' ', '_')] = "超时"
        except Exception as e:
            self.logger.error(f"{component_name} 执行异常: {e}")
            self.component_status[component_name.lower().replace(' ', '_')] = "异常"

    def start_load_test(self) -> Optional[subprocess.Popen]:
        """启动压测（模拟）"""
        self.logger.info("启动压测模拟器...")
        
        cmd = [
            sys.executable, "-c", 
            f"""
import time
import random
print("🔄 压测开始运行...")
count = 0
while True:
    count += 1
    if count % 10 == 0:
        print(f"压测进行中... 已运行 {{count//10}} 分钟")
    time.sleep(30)
            """
        ]
        
        return self._run_subprocess(cmd, "load_test", background=True)

    def start_data_collection(self, duration_minutes: int = 0) -> subprocess.Popen:
        """启动数据采集"""
        collector_script = self.project_root / "train-ticket-trace-collect" / "trace_collector.py"
        
        cmd = [
            sys.executable, str(collector_script),
            "--duration", str(duration_minutes)
        ]
        
        return self._run_subprocess(cmd, "data_collection", background=True)

    def process_collected_data(self, target_date: str = None):
        """处理采集的数据（图分析）"""
        processor_script = self.project_root / "train-ticket-trace-collect" / "graph_post_processor.py"
        
        cmd = [sys.executable, str(processor_script)]
        if target_date:
            cmd.extend(["--date", target_date])
        
        self._run_subprocess(cmd, "graph_processing", timeout=1800)

    def check_for_new_data(self, target_date: str = None) -> List[str]:
        """检查是否有新的分钟级数据文件"""
        target_date = target_date or datetime.now().strftime("%Y-%m-%d")
        trace_output_dir = self.project_root / "trace_output" / target_date / "csv"
        
        if not trace_output_dir.exists():
            return []
        
        # 获取所有CSV文件
        csv_files = list(trace_output_dir.glob("*.csv"))
        new_files = []
        
        current_time = datetime.now()
        
        for csv_file in csv_files:
            # 解析文件名 (格式: HH_MM.csv)
            try:
                filename = csv_file.stem
                hour, minute = filename.split('_')
                file_time = datetime.now().replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
                
                # 检查文件是否是新的且已经"成熟"（超过65秒）
                time_diff = (current_time - file_time).total_seconds()
                
                if time_diff >= self.config["real_time_mode"]["auto_process_delay_seconds"]:
                    file_key = f"{target_date}_{filename}"
                    if self.last_processed_minute is None or file_time > self.last_processed_minute:
                        new_files.append(str(csv_file))
                        self.last_processed_minute = file_time
                        
            except ValueError:
                continue
        
        return new_files

    def run_single_file_anomaly_detection(self, csv_file: str) -> Dict:
        """对单个CSV文件运行异常检测（模拟）"""
        self.logger.info(f"🔍 分析文件: {os.path.basename(csv_file)}")
        
        # 模拟异常检测
        import random
        import pandas as pd
        
        try:
            # 读取文件基本信息
            df = pd.read_csv(csv_file)
            trace_count = len(df['traceIdLow'].unique()) if 'traceIdLow' in df.columns else 0
            span_count = len(df)
            
            # 模拟异常检测结果
            anomaly_score = random.uniform(0.0, 1.0)
            threshold = self.config["real_time_mode"]["detection_threshold"]
            is_anomaly = anomaly_score > threshold
            
            result = {
                "file_name": os.path.basename(csv_file),
                "analysis_time": datetime.now().isoformat(),
                "trace_count": trace_count,
                "span_count": span_count,
                "anomaly_score": round(anomaly_score, 4),
                "threshold": threshold,
                "anomaly_detected": is_anomaly,
                "anomaly_types": []
            }
            
            if is_anomaly:
                # 随机生成异常类型
                possible_anomalies = ["high_latency", "error_spike", "unusual_pattern", "service_degradation"]
                result["anomaly_types"] = random.sample(possible_anomalies, random.randint(1, 2))
            
            return result
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            return {}

    def report_real_time_anomaly(self, csv_file: str, detection_result: Dict):
        """实时异常报告"""
        self.logger.warning("🚨 ================================")
        self.logger.warning("🚨 检测到实时异常！")
        self.logger.warning("🚨 ================================")
        self.logger.warning(f"📁 文件: {detection_result['file_name']}")
        self.logger.warning(f"⚠️  异常分数: {detection_result['anomaly_score']} (阈值: {detection_result['threshold']})")
        self.logger.warning(f"🏷️  异常类型: {', '.join(detection_result.get('anomaly_types', []))}")
        self.logger.warning(f"📊 Trace数量: {detection_result['trace_count']}")
        self.logger.warning(f"📈 Span数量: {detection_result['span_count']}")
        self.logger.warning("🚨 ================================")
        
        # 保存异常报告
        anomaly_report_dir = self.project_root / "anomaly_reports"
        anomaly_report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = anomaly_report_dir / f"realtime_anomaly_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(detection_result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 异常报告已保存: {report_file}")

    def process_new_files_real_time(self, csv_files: List[str]) -> bool:
        """实时处理新的CSV文件"""
        if not csv_files:
            return False
        
        self.logger.info(f"📋 发现 {len(csv_files)} 个新的数据文件，开始实时处理...")
        
        # 获取日期
        target_date = datetime.now().strftime("%Y-%m-%d")
        date_dir = self.project_root / "trace_output" / target_date
        
        # 检查是否需要图分析处理
        graph_flag = date_dir / ".graph_processed"
        need_graph_processing = not graph_flag.exists()
        
        if need_graph_processing:
            self.logger.info("🔧 首次处理，执行图分析...")
            self.process_collected_data(target_date)
        
        # 对每个新文件进行异常检测
        for csv_file in csv_files:
            try:
                detection_result = self.run_single_file_anomaly_detection(csv_file)
                if detection_result:
                    self.results["real_time_detections"].append({
                        "timestamp": datetime.now().isoformat(),
                        "file": os.path.basename(csv_file),
                        "result": detection_result
                    })
                    
                    # 如果检测到异常，立即报告
                    if detection_result.get("anomaly_detected", False):
                        self.report_real_time_anomaly(csv_file, detection_result)
                    else:
                        self.logger.info(f"✅ {os.path.basename(csv_file)} - 正常")
                        
            except Exception as e:
                self.logger.error(f"❌ 处理文件 {csv_file} 时出错: {e}")
        
        return True

    def run_real_time_monitoring(self):
        """运行实时监控模式"""
        self.logger.info("🚀 启动实时异常检测监控系统")
        self.logger.info("📊 系统将自动：")
        self.logger.info("   1. 启动压测模拟")
        self.logger.info("   2. 开始数据采集")
        self.logger.info("   3. 实时异常检测")
        self.logger.info("   4. 异常自动报警")
        self.logger.info("🛑 按 Ctrl+C 停止运行")
        
        self.monitoring_active = True
        self.results["start_time"] = datetime.now().isoformat()
        
        # 启动压测（后台运行）
        self.logger.info("🔄 启动压测...")
        load_test_process = self.start_load_test()
        
        # 启动数据采集（后台运行）
        self.logger.info("📡 启动数据采集...")
        collection_process = self.start_data_collection(duration_minutes=0)
        
        # 等待数据采集启动
        self.logger.info("⏳ 等待数据采集启动...")
        time.sleep(30)
        
        check_interval = self.config["real_time_mode"]["check_interval_seconds"]
        self.logger.info(f"🔍 开始实时监控 (每{check_interval}秒检查一次新数据)")
        
        try:
            while self.monitoring_active:
                # 检查新数据
                new_files = self.check_for_new_data()
                
                if new_files:
                    self.process_new_files_real_time(new_files)
                
                # 显示运行状态
                elapsed_hours = (time.time() - time.mktime(datetime.fromisoformat(self.results["start_time"]).timetuple())) / 3600
                detection_count = len(self.results["real_time_detections"])
                anomaly_count = len([d for d in self.results["real_time_detections"] if d["result"].get("anomaly_detected", False)])
                
                self.logger.info(f"📈 监控状态: 运行{elapsed_hours:.1f}小时 | 检测{detection_count}次 | 异常{anomaly_count}次")
                
                # 等待下次检查
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 用户中断监控")
        finally:
            self.monitoring_active = False
            
            # 停止所有后台进程
            if load_test_process:
                self.logger.info("🛑 停止压测...")
                load_test_process.terminate()
                try:
                    load_test_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    load_test_process.kill()
            
            if collection_process:
                self.logger.info("🛑 停止数据采集...")
                collection_process.terminate()
                try:
                    collection_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    collection_process.kill()
            
            self.results["end_time"] = datetime.now().isoformat()
            self.generate_final_report()

    def generate_final_report(self):
        """生成最终报告"""
        report_dir = self.project_root / "experiment_reports"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        detection_count = len(self.results["real_time_detections"])
        anomaly_count = len([d for d in self.results["real_time_detections"] if d["result"].get("anomaly_detected", False)])
        
        # 生成简化摘要
        summary_file = report_dir / f"monitoring_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Train Ticket 实时异常检测监控报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"监控时间: {self.results['start_time']} ~ {self.results['end_time']}\n")
            f.write(f"监控时长: {self._calculate_duration_minutes()/60:.1f} 小时\n")
            f.write(f"检测次数: {detection_count}\n")
            f.write(f"异常次数: {anomaly_count}\n")
            f.write(f"异常率: {(anomaly_count / max(detection_count, 1)) * 100:.1f}%\n\n")
            
            if anomaly_count > 0:
                f.write("异常检测详情:\n")
                for detection in self.results["real_time_detections"]:
                    if detection["result"].get("anomaly_detected", False):
                        result = detection["result"]
                        f.write(f"  时间: {detection['timestamp']}\n")
                        f.write(f"  文件: {detection['file']}\n")
                        f.write(f"  异常分数: {result['anomaly_score']}\n")
                        f.write(f"  异常类型: {', '.join(result.get('anomaly_types', []))}\n")
                        f.write("\n")
        
        self.logger.info("📋 ====== 监控完成 ======")
        self.logger.info(f"📊 总计检测: {detection_count} 次")
        self.logger.info(f"🚨 发现异常: {anomaly_count} 次")
        self.logger.info(f"📁 监控报告: {summary_file}")
        self.logger.info("📋 ========================")

    def _calculate_duration_minutes(self) -> float:
        """计算运行总时长"""
        if self.results["start_time"] and self.results["end_time"]:
            start = datetime.fromisoformat(self.results["start_time"])
            end = datetime.fromisoformat(self.results["end_time"])
            return (end - start).total_seconds() / 60
        return 0

def main():
    """主函数 - 一键启动"""
    print("=" * 60)
    print("🚀 Train Ticket 异常检测系统")
    print("=" * 60)
    print("一键启动模式：自动运行所有组件")
    print("功能包括：压测 + 数据采集 + 实时异常检测")
    print("按 Ctrl+C 可随时停止")
    print("=" * 60)
    
    try:
        controller = TrainTicketAnomalyDetectionController()
        controller.run_real_time_monitoring()
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 系统已停止")
        return 0
    except Exception as e:
        print(f"❌ 系统运行失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
