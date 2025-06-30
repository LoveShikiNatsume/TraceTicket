# -*- coding: utf-8 -*-
"""
Train Ticket 异常检测系统主控制器
一键启动：压测 + 数据采集 + 异常注入 + 实时异常检测

Author: LoveShikiNatsume
Date: 2025-06-18
Version: 1.1 实时监控流程验证
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
            "anomaly_detection": "未开始",
            "fault_injection": "未开始"
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
                "detection_threshold": 0.15,
                "warmup_minutes": 3  # 压测预热时间，之后开始注入故障
            },
            "data_collection": {
                "interval_seconds": 60,
                "lookback_period": "5m"
            },
            "scripts": {
                "load_test": "load-testing/load_test.py",
                "fault_injection": "fault-injection/fault_injector.py",
                "anomaly_detection": "anomaly-detection/vae_detector.py"
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
        """启动压测脚本（模拟调用）"""
        self.logger.info("🔄 启动压测脚本...")
        
        # 模拟启动外部压测脚本（实际应用时替换为真实脚本路径）
        script_path = self.project_root / self.config["scripts"]["load_test"]
        
        if not os.path.exists(script_path):
            # 模拟脚本不存在时的处理
            self.logger.warning(f"压测脚本不存在: {script_path}, 使用模拟实现")
            # 生成模拟脚本输出目录
            output_dir = self.project_root / "load-testing" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入模拟数据
            with open(output_dir / f"loadtest_status_{datetime.now().strftime('%Y%m%d')}.txt", "w") as f:
                f.write("Load testing active\n")
                f.write(f"Start time: {datetime.now().isoformat()}\n")
                f.write("Users: 100\n")
                f.write("Requests per second: 50\n")
        
        # 这里模拟启动压测脚本，返回一个假的进程
        # 实际应该是：return self._run_subprocess([sys.executable, str(script_path)], "load_test", background=True)
        
        # 占位用假进程
        cmd = [
            sys.executable, "-c", 
            """
import time
print("压测脚本启动")
while True:
    time.sleep(10)  # 模拟运行中
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

    def start_fault_injection(self) -> Optional[subprocess.Popen]:
        """启动故障注入（模拟调用外部脚本）"""
        self.logger.info("💥 启动故障注入脚本...")
        
        # 模拟启动外部故障注入脚本（实际应用时替换为真实脚本路径）
        script_path = self.project_root / self.config["scripts"]["fault_injection"]
        
        if not os.path.exists(script_path):
            # 模拟脚本不存在时生成记录
            self.logger.warning(f"故障注入脚本不存在: {script_path}, 使用模拟实现")
            # 创建记录目录
            record_dir = self.project_root / "fault_injection_records"
            record_dir.mkdir(exist_ok=True)
            
            # 写入模拟的故障记录
            self._generate_mock_fault_records()
        
        # 这里模拟启动故障注入脚本，返回一个假的进程
        # 实际应该是：return self._run_subprocess([sys.executable, str(script_path)], "fault_injection", background=True)
        
        # 占位用假进程
        cmd = [
            sys.executable, "-c", 
            """
import time
print("故障注入脚本启动")
while True:
    time.sleep(10)  # 模拟运行中
            """
        ]
        
        return self._run_subprocess(cmd, "fault_injection", background=True)

    def check_for_new_data(self, target_date: str = None) -> List[str]:
        """检查是否有新的分钟级数据文件"""
        target_date = target_date or datetime.now().strftime("%Y-%m-%d")
        trace_dir = self.project_root / "trace" / target_date / "csv"  # 修改为trace目录
        
        if not trace_dir.exists():
            return []
        
        # 获取所有CSV文件
        csv_files = list(trace_dir.glob("*.csv"))
        new_files = []
        
        current_time = datetime.now()
        
        for csv_file in csv_files:
            # 检查是否已有对应的.graph_processed标志文件
            flag_file = str(csv_file).replace('.csv', '.graph_processed')
            
            if os.path.exists(flag_file):
                continue  # 已经处理过，跳过
            
            # 解析文件名 (格式: HH_MM.csv)
            try:
                filename = csv_file.stem
                hour, minute = filename.split('_')
                file_time = datetime.now().replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
                
                # 检查文件是否已经"成熟"（超过65秒）
                time_diff = (current_time - file_time).total_seconds()
                
                if time_diff >= self.config["real_time_mode"]["auto_process_delay_seconds"]:
                    new_files.append(str(csv_file))
                        
            except ValueError:
                continue
        
        return new_files

    def process_collected_data(self, csv_file_path: str = None, target_date: str = None):
        """处理采集的数据（图分析）"""
        processor_script = self.project_root / "train-ticket-trace-collect" / "graph_post_processor.py"
        
        if csv_file_path:
            # 处理特定文件
            cmd = [sys.executable, str(processor_script), "--file", csv_file_path]
        else:
            # 处理整个日期
            cmd = [sys.executable, str(processor_script)]
            if target_date:
                cmd.extend(["--date", target_date])
        
        self._run_subprocess(cmd, "graph_processing", timeout=300)  # 降低超时时间

    def _generate_mock_fault_records(self):
        """生成模拟的故障注入记录文件（模拟外部注入脚本的行为）"""
        record_dir = self.project_root / "fault_injection_records"
        today = datetime.now().strftime("%Y-%m-%d")
        record_file = record_dir / f"fault_records_{today}.json"
        
        # 如果已经存在，就不重复生成
        if record_file.exists():
            return
        
        # 生成若干个间隔的故障记录
        current_time = datetime.now()
        records = []
        
        # 模拟每10分钟产生一次故障，持续5分钟
        for i in range(6):  # 6个周期，共60分钟
            # 故障开始时间：当前时间 + 预热时间 + i*10分钟
            fault_time = current_time + timedelta(minutes=self.config["real_time_mode"]["warmup_minutes"] + i*10)
            
            # 故障记录
            for j in range(5):  # 每次故障持续5分钟
                minute_time = fault_time + timedelta(minutes=j)
                
                # 随机选择故障类型
                fault_types = [
                    {"type": "high_latency", "description": "高延迟故障", "intensity": "medium"},
                    {"type": "error_injection", "description": "错误注入故障", "intensity": "high"},
                    {"type": "service_unavailable", "description": "服务不可用故障", "intensity": "high"},
                    {"type": "network_delay", "description": "网络延迟故障", "intensity": "low"}
                ]
                
                import random
                fault_type = random.choice(fault_types)
                
                record = {
                    "timestamp": minute_time.isoformat(),
                    "minute_key": minute_time.strftime("%H_%M"),
                    "date": today,
                    "fault_type": fault_type["type"],
                    "description": fault_type["description"],
                    "intensity": fault_type["intensity"],
                    "expected_anomaly": True
                }
                
                records.append(record)
            
            # 故障间歇期（5分钟）
            for j in range(5):
                minute_time = fault_time + timedelta(minutes=j+5)
                
                record = {
                    "timestamp": minute_time.isoformat(),
                    "minute_key": minute_time.strftime("%H_%M"),
                    "date": today,
                    "fault_type": "normal",
                    "description": "系统正常运行",
                    "intensity": "none",
                    "expected_anomaly": False
                }
                
                records.append(record)
        
        # 保存故障记录文件
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"已生成模拟故障记录: {record_file}")
        self.logger.info(f"共 {len(records)} 条记录，包括故障期和正常期")

    def process_new_files_real_time(self, csv_files: List[str]) -> bool:
        """实时处理新的CSV文件"""
        if not csv_files:
            return False
        
        self.logger.info(f"📋 发现 {len(csv_files)} 个新的数据文件，开始实时处理...")
        
        # 对每个新文件进行图分析和异常检测
        for csv_file in csv_files:
            try:
                # 先进行图分析处理
                self.logger.info(f"🔧 对文件进行图分析: {os.path.basename(csv_file)}")
                self.process_collected_data(csv_file_path=csv_file)
                
                # 检查是否生成了处理标志
                flag_file = csv_file.replace('.csv', '.graph_processed')
                if not os.path.exists(flag_file):
                    self.logger.warning(f"图分析可能失败，未找到标志文件: {os.path.basename(flag_file)}")
                    continue
                
                # 调用异常检测
                detection_result = self.run_anomaly_detection(csv_file)
                
                # 验证结果
                validation = self.validate_detection_result(detection_result)
                
                # 报告检测结果
                self.report_anomaly_detection(csv_file, detection_result, validation)
                
            except Exception as e:
                self.logger.error(f"❌ 处理文件 {csv_file} 时出错: {e}")
        
        return True

    def run_anomaly_detection(self, csv_file: str) -> Dict:
        """调用异常检测脚本（模拟）"""
        self.logger.info(f"🔍 对文件运行异常检测: {os.path.basename(csv_file)}")
        
        # 模拟启动外部异常检测脚本（实际应用时替换为真实脚本路径）
        script_path = self.project_root / self.config["scripts"]["anomaly_detection"]
        
        if not os.path.exists(script_path):
            self.logger.warning(f"异常检测脚本不存在: {script_path}, 使用模拟实现")
        
        # 这里模拟异常检测结果，实际应该调用VAE脚本
        # 实际应该是：
        # result = self._run_subprocess([sys.executable, str(script_path), "--file", csv_file], "anomaly_detection")
        # 然后解析result.stdout获取结果
        
        # 读取文件基本信息（假装我们分析了它）
        import pandas as pd
        try:
            df = pd.read_csv(csv_file)
            trace_count = len(df['traceIdLow'].unique()) if 'traceIdLow' in df.columns else 0
            span_count = len(df)
        except Exception as e:
            self.logger.error(f"读取CSV文件失败: {e}")
            trace_count = 0
            span_count = 0
        
        # 模拟异常检测结果
        import random
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
        
        # 生成异常检测输出目录
        output_dir = self.project_root / "anomaly_detection" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 将结果保存为JSON文件（模拟异常检测脚本的输出）
        result_file = output_dir / f"detection_{os.path.basename(csv_file).replace('.csv', '')}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        return result

    def load_fault_injection_records(self, target_date: str = None) -> List[Dict]:
        """加载故障注入记录"""
        target_date = target_date or datetime.now().strftime("%Y-%m-%d")
        fault_record_dir = self.project_root / "fault_injection_records"
        record_file = fault_record_dir / f"fault_records_{target_date}.json"
        
        if not record_file.exists():
            return []
        
        try:
            with open(record_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载故障记录失败: {e}")
            return []

    def validate_detection_result(self, detection_result: Dict) -> Dict:
        """验证检测结果与故障注入记录的匹配度"""
        file_name = detection_result.get("file_name", "")
        if not file_name.endswith(".csv"):
            return {"validation": "skipped", "reason": "invalid_filename"}
        
        # 从文件名提取时间 (HH_MM.csv)
        try:
            minute_key = file_name.replace(".csv", "")
            target_date = datetime.now().strftime("%Y-%m-%d")
        except:
            return {"validation": "failed", "reason": "filename_parse_error"}
        
        # 加载故障记录
        fault_records = self.load_fault_injection_records(target_date)
        
        # 查找对应时间的故障记录
        matching_record = None
        for record in fault_records:
            if record.get("minute_key") == minute_key:
                matching_record = record
                break
        
        if not matching_record:
            return {
                "validation": "no_fault_record",
                "reason": f"未找到时间 {minute_key} 的故障记录"
            }
        
        # 验证检测结果
        expected_anomaly = matching_record.get("expected_anomaly", False)
        detected_anomaly = detection_result.get("anomaly_detected", False)
        
        validation_result = {
            "validation": "completed",
            "minute_key": minute_key,
            "expected_anomaly": expected_anomaly,
            "detected_anomaly": detected_anomaly,
            "fault_info": {
                "type": matching_record.get("fault_type"),
                "description": matching_record.get("description"),
                "intensity": matching_record.get("intensity")
            }
        }
        
        # 判断检测准确性
        if expected_anomaly == detected_anomaly:
            if expected_anomaly:
                validation_result["accuracy"] = "true_positive"  # 正确检测到异常
                validation_result["result"] = "✅ 正确检测"
            else:
                validation_result["accuracy"] = "true_negative"  # 正确识别正常
                validation_result["result"] = "✅ 正确识别"
        else:
            if expected_anomaly and not detected_anomaly:
                validation_result["accuracy"] = "false_negative"  # 漏报
                validation_result["result"] = "❌ 漏报异常"
            else:
                validation_result["accuracy"] = "false_positive"  # 误报
                validation_result["result"] = "❌ 误报异常"
        
        return validation_result
    
    def report_anomaly_detection(self, csv_file: str, detection_result: Dict, validation: Dict):
        """报告异常检测结果"""
        if detection_result.get("anomaly_detected", False):
            self.logger.warning("🚨 ================================")
            self.logger.warning("🚨 检测到实时异常！")
            self.logger.warning("🚨 ================================")
            self.logger.warning(f"📁 文件: {detection_result['file_name']}")
            self.logger.warning(f"⚠️  异常分数: {detection_result['anomaly_score']} (阈值: {detection_result['threshold']})")
            self.logger.warning(f"🏷️  异常类型: {', '.join(detection_result.get('anomaly_types', []))}")
            self.logger.warning(f"📊 Trace数量: {detection_result['trace_count']}")
            self.logger.warning(f"📈 Span数量: {detection_result['span_count']}")
            
            # 显示验证结果
            if validation.get("validation") == "completed":
                self.logger.warning(f"🔍 验证结果: {validation.get('result', '未知')}")
                self.logger.warning(f"📋 故障类型: {validation.get('fault_info', {}).get('description', '未知')}")
                self.logger.warning(f"🎯 准确性: {validation.get('accuracy', '未知')}")
            
            self.logger.warning("🚨 ================================")
        else:
            # 显示正常检测结果的验证
            if validation.get("validation") == "completed":
                result_emoji = "✅" if validation.get("accuracy") in ["true_negative", "true_positive"] else "❌"
                self.logger.info(f"{result_emoji} {os.path.basename(csv_file)} - 正常 ({validation.get('result', '未验证')})")
            else:
                self.logger.info(f"✅ {os.path.basename(csv_file)} - 正常")

        # 保存到结果列表
        self.results["real_time_detections"].append({
            "timestamp": datetime.now().isoformat(),
            "file": os.path.basename(csv_file),
            "result": detection_result,
            "validation": validation
        })

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
                # 调用异常检测（模拟）
                detection_result = self.run_anomaly_detection(csv_file)
                
                # 验证结果
                validation = self.validate_detection_result(detection_result)
                
                # 报告检测结果
                self.report_anomaly_detection(csv_file, detection_result, validation)
                
            except Exception as e:
                self.logger.error(f"❌ 处理文件 {csv_file} 时出错: {e}")
        
        return True

    def run_real_time_monitoring(self):
        """运行实时监控模式"""
        self.logger.info("🚀 启动实时异常检测监控系统")
        self.logger.info("📊 系统将自动：")
        self.logger.info("   1. 启动压测")
        self.logger.info("   2. 开始数据采集")
        self.logger.info(f"   3. {self.config['real_time_mode']['warmup_minutes']}分钟后自动启动故障注入")
        self.logger.info("   4. 实时异常检测")
        self.logger.info("   5. 检测结果验证")
        self.logger.info("🛑 按 Ctrl+C 停止运行")
        
        self.monitoring_active = True
        self.results["start_time"] = datetime.now().isoformat()
        
        # 启动压测
        self.logger.info("🔄 启动压测...")
        load_test_process = self.start_load_test()
        
        # 启动数据采集
        self.logger.info("📡 启动数据采集...")
        collection_process = self.start_data_collection(duration_minutes=0)  # 持续运行
        
        # 等待预热期结束
        warmup_minutes = self.config["real_time_mode"]["warmup_minutes"]
        self.logger.info(f"⏳ 等待 {warmup_minutes} 分钟预热期...")
        
        # 预热期后再启动故障注入
        fault_injection_process = None
        
        # 启动监控循环
        check_interval = self.config["real_time_mode"]["check_interval_seconds"]
        self.logger.info(f"🔍 开始实时监控 (每{check_interval}秒检查一次新数据)")
        
        start_time = time.time()
        try:
            while self.monitoring_active:
                elapsed_minutes = (time.time() - start_time) / 60
                
                # 到达预热时间后，启动故障注入
                if elapsed_minutes >= warmup_minutes and fault_injection_process is None:
                    self.logger.info(f"🔥 预热期结束，启动故障注入...")
                    fault_injection_process = self.start_fault_injection()
                    self.component_status["fault_injection"] = "运行中"
                
                # 检查新数据
                new_files = self.check_for_new_data()
                
                if new_files:
                    self.process_new_files_real_time(new_files)
                
                # 显示运行状态
                elapsed_hours = elapsed_minutes / 60
                detection_count = len(self.results["real_time_detections"])
                anomaly_count = len([d for d in self.results["real_time_detections"] if d["result"].get("anomaly_detected", False)])
                
                # 计算验证统计
                validated_detections = [d for d in self.results["real_time_detections"] 
                                      if d["validation"].get("validation") == "completed"]
                accuracy_stats = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
                
                for detection in validated_detections:
                    accuracy = detection["validation"].get("accuracy", "unknown")
                    if accuracy in accuracy_stats:
                        accuracy_stats[accuracy] += 1
                
                total_validated = sum(accuracy_stats.values())
                accuracy_rate = ((accuracy_stats["true_positive"] + accuracy_stats["true_negative"]) / max(total_validated, 1)) * 100
                
                self.logger.info(f"📈 监控状态: 运行{elapsed_hours:.1f}小时 | 检测{detection_count}次 | 异常{anomaly_count}次 | 准确率{accuracy_rate:.1f}%")
                
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
            
            if fault_injection_process:
                self.logger.info("🛑 停止故障注入...")
                fault_injection_process.terminate()
                try:
                    fault_injection_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    fault_injection_process.kill()
            
            self.results["end_time"] = datetime.now().isoformat()
            self.generate_final_report()

    def generate_final_report(self):
        """生成最终报告"""
        report_dir = self.project_root / "experiment_reports"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"experiment_report_{timestamp}.json"
        
        detection_count = len(self.results["real_time_detections"])
        anomaly_count = len([d for d in self.results["real_time_detections"] if d["result"].get("anomaly_detected", False)])
        
        # 计算验证统计
        validated_detections = [d for d in self.results["real_time_detections"] 
                              if d["validation"].get("validation") == "completed"]
        
        accuracy_stats = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
        for detection in validated_detections:
            accuracy = detection["validation"].get("accuracy", "unknown")
            if accuracy in accuracy_stats:
                accuracy_stats[accuracy] += 1
        
        total_validated = sum(accuracy_stats.values())
        accuracy_rate = ((accuracy_stats["true_positive"] + accuracy_stats["true_negative"]) / max(total_validated, 1)) * 100
        
        # 生成完整报告
        full_report = {
            "experiment_info": {
                "start_time": self.results["start_time"],
                "end_time": self.results["end_time"],
                "duration_minutes": self._calculate_duration_minutes(),
                "warmup_minutes": self.config["real_time_mode"]["warmup_minutes"],
                "config": self.config
            },
            "component_status": self.component_status,
            "detection_summary": {
                "total": detection_count,
                "anomalies": anomaly_count,
                "anomaly_rate": (anomaly_count / max(detection_count, 1)) * 100,
                "accuracy_rate": accuracy_rate,
                "true_positive": accuracy_stats["true_positive"],
                "true_negative": accuracy_stats["true_negative"],
                "false_positive": accuracy_stats["false_positive"],
                "false_negative": accuracy_stats["false_negative"]
            },
            "detections": self.results["real_time_detections"]
        }
        
        # 保存完整报告
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        # 生成简化摘要
        summary_file = report_dir / f"experiment_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Train Ticket 异常检测实验报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"实验时间: {self.results['start_time']} ~ {self.results['end_time']}\n")
            f.write(f"总时长: {self._calculate_duration_minutes()/60:.1f} 小时\n")
            f.write(f"预热时长: {self.config['real_time_mode']['warmup_minutes']} 分钟\n\n")
            
            f.write("组件状态:\n")
            for component, status in self.component_status.items():
                f.write(f"  {component}: {status}\n")
            f.write("\n")
            
            f.write("检测统计:\n")
            f.write(f"  总检测次数: {detection_count}\n")
            f.write(f"  异常检测数: {anomaly_count}\n")
            f.write(f"  异常率: {(anomaly_count / max(detection_count, 1)) * 100:.1f}%\n")
            f.write(f"  检测准确率: {accuracy_rate:.1f}%\n\n")
            
            f.write("准确性分析:\n")
            f.write(f"  正确检测异常(TP): {accuracy_stats['true_positive']}\n")
            f.write(f"  正确识别正常(TN): {accuracy_stats['true_negative']}\n") 
            f.write(f"  误报异常(FP): {accuracy_stats['false_positive']}\n")
            f.write(f"  漏报异常(FN): {accuracy_stats['false_negative']}\n")
        
        self.logger.info("📋 ====== 实验完成 ======")
        self.logger.info(f"📊 总计检测: {detection_count} 次")
        self.logger.info(f"🚨 发现异常: {anomaly_count} 次") 
        self.logger.info(f"🎯 检测准确率: {accuracy_rate:.1f}%")
        self.logger.info(f"📁 完整报告: {report_file}")
        self.logger.info(f"📄 摘要报告: {summary_file}")
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
    print("功能包括：压测 + 数据采集 + 故障注入 + 实时异常检测")
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