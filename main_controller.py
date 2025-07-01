# -*- coding: utf-8 -*-
"""
Train Ticket 异常检测系统主控制器

Author: LoveShikiNatsume
Date: 2025-06-18
Version: 1.4 使用实际压测脚本测试
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
    """Train Ticket 异常检测系统主控制器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config = self._load_default_config()
        self.logger = self._setup_logging()
        
        # 记录脚本启动时间，只处理启动后的数据
        self.script_start_time = datetime.now()
        
        # 各组件的状态
        self.component_status = {
            "load_test": "未开始",
            "data_collection": "未开始",
            "metrics_collection": "未开始",  # 新增
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
        
        self.logger.info("Train Ticket 异常检测系统启动")

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
            "scripts": {  # 执行脚本路径配置
                "load_test": "train-ticket-auto-query/xxx.py",
                "fault_injection": "train-ticket-chaos-mesh/xxx.py",
                "anomaly_detection": "anomaly-detection/vae_detector.py"
            }
        }

    def _setup_logging(self):
        """设置日志"""
        logger = logging.getLogger('MainController')
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - [主控制器] - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def _run_subprocess(self, cmd: List[str], component_name: str, 
                       timeout: Optional[int] = None, 
                       background: bool = False) -> subprocess.Popen:
        """运行子进程"""
        self.logger.info(f"启动组件: {component_name}")
        
        try:
            if background:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    cwd=str(self.project_root)
                )
                self.component_status[component_name.lower().replace(' ', '_')] = "运行中"
                return process
            else:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=timeout,
                    cwd=str(self.project_root)
                )
                
                if result.returncode == 0:
                    self.logger.info(f"{component_name} 执行完成")
                    self.component_status[component_name.lower().replace(' ', '_')] = "完成"
                else:
                    self.logger.error(f"{component_name} 执行失败: {result.stderr}")
                    self.component_status[component_name.lower().replace(' ', '_')] = "失败"
                
                return result
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"{component_name} 执行超时")
            self.component_status[component_name.lower().replace(' ', '_')] = "超时"
            return None
        except Exception as e:
            self.logger.error(f"{component_name} 执行异常: {e}")
            self.component_status[component_name.lower().replace(' ', '_')] = "异常"
            return None

    def start_load_test(self) -> Optional[subprocess.Popen]:
        """启动压测脚本"""
        self.logger.info("启动压测模块...")
        
        script_path = self.project_root / self.config["scripts"]["load_test"]
        run_script = self.project_root / "train-ticket-auto-query" / "run.py"
        
        if os.path.exists(run_script):
            self.logger.info(f"使用实际压测脚本: {run_script}")
            try:
                cmd = [sys.executable, str(run_script)]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    cwd=str(self.project_root)
                )
                
                time.sleep(2)  # 等待脚本启动
                if process.poll() is None:
                    self.logger.info("压测模块启动成功")
                    return process
                else:
                    self.logger.error(f"压测启动失败，退出码: {process.returncode}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"压测启动异常: {e}")
                return None
        
        elif not os.path.exists(script_path):
            self.logger.info(f"使用模拟压测实现 (脚本路径: {script_path})")
            
            try:
                cmd = [
                    sys.executable, "-c",
                    """
import time
import sys
import signal

def signal_handler(signum, frame):
    print("模拟压测收到停止信号")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

try:
    print("模拟压测运行中...")
    while True:
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    print("模拟压测已停止")
                    """
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    cwd=str(self.project_root)
                )
                
                time.sleep(1)
                if process.poll() is None:
                    self.logger.info("压测模块启动成功")
                    return process
                else:
                    self.logger.error(f"压测启动失败，退出码: {process.returncode}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"压测启动异常: {e}")
                return None
        
        else:
            self.logger.error(f"压测脚本不存在: {script_path}")
            return None

    def start_data_collection(self, duration_minutes: int = 0) -> Optional[subprocess.Popen]:
        """启动数据采集"""
        collector_script = self.project_root / "train-ticket-trace-collect" / "trace_collector.py"
        
        if not collector_script.exists():
            self.logger.error(f"数据采集脚本不存在: {collector_script}")
            return None
        
        try:
            cmd = [
                sys.executable, str(collector_script),
                "--duration", str(duration_minutes)
            ]
            
            process = self._run_subprocess(cmd, "data_collection", background=True)
            if process:
                self.logger.info("数据采集启动成功")
                return process
            else:
                self.logger.error("数据采集启动失败")
                return None
        except Exception as e:
            self.logger.error(f"数据采集启动异常: {e}")
            return None

    def start_metrics_collection(self, duration_minutes: int = 0) -> Optional[subprocess.Popen]:
        """启动指标采集"""
        metrics_script = self.project_root / "train-ticket-trace-collect" / "metrics_collector.py"
        
        if not metrics_script.exists():
            self.logger.warning(f"指标采集脚本不存在: {metrics_script}")
            return None
        
        try:
            cmd = [
                sys.executable, str(metrics_script),
                "--duration", str(duration_minutes)
            ]
            
            process = self._run_subprocess(cmd, "metrics_collection", background=True)
            if process:
                self.logger.info("指标采集启动成功")
                return process
            else:
                self.logger.error("指标采集启动失败")
                return None
        except Exception as e:
            self.logger.error(f"指标采集启动异常: {e}")
            return None

    def start_fault_injection(self) -> Optional[subprocess.Popen]:
        """启动故障注入"""
        self.logger.info("启动故障注入模块...")
        
        script_path = self.project_root / self.config["scripts"]["fault_injection"]
        
        if not os.path.exists(script_path):
            self.logger.info(f"使用模拟故障注入实现 (脚本路径: {script_path})")
            # 创建记录目录
            record_dir = self.project_root / "fault_injection_records"
            record_dir.mkdir(exist_ok=True)
            
            # 写入模拟的故障记录
            self._generate_mock_fault_records()
        
        # 使用简化的模拟方式
        try:
            cmd = [
                sys.executable, "-c",
                """
import time
import sys
import signal

def signal_handler(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

try:
    while True:
        time.sleep(300)  # 每5分钟检查一次
except (KeyboardInterrupt, SystemExit):
    pass
                """
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=str(self.project_root)
            )
            
            time.sleep(1)
            if process.poll() is None:
                self.logger.info("故障注入模块启动成功")
                return process
            else:
                self.logger.error(f"故障注入启动失败，退出码: {process.returncode}")
                return None
                
        except Exception as e:
            self.logger.error(f"故障注入启动异常: {e}")
            return None

    def check_for_new_data(self, target_date: str = None) -> List[str]:
        """检查是否有新的分钟级数据文件"""
        target_date = target_date or datetime.now().strftime("%Y-%m-%d")
        trace_dir = self.project_root / "trace" / target_date / "csv"
        
        if not trace_dir.exists():
            return []
        
        csv_files = list(trace_dir.glob("*.csv"))
        new_files = []
        
        current_time = datetime.now()
        
        for csv_file in csv_files:
            # 检查是否已有对应的.graph_processed标志文件
            flag_file = str(csv_file).replace('.csv', '.graph_processed')
            
            if os.path.exists(flag_file):
                continue
            
            # 解析文件名
            try:
                filename = csv_file.stem
                hour, minute = filename.split('_')
                
                # 构造文件对应的时间
                file_time = current_time.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
                
                # 如果文件时间早于脚本启动时间，跳过历史数据
                if file_time < self.script_start_time:
                    self.logger.debug(f"跳过历史数据文件: {filename} (文件时间: {file_time.strftime('%H:%M')}, 启动时间: {self.script_start_time.strftime('%H:%M')})")
                    continue
                
                # 检查文件是否已经"成熟"（超过65秒）
                time_diff = (current_time - file_time).total_seconds()
                
                if time_diff >= self.config["real_time_mode"]["auto_process_delay_seconds"]:
                    new_files.append(str(csv_file))
                    self.logger.debug(f"新数据文件: {filename} (等待时间: {time_diff:.0f}s)")

            except ValueError:
                continue
        
        return new_files

    def process_collected_data(self, csv_file_path: str = None, target_date: str = None):
        """处理采集的数据（图分析）"""
        processor_script = self.project_root / "train-ticket-trace-collect" / "graph_post_processor.py"
        
        # 检查脚本是否存在
        if not processor_script.exists():
            self.logger.error(f"图分析脚本不存在: {processor_script}")
            return False
        
        try:
            if csv_file_path:
                # 处理特定文件
                cmd = [sys.executable, str(processor_script), "--file", csv_file_path]
            else:
                # 处理整个日期
                cmd = [sys.executable, str(processor_script)]
                if target_date:
                    cmd.extend(["--date", target_date])
            
            result = self._run_subprocess(cmd, "graph_processing", timeout=300)
            return result is not None and (not hasattr(result, 'returncode') or result.returncode == 0)
        except Exception as e:
            self.logger.error(f"图分析处理异常: {e}")
            return False

    def _calculate_duration_minutes(self) -> float:
        """计算运行总时长"""
        if self.results["start_time"] and self.results["end_time"]:
            from datetime import datetime as dt
            start = dt.strptime(self.results["start_time"][:19], '%Y-%m-%dT%H:%M:%S')
            end = dt.strptime(self.results["end_time"][:19], '%Y-%m-%dT%H:%M:%S')
            return (end - start).total_seconds() / 60
        return 0

    def run_real_time_monitoring(self):
        """运行实时监控模式"""
        self.logger.info("启动实时异常检测监控系统")
        self.logger.info("系统配置:")
        self.logger.info(f"  - 预热时间: {self.config['real_time_mode']['warmup_minutes']} 分钟")
        self.logger.info(f"  - 检测阈值: {self.config['real_time_mode']['detection_threshold']}")
        self.logger.info(f"  - 检查间隔: {self.config['real_time_mode']['check_interval_seconds']} 秒")
        self.logger.info(f"脚本启动时间: {self.script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("注意: 仅处理脚本启动后生成的新数据")
        
        self.monitoring_active = True
        self.results["start_time"] = datetime.now().isoformat()
        
        # 启动压测
        load_test_process = self.start_load_test()
        if not load_test_process:
            self.logger.warning("压测启动失败，继续监控")
        
        # 启动数据采集
        collection_process = self.start_data_collection(duration_minutes=0)
        
        # 启动指标采集  # 新增
        metrics_process = self.start_metrics_collection(duration_minutes=0)
        if not metrics_process:
            self.logger.warning("指标采集启动失败，但将继续监控")
        
        # 检查关键组件是否成功启动
        if not collection_process:
            self.logger.error("数据采集启动失败，无法继续监控")
            self.monitoring_active = False
            return
        
        # 等待预热期结束
        warmup_minutes = self.config["real_time_mode"]["warmup_minutes"]
        self.logger.info(f"等待预热期 ({warmup_minutes} 分钟)...")
        
        # 故障注入相关状态
        fault_injection_process = None
        fault_injection_attempted = False
        
        # 启动监控循环
        check_interval = self.config["real_time_mode"]["check_interval_seconds"]
        self.logger.info(f"开始实时监控 (检查间隔: {check_interval}s)")
        
        start_time = time.time()
        try:
            while self.monitoring_active:
                elapsed_minutes = (time.time() - start_time) / 60
                
                # 到达预热时间后，启动故障注入（只尝试一次）
                if elapsed_minutes >= warmup_minutes and not fault_injection_attempted:
                    self.logger.info("预热期结束，启动故障注入")
                    fault_injection_attempted = True
                    fault_injection_process = self.start_fault_injection()
                    if fault_injection_process:
                        self.component_status["fault_injection"] = "运行中"
                    else:
                        self.component_status["fault_injection"] = "失败"
                        self.logger.warning("故障注入启动失败")
                
                # 检查新数据
                new_files = self.check_for_new_data()
                
                if new_files:
                    self.process_new_files_real_time(new_files)
                
                # 显示运行状态
                elapsed_hours = elapsed_minutes / 60
                detection_count = len(self.results["real_time_detections"])
                anomaly_count = len([d for d in self.results["real_time_detections"] if d["result"].get("anomaly_detected", False)])
                
                # 计算验证统计
                all_detections = self.results["real_time_detections"]
                validated_detections = [d for d in all_detections if d["validation"].get("validation") == "completed"]
                
                accuracy_stats = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
                for detection in validated_detections:
                    accuracy = detection["validation"].get("accuracy", "unknown")
                    if accuracy in accuracy_stats:
                        accuracy_stats[accuracy] += 1
                
                # 准确率计算
                total_verified = sum(accuracy_stats.values())
                if total_verified > 0:
                    accuracy_rate = ((accuracy_stats["true_positive"] + accuracy_stats["true_negative"]) / total_verified) * 100
                    accuracy_info = f"{accuracy_rate:.1f}%"
                else:
                    accuracy_info = "待验证"
                
                # 组件状态
                components_status = []
                if load_test_process:
                    components_status.append("压测:运行")
                else:
                    components_status.append("压测:失败")
                    
                if collection_process:
                    components_status.append("采集:运行")
                else:
                    components_status.append("采集:失败")
                
                if metrics_process:  # 新增
                    components_status.append("指标:运行")
                else:
                    components_status.append("指标:失败")
                
                if fault_injection_process:
                    components_status.append("故障:运行")
                elif fault_injection_attempted:
                    components_status.append("故障:失败")
                elif elapsed_minutes >= warmup_minutes:
                    components_status.append("故障:启动中")
                else:
                    components_status.append(f"故障:预热中({warmup_minutes - elapsed_minutes:.1f}min)")
                
                # 构建监控状态信息
                status_info = [
                    f"运行时间: {elapsed_hours:.1f}h",
                    f"检测次数: {detection_count}",
                    f"异常次数: {anomaly_count}",
                    f"准确率: {accuracy_info}",
                    f"验证: {len(validated_detections)}/{detection_count}",
                    " | ".join(components_status)
                ]
                
                self.logger.info(f"监控状态: {' | '.join(status_info)}")
                
                # 等待下次检查
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            self.logger.info("用户中断监控")
        except Exception as e:
            self.logger.error(f"监控异常: {e}")
        finally:
            self.monitoring_active = False
            
            # 停止所有后台进程 (添加metrics_process)
            self._cleanup_processes(load_test_process, collection_process, metrics_process, fault_injection_process)
            
            self.results["end_time"] = datetime.now().isoformat()
            
            self.show_final_summary()

    def show_final_summary(self):
        """显示最终摘要（不生成文件）"""
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
        
        self.logger.info("=" * 50)
        self.logger.info("监控完成 - 最终统计")
        self.logger.info("=" * 50)
        self.logger.info(f"总检测次数: {detection_count}")
        self.logger.info(f"检测到异常: {anomaly_count}")
        if total_validated > 0:
            self.logger.info(f"检测准确率: {accuracy_rate:.1f}%")
            self.logger.info(f"  真阳性 (正确检测异常): {accuracy_stats['true_positive']}")
            self.logger.info(f"  真阴性 (正确识别正常): {accuracy_stats['true_negative']}")
            self.logger.info(f"  假阳性 (误报): {accuracy_stats['false_positive']}")
            self.logger.info(f"  假阴性 (漏报): {accuracy_stats['false_negative']}")
        self.logger.info("=" * 50)

    def _cleanup_processes(self, *processes):
        """清理后台进程"""
        process_names = ["压测", "数据采集", "指标采集", "故障注入"]
        
        for i, process in enumerate(processes):
            if process:
                name = process_names[i] if i < len(process_names) else f"进程{i}"
                self.logger.info(f"停止{name}...")
                try:
                    process.terminate()
                    process.wait(timeout=30)
                    self.logger.info(f"{name}已停止")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"{name}未响应，强制结束")
                    try:
                        process.kill()
                        self.logger.info(f"{name}已强制停止")
                    except:
                        self.logger.error(f"无法停止{name}")
                except Exception as e:
                    self.logger.error(f"停止{name}时出错: {e}")

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
            # 计算故障开始时间
            fault_start_minutes = self.config["real_time_mode"]["warmup_minutes"] + i*10
            
            # 故障记录
            for j in range(5):  # 每次故障持续5分钟
                minute_offset = fault_start_minutes + j
                
                # 计算具体时间
                fault_time = current_time + timedelta(minutes=minute_offset)
                
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
                    "timestamp": fault_time.isoformat(),
                    "minute_key": fault_time.strftime("%H_%M"),
                    "date": today,
                    "fault_type": fault_type["type"],
                    "description": fault_type["description"],
                    "intensity": fault_type["intensity"],
                    "expected_anomaly": True
                }
                
                records.append(record)
            
            # 故障间歇期（5分钟）
            for j in range(5):
                minute_offset = fault_start_minutes + j + 5
                fault_time = current_time + timedelta(minutes=minute_offset)
                
                record = {
                    "timestamp": fault_time.isoformat(),
                    "minute_key": fault_time.strftime("%H_%M"),
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
        
        self.logger.info(f"生成故障记录文件: {record_file}")
        self.logger.info(f"记录数量: {len(records)} (包含故障期和正常期)")

    def process_new_files_real_time(self, csv_files: List[str]) -> bool:
        """实时处理新的CSV文件"""
        if not csv_files:
            return False
        
        self.logger.info(f"处理新数据文件: {len(csv_files)} 个")
        
        # 对每个新文件进行图分析和异常检测
        for csv_file in csv_files:
            try:
                # 先进行图分析处理
                self.logger.debug(f"图分析: {os.path.basename(csv_file)}")
                success = self.process_collected_data(csv_file_path=csv_file)
                
                if not success:
                    self.logger.warning(f"图分析失败: {os.path.basename(csv_file)}")
                    continue
                
                # 检查是否生成了处理标志
                flag_file = csv_file.replace('.csv', '.graph_processed')
                if not os.path.exists(flag_file):
                    self.logger.warning(f"图分析标志文件缺失: {os.path.basename(flag_file)}")
                    continue
                
                # 调用异常检测
                detection_result = self.run_anomaly_detection(csv_file)
                
                # 验证结果
                validation = self.validate_detection_result(detection_result)
                
                # 报告检测结果
                self.report_anomaly_detection(csv_file, detection_result, validation)
                
            except Exception as e:
                self.logger.error(f"处理文件失败 {csv_file}: {e}")
        
        return True

    def run_anomaly_detection(self, csv_file: str) -> Dict:
        """调用异常检测脚本（模拟）"""
        self.logger.debug(f"异常检测: {os.path.basename(csv_file)}")
        
        script_path = self.project_root / self.config["scripts"]["anomaly_detection"]
        
        if not os.path.exists(script_path):
            self.logger.debug(f"使用模拟异常检测 (脚本路径: {script_path})")
        
        # 读取文件基本信息
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
            possible_anomalies = ["high_latency", "error_spike", "unusual_pattern", "service_degradation"]
            result["anomaly_types"] = random.sample(possible_anomalies, random.randint(1, 2))
            
        return result

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
            # 没有故障记录 = 期望正常
            expected_anomaly = False
            detected_anomaly = detection_result.get("anomaly_detected", False)
            
            validation_result = {
                "validation": "completed",
                "minute_key": minute_key,
                "expected_anomaly": expected_anomaly,
                "detected_anomaly": detected_anomaly,
                "fault_info": {
                    "type": "normal",
                    "description": "未注入故障，期望正常",
                    "intensity": "none"
                }
            }
        else:
            # 找到故障记录，使用记录中的期望
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
                validation_result["result"] = "true_positive: 正确检测到异常"
            else:
                validation_result["accuracy"] = "true_negative"  # 正确识别正常
                validation_result["result"] = "true_negative: 正确识别正常"
        else:
            if expected_anomaly and not detected_anomaly:
                validation_result["accuracy"] = "false_negative"  # 漏报
                validation_result["result"] = "false_negative: 漏报异常"
            else:
                validation_result["accuracy"] = "false_positive"  # 误报
                validation_result["result"] = "false_positive: 误报异常"
        
        return validation_result

    def load_fault_injection_records(self, target_date: str = None) -> List[Dict]:
        """加载故障注入记录"""
        target_date = target_date or datetime.now().strftime("%Y-%m-%d")
        fault_record_dir = self.project_root / "fault_injection_records"
        record_file = fault_record_dir / f"fault_records_{target_date}.json"
        
        if not record_file.exists():
            self.logger.debug(f"故障记录文件不存在: {record_file}")
            return []
        
        try:
            with open(record_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            self.logger.debug(f"已加载 {len(records)} 条故障记录")
            return records
        except Exception as e:
            self.logger.error(f"加载故障记录失败: {e}")
            return []

    def report_anomaly_detection(self, csv_file: str, detection_result: Dict, validation: Dict):
        """报告异常检测结果"""
        if detection_result.get("anomaly_detected", False):
            self.logger.warning("=" * 60)
            self.logger.warning("ANOMALY DETECTED")
            self.logger.warning("=" * 60)
            self.logger.warning(f"文件: {detection_result['file_name']}")
            self.logger.warning(f"异常分数: {detection_result['anomaly_score']} (阈值: {detection_result['threshold']})")
            self.logger.warning(f"异常类型: {', '.join(detection_result.get('anomaly_types', []))}")
            self.logger.warning(f"Trace数量: {detection_result['trace_count']}")
            self.logger.warning(f"Span数量: {detection_result['span_count']}")
            
            # 显示验证结果
            if validation.get("validation") == "completed":
                self.logger.warning(f"验证结果: {validation.get('result', '未知')}")
                self.logger.warning(f"故障信息: {validation.get('fault_info', {}).get('description', '未知')}")
                self.logger.warning(f"准确性评估: {validation.get('accuracy', '未知')}")
            
            self.logger.warning("=" * 60)
            
        else:
            # 显示正常检测结果的验证
            if validation.get("validation") == "completed":
                status = "CORRECT" if validation.get("accuracy") in ["true_negative", "true_positive"] else "INCORRECT"
                self.logger.info(f"{os.path.basename(csv_file)} - NORMAL ({status})")
            else:
                self.logger.info(f"{os.path.basename(csv_file)} - NORMAL")

        # 保存到内存中的结果列表
        self.results["real_time_detections"].append({
            "timestamp": datetime.now().isoformat(),
            "file": os.path.basename(csv_file),
            "result": detection_result,
            "validation": validation
        })

def main():
    """主函数 - 一键启动"""
    print("=" * 60)
    print("Train Ticket Anomaly Detection System")
    print("=" * 60)
    
    try:
        controller = TrainTicketAnomalyDetectionController()
        controller.run_real_time_monitoring()
        return 0
        
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
        return 0
    except Exception as e:
        print(f"System execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())