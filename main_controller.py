# -*- coding: utf-8 -*-
"""
Train Ticket 异常检测系统主控制器

Author: LoveShikiNatsume
Date: 2025-07-01
Version: 1.6 修改异常检测逻辑，基于CSV标签进行准确性验证
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
import requests  # 添加这行

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
            "metrics_collection": "未开始",
            "anomaly_detection": "未开始",
            "tracevae_service": "未开始"  # 添加服务状态
        }
        
        # 压测统计
        self.load_test_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "last_update": None
        }
        
        # 压测统计
        self.load_test_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "last_update": None
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
        
        # TraceVAE服务相关
        self.tracevae_service_process = None
        self.tracevae_service_url = os.getenv("TRACEVAE_SERVICE_URL", "http://localhost:8000")
        
        self.logger.info("Train Ticket 异常检测系统启动")

    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        return {
            "real_time_mode": {
                "enabled": True,
                "check_interval_seconds": 1,  # 检查新文件的间隔，改为1秒
                "status_display_interval_seconds": 30,  # 状态显示间隔，保持30秒
                "auto_process_delay_seconds": 65,
                "detection_threshold": 0.15,
                "warmup_minutes": 3
            },
            "data_collection": {
                "interval_seconds": 60,
                "lookback_period": "5m"
            },
            "scripts": {
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
        
        run_script = self.project_root / "train-ticket-auto-query" / "run.py"
        
        if not os.path.exists(run_script):
            self.logger.error(f"压测脚本不存在: {run_script}")
            return None
        
        try:
            cmd = [sys.executable, str(run_script)]
            
            # 设置环境变量启用安静模式
            env = os.environ.copy()
            env['PRESSURE_TEST_QUIET'] = 'true'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
                bufsize=1,
                cwd=str(self.project_root),
                env=env
            )
            
            # 启动一个线程来读取压测输出并统计成功请求数
            def read_and_count_stats():
                try:
                    for line in process.stdout:
                        line = line.strip()
                        if "[压测状态]" in line and "完成请求:" in line and "成功率:" in line:
                            try:
                                parts = line.split("完成请求:")[1].split(",")[0].strip()
                                total_requests = int(parts)
                                
                                success_parts = line.split("成功率:")[1].split("%")[0].strip()
                                success_rate = float(success_parts)
                                
                                successful_requests = int(total_requests * success_rate / 100)
                                
                                self.load_test_stats["total_requests"] = total_requests
                                self.load_test_stats["successful_requests"] = successful_requests
                                self.load_test_stats["last_update"] = datetime.now()
                                
                            except (ValueError, IndexError):
                                pass
                except:
                    pass
            
            stats_thread = threading.Thread(target=read_and_count_stats, daemon=True)
            stats_thread.start()
            
            time.sleep(3)
            if process.poll() is None:
                self.logger.info("压测模块启动成功")
                self.component_status["load_test"] = "运行中"
                return process
            else:
                self.logger.error(f"压测启动失败，退出码: {process.returncode}")
                return None
                
        except Exception as e:
            self.logger.error(f"压测启动异常: {e}")
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
                "--duration", str(duration_minutes),
                "--interval", "1"  # 添加1秒间隔参数
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
            # 检查是否已有对应的.label_processed标志文件
            flag_file = str(csv_file).replace('.csv', '.label_processed')
            
            if os.path.exists(flag_file):
                continue
            
            # 解析文件名
            try:
                filename = csv_file.stem
                hour, minute = filename.split('_')
                
                # 构造文件对应的时间
                file_time = current_time.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
                
                # 修改时间过滤逻辑：只跳过早于启动分钟的文件
                # 将启动时间向下取整到分钟级别，包含启动当分钟的文件
                script_start_minute = self.script_start_time.replace(second=0, microsecond=0)
                
                if file_time < script_start_minute:
                    self.logger.debug(f"跳过历史数据文件: {filename} (文件时间: {file_time.strftime('%H:%M')}, 启动分钟: {script_start_minute.strftime('%H:%M')})")
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
        """处理采集的数据（异常标签生成）"""
        processor_script = self.project_root / "train-ticket-trace-collect" / "trace_label_processor.py"
        
        # 检查脚本是否存在
        if not processor_script.exists():
            self.logger.error(f"标签处理脚本不存在: {processor_script}")
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
            
            result = self._run_subprocess(cmd, "label_processing", timeout=300)
            return result is not None and (not hasattr(result, 'returncode') or result.returncode == 0)
        except Exception as e:
            self.logger.error(f"标签处理异常: {e}")
            return False

    def run_real_time_monitoring(self):
        """运行实时监控模式"""
        self.logger.info("启动实时异常检测监控系统")
        self.logger.info("系统配置:")
        self.logger.info(f"  - 检测阈值: {self.config['real_time_mode']['detection_threshold']}")
        self.logger.info(f"  - 检查间隔: {self.config['real_time_mode']['check_interval_seconds']} 秒")
        self.logger.info(f"  - 状态显示间隔: {self.config['real_time_mode']['status_display_interval_seconds']} 秒")
        self.logger.info(f"脚本启动时间: {self.script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.monitoring_active = True
        self.results["start_time"] = datetime.now().isoformat()
        
        # 启动压测
        load_test_process = self.start_load_test()
        if not load_test_process:
            self.logger.warning("压测启动失败，继续监控")
        
        # 启动数据采集
        collection_process = self.start_data_collection(duration_minutes=0)
        
        # 启动指标采集
        metrics_process = self.start_metrics_collection(duration_minutes=0)
        if not metrics_process:
            self.logger.warning("指标采集启动失败，但将继续监控")
        
        # 检查关键组件是否成功启动
        if not collection_process:
            self.logger.error("数据采集启动失败，无法继续监控")
            self.monitoring_active = False
            return
        
        # 启动监控循环
        check_interval = self.config["real_time_mode"]["check_interval_seconds"]
        status_display_interval = self.config["real_time_mode"]["status_display_interval_seconds"]
        self.logger.info(f"开始实时监控 (检查间隔: {check_interval}s, 状态显示间隔: {status_display_interval}s)")
        
        start_time = time.time()
        last_status_display = 0  # 记录上次状态显示时间
        
        try:
            while self.monitoring_active:
                current_time = time.time()
                elapsed_minutes = (current_time - start_time) / 60
                
                # 检查新数据
                new_files = self.check_for_new_data()
                
                if new_files:
                    self.process_new_files_real_time(new_files)
                
                # 只在达到状态显示间隔时才显示状态
                if current_time - last_status_display >= status_display_interval:
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
                    
                    # 组件状态和压测统计
                    components_status = []
                    
                    if load_test_process and load_test_process.poll() is None:
                        # 压测运行中，显示统计信息
                        total_req = self.load_test_stats["total_requests"]
                        success_req = self.load_test_stats["successful_requests"]
                        if total_req > 0:
                            success_rate = (success_req / total_req) * 100
                            components_status.append(f"压测:运行({success_req}/{total_req}, {success_rate:.1f}%)")
                        else:
                            components_status.append("压测:运行(统计中...)")
                    else:
                        components_status.append("压测:失败")
                        
                    if collection_process and collection_process.poll() is None:
                        components_status.append("采集:运行")
                    else:
                        components_status.append("采集:失败")
                
                    if metrics_process and metrics_process.poll() is None:
                        components_status.append("指标:运行")
                    else:
                        components_status.append("指标:失败")
                    
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
                    last_status_display = current_time
                
                # 等待下次检查（短间隔）
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            self.logger.info("用户中断监控")
        except Exception as e:
            self.logger.error(f"监控异常: {e}")
        finally:
            self.monitoring_active = False
            
            # 停止所有后台进程
            self._cleanup_processes(load_test_process, collection_process, metrics_process)
            
            self.results["end_time"] = datetime.now().isoformat()
            
            self.show_final_summary()

    def show_final_summary(self):
        """显示最终摘要"""
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
        
        # 显示压测最终统计
        total_req = self.load_test_stats["total_requests"]
        success_req = self.load_test_stats["successful_requests"]
        if total_req > 0:
            success_rate = (success_req / total_req) * 100
            self.logger.info(f"压测最终统计: {success_req}/{total_req} 成功率: {success_rate:.1f}%")
        
        self.logger.info("=" * 50)

    def _cleanup_processes(self, *processes):
        """清理后台进程"""
        process_names = ["压测", "数据采集", "指标采集"]
        
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

    def process_new_files_real_time(self, csv_files: List[str]) -> bool:
        """实时处理新的CSV文件"""
        if not csv_files:
            return False
        
        self.logger.info(f"处理新数据文件: {len(csv_files)} 个")
        
        # 对每个新文件进行标签生成和异常检测
        for csv_file in csv_files:
            try:
                # 先进行标签生成处理
                self.logger.debug(f"标签生成: {os.path.basename(csv_file)}")
                success = self.process_collected_data(csv_file_path=csv_file)
                
                if not success:
                    self.logger.warning(f"标签生成失败: {os.path.basename(csv_file)}")
                    continue
                
                # 检查是否生成了处理标志
                flag_file = csv_file.replace('.csv', '.label_processed')
                if not os.path.exists(flag_file):
                    self.logger.warning(f"标签处理标志文件缺失: {os.path.basename(flag_file)}")
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

    def get_expected_anomaly_from_csv(self, csv_file: str) -> Dict:
        """从CSV文件的标签列读取期望的异常结果"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            # 检查是否为14列格式（包含标签）
            if len(df.columns) != 14:
                return {
                    "has_labels": False,
                    "reason": f"CSV文件列数不匹配，期望14列，实际{len(df.columns)}列"
                }
            
            # 检查标签列是否存在
            label_columns = ['nodeLatencyLabel', 'graphLatencyLabel', 'graphStructureLabel']
            missing_columns = [col for col in label_columns if col not in df.columns]
            
            if missing_columns:
                return {
                    "has_labels": False,
                    "reason": f"缺少标签列: {missing_columns}"
                }
            
            # 统计标签分布
            node_labels = df['nodeLatencyLabel'].value_counts().to_dict()
            graph_labels = df['graphLatencyLabel'].value_counts().to_dict()
            
            # 判断是否期望异常：任一标签有异常值(1)则期望异常
            has_node_anomaly = node_labels.get(1, 0) > 0
            has_graph_anomaly = graph_labels.get(1, 0) > 0
            expected_anomaly = has_node_anomaly or has_graph_anomaly
            
            return {
                "has_labels": True,
                "expected_anomaly": expected_anomaly,
                "total_spans": len(df),
                "unique_traces": len(df['traceIdLow'].unique()) if 'traceIdLow' in df.columns else 0,
                "label_distribution": {
                    "nodeLatencyLabel": node_labels,
                    "graphLatencyLabel": graph_labels,
                    "anomaly_spans": node_labels.get(1, 0) + graph_labels.get(1, 0)
                },
                "anomaly_reason": "标签显示异常" if expected_anomaly else "标签显示正常"
            }
            
        except Exception as e:
            return {
                "has_labels": False,
                "reason": f"读取CSV文件失败: {e}"
            }

    # def call_anomaly_detection_model(self, csv_file: str) -> Dict:
    #     """调用异常检测模型（当前为模拟实现）"""
    #     self.logger.debug(f"异常检测: {os.path.basename(csv_file)}")
        
    #     # TODO: 这里可以替换为真实的异常检测模型调用
    #     # 例如：result = subprocess.run([python, model_script, csv_file])
        
    #     script_path = self.project_root / self.config["scripts"]["anomaly_detection"]
        
    #     if not os.path.exists(script_path):
    #         self.logger.debug(f"使用模拟异常检测 (真实模型路径: {script_path})")
        
    #     # 读取文件基本信息用于模拟
    #     try:
    #         import pandas as pd
    #         df = pd.read_csv(csv_file)
    #         trace_count = len(df['traceIdLow'].unique()) if 'traceIdLow' in df.columns else 0
    #         span_count = len(df)
    #     except Exception as e:
    #         self.logger.error(f"读取CSV文件失败: {e}")
    #         trace_count = 0
    #         span_count = 0
        
    #     # 模拟异常检测结果
    #     import random
    #     anomaly_score = random.uniform(0.0, 1.0)
    #     threshold = self.config["real_time_mode"]["detection_threshold"]
    #     is_anomaly = anomaly_score > threshold
        
    #     result = {
    #         "file_name": os.path.basename(csv_file),
    #         "analysis_time": datetime.now().isoformat(),
    #         "trace_count": trace_count, 
    #         "span_count": span_count,
    #         "anomaly_score": round(anomaly_score, 4),
    #         "threshold": threshold,
    #         "anomaly_detected": is_anomaly,
    #         "model_confidence": round(abs(anomaly_score - threshold), 4),
    #         "anomaly_types": []
    #     }
        
    #     if is_anomaly:
    #         possible_anomalies = ["high_latency", "error_spike", "unusual_pattern", "service_degradation"]
    #         result["anomaly_types"] = random.sample(possible_anomalies, random.randint(1, 2))
            
    #     return result

    def call_anomaly_detection_model(self, csv_file: str) -> Dict:
        """调用TraceVAE异常检测模型"""
        self.logger.debug(f"异常检测: {os.path.basename(csv_file)}")
        
        try:
            # 使用TraceVAE在线检测
            result = self._call_tracevae_online_detection(csv_file)
            if result:
                return result
        except Exception as e:
            self.logger.warning(f"TraceVAE在线检测失败，使用降级模式: {e}")
        
        # 降级到原有的模拟检测
        return self._fallback_simulation_detection(csv_file)

    def _call_tracevae_online_detection(self, csv_file: str) -> Dict:
        """调用TraceVAE在线检测服务"""
        import asyncio
        import sys
        
        # 添加csv_file_monitor.py所在路径（假设在同一目录或子目录）
        monitor_path = str(self.project_root)  # 或者具体的路径
        if monitor_path not in sys.path:
            sys.path.append(monitor_path)
        
        # 导入CSV监控器
        from csv_file_monitor import CSVFileMonitor
        
        # TraceVAE服务URL（可以通过环境变量配置）
        tracevae_url = os.getenv("TRACEVAE_SERVICE_URL", "http://localhost:8000")
        
        # 创建监控器
        monitor = CSVFileMonitor(tracevae_url)
        
        # 异步调用检测
        async def detect():
            return await monitor.process_entire_csv_file(csv_file, batch_size=20)
        
        # 运行检测
        tracevae_result = asyncio.run(detect())
        
        # 转换为Train Ticket期望的格式
        return self._convert_tracevae_to_train_ticket_format(tracevae_result, csv_file)

    def _convert_tracevae_to_train_ticket_format(self, tracevae_result: Dict, csv_file: str) -> Dict:
        """将TraceVAE结果转换为Train Ticket期望的格式"""
        try:
            file_name = os.path.basename(csv_file)
            
            # 提取TraceVAE结果
            status = tracevae_result.get('status', 'NORMAL')
            total_traces = tracevae_result.get('total_traces', 0)
            anomaly_traces = tracevae_result.get('anomaly_traces', 0)
            anomaly_percentage = tracevae_result.get('anomaly_percentage', 0.0)
            
            # 判断是否异常
            is_anomaly = status == 'ANOMALY'
            
            # 计算异常分数 (0.0-1.0)
            anomaly_score = anomaly_percentage / 100.0 if total_traces > 0 else 0.0
            
            # 获取阈值
            threshold = self.config["real_time_mode"]["detection_threshold"]
            
            # 计算置信度
            confidence = abs(anomaly_score - threshold)
            
            # 构建异常类型
            anomaly_types = []
            if is_anomaly:
                if anomaly_percentage > 80:
                    anomaly_types.append("high_anomaly_rate")
                elif anomaly_percentage > 50:
                    anomaly_types.append("moderate_anomaly_rate")
                else:
                    anomaly_types.append("low_anomaly_rate")
                
                # 添加TraceVAE特定的异常类型
                anomaly_details = tracevae_result.get('anomaly_types', {})
                for anomaly_type in anomaly_details.keys():
                    anomaly_types.append(f"tracevae_{anomaly_type}")
            
            return {
                "file_name": file_name,
                "analysis_time": datetime.now().isoformat(),
                "trace_count": total_traces,
                "span_count": total_traces,  # 简化处理
                "anomaly_score": round(anomaly_score, 4),
                "threshold": threshold,
                "anomaly_detected": is_anomaly,
                "model_confidence": round(confidence, 4),
                "anomaly_types": anomaly_types,
                "detection_method": "tracevae_online",
                "tracevae_details": {
                    "status": status,
                    "total_traces": total_traces,
                    "anomaly_traces": anomaly_traces,
                    "normal_traces": tracevae_result.get('normal_traces', 0),
                    "anomaly_percentage": anomaly_percentage,
                    "anomaly_types": tracevae_result.get('anomaly_types', {}),
                    "detection_timestamp": tracevae_result.get('detection_timestamp')
                }
            }
            
        except Exception as e:
            self.logger.error(f"TraceVAE结果转换失败: {e}")
            raise Exception(f"结果转换失败: {e}")

    def _fallback_simulation_detection(self, csv_file: str) -> Dict:
        """降级到原有的模拟检测"""
        self.logger.info(f"使用模拟检测: {os.path.basename(csv_file)}")
        
        # 这里是原来的模拟检测逻辑
        script_path = self.project_root / self.config["scripts"]["anomaly_detection"]
        
        if not os.path.exists(script_path):
            self.logger.debug(f"使用模拟异常检测 (真实模型路径: {script_path})")
        
        # 读取文件基本信息用于模拟
        try:
            import pandas as pd
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
            "model_confidence": round(abs(anomaly_score - threshold), 4),
            "anomaly_types": [],
            "detection_method": "simulation"
        }
        
        if is_anomaly:
            possible_anomalies = ["high_latency", "error_spike", "unusual_pattern", "service_degradation"]
            result["anomaly_types"] = random.sample(possible_anomalies, random.randint(1, 2))
            
        return result



    def validate_detection_accuracy(self, expected_anomaly: bool, detected_anomaly: bool, 
                                   label_info: Dict, detection_result: Dict) -> Dict:
        """验证检测准确性"""
        # 计算准确性类别
        if expected_anomaly and detected_anomaly:
            accuracy = "true_positive"
            result_desc = "正确检测到异常"
        elif not expected_anomaly and not detected_anomaly:
            accuracy = "true_negative"
            result_desc = "正确识别正常"
        elif not expected_anomaly and detected_anomaly:
            accuracy = "false_positive"
            result_desc = "误报异常"
        else:  # expected_anomaly and not detected_anomaly
            accuracy = "false_negative"
            result_desc = "漏报异常"
        
        validation_result = {
            "validation": "completed",
            "expected_anomaly": expected_anomaly,
            "detected_anomaly": detected_anomaly,
            "accuracy": accuracy,
            "result": f"{accuracy}: {result_desc}",
            "label_info": {
                "total_spans": label_info.get("total_spans", 0),
                "unique_traces": label_info.get("unique_traces", 0),
                "anomaly_spans": label_info.get("label_distribution", {}).get("anomaly_spans", 0),
                "anomaly_reason": label_info.get("anomaly_reason", "未知")
            },
            "model_info": {
                "anomaly_score": detection_result.get("anomaly_score", 0),
                "threshold": detection_result.get("threshold", 0),
                "confidence": detection_result.get("model_confidence", 0)
            }
        }
        
        return validation_result

    def run_anomaly_detection(self, csv_file: str) -> Dict:
        """运行异常检测流程"""
        # 1. 读取CSV标签获取期望结果
        label_info = self.get_expected_anomaly_from_csv(csv_file)
        
        if not label_info.get("has_labels", False):
            return {
                "file_name": os.path.basename(csv_file),
                "error": "无法读取标签",
                "reason": label_info.get("reason", "未知错误"),
                "anomaly_detected": False
            }
        
        # 2. 调用异常检测模型
        detection_result = self.call_anomaly_detection_model(csv_file)
        
        # 3. 验证检测准确性
        expected_anomaly = label_info.get("expected_anomaly", False)
        detected_anomaly = detection_result.get("anomaly_detected", False)
        
        validation = self.validate_detection_accuracy(
            expected_anomaly, detected_anomaly, label_info, detection_result
        )
        
        # 4. 合并结果
        detection_result["validation"] = validation
        
        return detection_result

    def validate_detection_result(self, detection_result: Dict) -> Dict:
        """提取验证结果（兼容原有接口）"""
        return detection_result.get("validation", {
            "validation": "failed",
            "reason": "无验证信息"
        })

    def report_anomaly_detection(self, csv_file: str, detection_result: Dict, validation: Dict):
        """报告异常检测结果"""
        file_name = os.path.basename(csv_file)
        
        # 检查是否有错误
        if "error" in detection_result:
            self.logger.error(f"检测失败 {file_name}: {detection_result.get('reason', '未知错误')}")
            return
        
        detected_anomaly = detection_result.get("anomaly_detected", False)
        expected_anomaly = validation.get("expected_anomaly", False)
        accuracy = validation.get("accuracy", "unknown")
        
        # 构建结果信息
        model_info = validation.get("model_info", {})
        label_info = validation.get("label_info", {})
        
        anomaly_score = model_info.get("anomaly_score", 0)
        threshold = model_info.get("threshold", 0)
        confidence = model_info.get("confidence", 0)
        
        total_spans = label_info.get("total_spans", 0)
        unique_traces = label_info.get("unique_traces", 0)
        anomaly_spans = label_info.get("anomaly_spans", 0)
        
        if detected_anomaly:
            # 检测到异常
            status_icon = "✓" if accuracy in ["true_positive"] else "✗"
            self.logger.warning("=" * 60)
            self.logger.warning(f"异常检测 {status_icon} {file_name}")
            self.logger.warning("=" * 60)
            self.logger.warning(f"模型结果: 异常 (分数: {anomaly_score}, 阈值: {threshold}, 置信度: {confidence})")
            self.logger.warning(f"标签期望: {'异常' if expected_anomaly else '正常'} (异常span: {anomaly_spans}/{total_spans})")
            self.logger.warning(f"准确性评估: {validation.get('result', '未知')}")
            self.logger.warning(f"数据统计: {unique_traces} traces, {total_spans} spans")
            
            if detection_result.get("anomaly_types"):
                self.logger.warning(f"异常类型: {', '.join(detection_result['anomaly_types'])}")
            
            self.logger.warning("=" * 60)
        else:
            # 检测为正常
            status_icon = "✓" if accuracy in ["true_negative"] else "✗"
            status = "CORRECT" if accuracy in ["true_negative", "true_positive"] else "INCORRECT"
            
            self.logger.info(f"正常检测 {status_icon} {file_name} - {status}")
            self.logger.info(f"  模型: 正常 (分数: {anomaly_score}) | 标签: {'异常' if expected_anomaly else '正常'} | {validation.get('result', '未知')}")
        
        # 保存到内存中的结果列表
        self.results["real_time_detections"].append({
            "timestamp": datetime.now().isoformat(),
            "file": file_name,
            "result": detection_result,
            "validation": validation
        })

    def process_detection_json_results(self, target_date: str = None):
        """处理检测结果JSON文件并输出到控制台"""
        target_date = target_date or datetime.now().strftime("%Y-%m-%d")
        
        # 查找JSON结果文件
        project_dir = self.project_root
        json_files = list(project_dir.glob("*.json"))
        
        # 过滤出检测结果文件（根据文件名模式）
        detection_json_files = []
        for json_file in json_files:
            # 假设检测结果文件名格式为 时间.json (如 19_09.json)
            if json_file.stem.count('_') == 1 and json_file.stem.replace('_', '').isdigit():
                detection_json_files.append(json_file)
        
        if not detection_json_files:
            self.logger.info("未找到检测结果JSON文件")
            return
        
        # 按文件名排序
        detection_json_files.sort()
        
        self.logger.info(f"找到 {len(detection_json_files)} 个检测结果文件")
        self.logger.info("=" * 80)
        
        total_files = 0
        anomaly_files = 0
        normal_files = 0
        
        for json_file in detection_json_files:
            try:
                result_data = self.read_detection_json(json_file)
                if result_data:
                    self.display_detection_result(json_file.name, result_data)
                    
                    total_files += 1
                    if result_data.get('status') == 'ANOMALY':
                        anomaly_files += 1
                    else:
                        normal_files += 1
                        
            except Exception as e:
                self.logger.error(f"处理文件 {json_file} 失败: {e}")
        
        # 输出汇总统计
        self.display_summary_statistics(total_files, anomaly_files, normal_files)

    def read_detection_json(self, json_file_path: Path) -> Dict:
        """读取检测结果JSON文件"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"读取JSON文件失败 {json_file_path}: {e}")
            return {}

    def display_detection_result(self, filename: str, result_data: Dict):
        """显示单个检测结果"""
        status = result_data.get('status', 'UNKNOWN')
        csv_file = result_data.get('csv_file', filename.replace('.json', '.csv'))
        total_traces = result_data.get('total_traces', 0)
        anomaly_traces = result_data.get('anomaly_traces', 0)
        normal_traces = result_data.get('normal_traces', 0)
        anomaly_percentage = result_data.get('anomaly_percentage', 0.0)
        anomaly_threshold = result_data.get('anomaly_threshold', 0.2)
        detection_timestamp = result_data.get('detection_timestamp', 'Unknown')
        
        # 状态图标和颜色
        if status == 'ANOMALY':
            status_icon = "🚨"
            status_color = "异常"
        else:
            status_icon = "✅"
            status_color = "正常"
        
        # 显示检测结果
        print(f"\n{status_icon} 文件: {csv_file}")
        print(f"📋 检测状态: {status_color}")
        print(f"📊 Traces统计: 总计 {total_traces} | 正常 {normal_traces} | 异常 {anomaly_traces}")
        print(f"📈 异常比例: {anomaly_percentage:.2f}% (阈值: {anomaly_threshold}%)")
        print(f"🕒 检测时间: {detection_timestamp}")
        
        # 如果有异常，显示异常类型
        if status == 'ANOMALY':
            anomaly_types = result_data.get('anomaly_types', {})
            if anomaly_types:
                print(f"🏷️  异常类型:")
                for anomaly_type, count in anomaly_types.items():
                    print(f"   - {anomaly_type}: {count} 次")
            
            # 显示异常详情示例
            anomaly_details = result_data.get('anomaly_details', [])
            if anomaly_details:
                print(f"🔍 异常示例:")
                for i, detail in enumerate(anomaly_details[:3]):  # 只显示前3个
                    trace_id = detail.get('traceID', 'Unknown')
                    anomaly_type = detail.get('anomaly_type', 'Unknown')
                    confidence = detail.get('confidence', 0)
                    print(f"   {i+1}. {trace_id}: {anomaly_type} (置信度: {confidence:.3f})")
        
        print("-" * 60)

    def display_summary_statistics(self, total_files: int, anomaly_files: int, normal_files: int):
        """显示汇总统计"""
        print("\n" + "=" * 80)
        print("📊 检测结果汇总统计")
        print("=" * 80)
        print(f"📁 总文件数: {total_files}")
        print(f"🚨 异常文件: {anomaly_files}")
        print(f"✅ 正常文件: {normal_files}")
        
        if total_files > 0:
            anomaly_rate = (anomaly_files / total_files) * 100
            print(f"📈 异常文件比例: {anomaly_rate:.1f}%")
        
        print("=" * 80)

    def run_json_analysis_mode(self):
        """运行JSON分析模式"""
        self.logger.info("启动JSON检测结果分析模式")
        
        try:
            self.process_detection_json_results()
        except Exception as e:
            self.logger.error(f"JSON分析失败: {e}")

    def check_tracevae_service(self) -> bool:
        """检查TraceVAE在线检测服务是否运行"""
        try:
            response = requests.get(f"{self.tracevae_service_url}/health", timeout=5)
            if response.status_code == 200:
                self.logger.info("TraceVAE在线检测服务运行正常")
                return True
            else:
                self.logger.warning(f"TraceVAE服务响应异常: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"无法连接到TraceVAE服务: {e}")
            return False

    def start_tracevae_service(self) -> bool:
        """启动TraceVAE在线检测服务"""
        tracevae_dir = self.project_root / "tracevae" / "online_detector"
        run_script = tracevae_dir / "run.py"
        
        if not tracevae_dir.exists():
            self.logger.error(f"TraceVAE目录不存在: {tracevae_dir}")
            return False
        
        if not run_script.exists():
            self.logger.error(f"TraceVAE启动脚本不存在: {run_script}")
            return False
        
        try:
            self.logger.info("启动TraceVAE在线检测服务...")
            
            # 启动服务
            self.tracevae_service_process = subprocess.Popen(
                [sys.executable, "run.py"],
                cwd=str(tracevae_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 等待服务启动
            max_wait_time = 30  # 最多等待30秒
            wait_interval = 2   # 每2秒检查一次
            
            for i in range(max_wait_time // wait_interval):
                time.sleep(wait_interval)
                
                # 检查进程是否还在运行
                if self.tracevae_service_process.poll() is not None:
                    # 进程已退出，读取错误信息
                    _, stderr = self.tracevae_service_process.communicate()
                    self.logger.error(f"TraceVAE服务启动失败: {stderr}")
                    return False
                
                # 检查服务是否可用
                if self.check_tracevae_service():
                    self.logger.info("TraceVAE在线检测服务启动成功")
                    self.component_status["tracevae_service"] = "运行中"
                    return True
                
                self.logger.info(f"等待TraceVAE服务启动... ({i+1}/{max_wait_time//wait_interval})")
            
            # 超时未启动成功
            self.logger.error("TraceVAE服务启动超时")
            self.stop_tracevae_service()
            return False
            
        except Exception as e:
            self.logger.error(f"启动TraceVAE服务异常: {e}")
            return False

    def stop_tracevae_service(self):
        """停止TraceVAE在线检测服务"""
        if self.tracevae_service_process:
            try:
                self.logger.info("停止TraceVAE在线检测服务...")
                self.tracevae_service_process.terminate()
                self.tracevae_service_process.wait(timeout=10)
                self.logger.info("TraceVAE服务已停止")
            except subprocess.TimeoutExpired:
                self.logger.warning("TraceVAE服务未响应，强制结束")
                self.tracevae_service_process.kill()
            except Exception as e:
                self.logger.error(f"停止TraceVAE服务时出错: {e}")
            finally:
                self.tracevae_service_process = None
                self.component_status["tracevae_service"] = "已停止"

    def ensure_tracevae_service_running(self) -> bool:
        """确保TraceVAE服务正在运行"""
        # 检查服务是否运行
        if self.check_tracevae_service():
            return True
        
        # 服务未运行，尝试启动
        self.logger.info("TraceVAE服务未运行，正在启动...")
        return self.start_tracevae_service()

    def _call_tracevae_online_detection(self, csv_file: str) -> Dict:
        """调用TraceVAE在线检测服务"""
        # 确保服务正在运行
        if not self.ensure_tracevae_service_running():
            raise Exception("TraceVAE在线检测服务不可用")
        
        import asyncio
        import sys
        
        # 添加csv_file_monitor.py所在路径
        monitor_path = str(self.project_root)
        if monitor_path not in sys.path:
            sys.path.append(monitor_path)
        
        # 导入CSV监控器
        from csv_file_monitor import CSVFileMonitor
        
        # 创建监控器
        monitor = CSVFileMonitor(self.tracevae_service_url)
        
        # 异步调用检测
        async def detect():
            return await monitor.process_entire_csv_file(csv_file, batch_size=20)
        
        # 运行检测
        tracevae_result = asyncio.run(detect())
        
        # 转换为Train Ticket期望的格式
        return self._convert_tracevae_to_train_ticket_format(tracevae_result, csv_file)

    def run_real_time_monitoring(self):
        """运行实时监控模式"""
        self.logger.info("启动实时异常检测监控系统")
        self.logger.info("系统配置:")
        self.logger.info(f"  - 检测阈值: {self.config['real_time_mode']['detection_threshold']}")
        self.logger.info(f"  - 检查间隔: {self.config['real_time_mode']['check_interval_seconds']} 秒")
        self.logger.info(f"  - 状态显示间隔: {self.config['real_time_mode']['status_display_interval_seconds']} 秒")
        self.logger.info(f"脚本启动时间: {self.script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.monitoring_active = True
        self.results["start_time"] = datetime.now().isoformat()
        
        # 首先确保TraceVAE服务运行
        self.logger.info("检查TraceVAE在线检测服务...")
        if not self.ensure_tracevae_service_running():
            self.logger.error("TraceVAE服务启动失败，无法进行异常检测")
            self.monitoring_active = False
            return
        
        # 启动压测
        load_test_process = self.start_load_test()
        if not load_test_process:
            self.logger.warning("压测启动失败，继续监控")
        
        # 启动数据采集
        collection_process = self.start_data_collection(duration_minutes=0)
        
        # 启动指标采集
        metrics_process = self.start_metrics_collection(duration_minutes=0)
        if not metrics_process:
            self.logger.warning("指标采集启动失败，但将继续监控")
        
        # 检查关键组件是否成功启动
        if not collection_process:
            self.logger.error("数据采集启动失败，无法继续监控")
            self.monitoring_active = False
            return
        
        # 启动监控循环
        check_interval = self.config["real_time_mode"]["check_interval_seconds"]
        status_display_interval = self.config["real_time_mode"]["status_display_interval_seconds"]
        self.logger.info(f"开始实时监控 (检查间隔: {check_interval}s, 状态显示间隔: {status_display_interval}s)")
        
        start_time = time.time()
        last_status_display = 0
        
        try:
            while self.monitoring_active:
                current_time = time.time()
                elapsed_minutes = (current_time - start_time) / 60
                
                # 检查新数据
                new_files = self.check_for_new_data()
                
                if new_files:
                    self.process_new_files_real_time(new_files)
                
                # 只在达到状态显示间隔时才显示状态
                if current_time - last_status_display >= status_display_interval:
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
                    
                    # 组件状态和压测统计
                    components_status = []
                    
                    # TraceVAE服务状态
                    if self.check_tracevae_service():
                        components_status.append("检测服务:运行")
                    else:
                        components_status.append("检测服务:异常")
                    
                    if load_test_process and load_test_process.poll() is None:
                        # 压测运行中，显示统计信息
                        total_req = self.load_test_stats["total_requests"]
                        success_req = self.load_test_stats["successful_requests"]
                        if total_req > 0:
                            success_rate = (success_req / total_req) * 100
                            components_status.append(f"压测:运行({success_req}/{total_req}, {success_rate:.1f}%)")
                        else:
                            components_status.append("压测:运行(统计中...)")
                    else:
                        components_status.append("压测:失败")
                        
                    if collection_process and collection_process.poll() is None:
                        components_status.append("采集:运行")
                    else:
                        components_status.append("采集:失败")
                
                    if metrics_process and metrics_process.poll() is None:
                        components_status.append("指标:运行")
                    else:
                        components_status.append("指标:失败")
                    
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
                    last_status_display = current_time
                
                # 等待下次检查（短间隔）
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            self.logger.info("用户中断监控")
        except Exception as e:
            self.logger.error(f"监控异常: {e}")
        finally:
            self.monitoring_active = False
            
            # 停止所有后台进程
            self._cleanup_processes(load_test_process, collection_process, metrics_process)
            
            # 停止TraceVAE服务
            self.stop_tracevae_service()
            
            self.results["end_time"] = datetime.now().isoformat()
            
            self.show_final_summary()

    def _cleanup_processes(self, *processes):
        """清理后台进程"""
        process_names = ["压测", "数据采集", "指标采集"]
        
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


def main():
    """主函数 - 一键启动"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Ticket 异常检测系统")
    parser.add_argument("--mode", choices=["monitor", "analyze"], default="monitor",
                       help="运行模式: monitor=实时监控, analyze=分析JSON结果")
    parser.add_argument("--date", type=str, help="指定分析日期 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Train Ticket Anomaly Detection System")
    print("=" * 60)
    
    try:
        controller = TrainTicketAnomalyDetectionController()
        
        if args.mode == "monitor":
            # 原有的实时监控模式
            controller.run_real_time_monitoring()
        elif args.mode == "analyze":
            # 新的JSON分析模式
            controller.run_json_analysis_mode()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
        return 0
    except Exception as e:
        print(f"System execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

