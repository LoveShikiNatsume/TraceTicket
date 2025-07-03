# -*- coding: utf-8 -*-
"""
Train Ticket Prometheus Metrics Collector
采集系统性能指标用于异常检测分析

Author: LoveShikiNatsume
Date: 2025-06-18
"""

import requests
import json
import time
import logging
import os
import csv
from datetime import datetime
from typing import List, Dict, Optional
from config import Config

class PrometheusMetricsCollector:
    """普罗米修斯指标采集器"""
    
    def __init__(self):
        self.config = Config()
        self.prometheus_url = f"http://{self.config.PROMETHEUS_HOST}:{self.config.PROMETHEUS_PORT}"
        
        # 修改基础输出目录 - metrics与trace同级
        self.base_output_dir = os.path.dirname(self.config.ensure_output_dir())  # 获取trace的父目录
        
        self.logger = self._setup_logging()
        self.session = requests.Session()
        self.session.timeout = self.config.REQUEST_TIMEOUT
        
        # 初始的默认指标配置（将在test_connection中被智能配置覆盖）
        self.key_metrics = {
            'cpu_usage_rate': 'rate(container_cpu_usage_seconds_total[1m])',
            'memory_usage_bytes': 'container_memory_usage_bytes',
            'network_latency_seconds': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))',
            'request_success_rate': '(rate(http_requests_total[1m]) - rate(http_requests_total{status=~"5.."}[1m])) / rate(http_requests_total[1m])'
        }
        
        self.collected_metrics = []
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_samples": 0,
            "failed_queries": 0
        }
        
        self.logger.info("普罗米修斯指标采集器初始化完成")
        self.logger.info(f"Prometheus URL: {self.prometheus_url}")
        self.logger.info(f"输出目录: {self.base_output_dir}")

    def _get_current_date_dirs(self):
        """获取当前日期的目录路径 - metrics与trace同级"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_dir = os.path.join(self.base_output_dir, "metrics", today)
        csv_dir = os.path.join(today_dir, "csv")
        
        # 确保目录存在
        os.makedirs(csv_dir, exist_ok=True)
        
        return today, today_dir, csv_dir

    def _setup_logging(self):
        """设置日志"""
        logger = logging.getLogger('MetricsCollector')
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def test_connection(self) -> bool:
        """测试与Prometheus的连接并智能检测可用指标"""
        try:
            response = self.session.get(f"{self.prometheus_url}/api/v1/label/__name__/values")
            if response.status_code == 200:
                data = response.json()
                available_metrics = data.get('data', [])
                metric_count = len(available_metrics)
                self.logger.info(f"连接成功，发现指标: {metric_count} 个")
                
                # 智能检测并配置指标查询
                detected_metrics = self._detect_available_metrics(available_metrics)
                
                if detected_metrics:
                    self.key_metrics = detected_metrics
                    self.logger.info(f"成功配置 {len(detected_metrics)} 个核心指标")
                    for metric_name, query in detected_metrics.items():
                        self.logger.info(f"  {metric_name}: {query}")
                    return True
                else:
                    self.logger.warning("未找到可用的系统指标")
                    return False
            else:
                self.logger.error(f"连接失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            return False

    def _detect_available_metrics(self, available_metrics: List[str]) -> Dict[str, str]:
        """基于可用指标智能检测和配置查询"""
        detected = {}
        
        # CPU使用率检测
        cpu_metrics = [
            'container_cpu_usage_seconds_total',
            'process_cpu_seconds_total', 
            'node_cpu_seconds_total',
            'cpu_usage_seconds_total'
        ]
        for metric in cpu_metrics:
            if any(metric in m for m in available_metrics):
                detected['cpu_usage_rate'] = f'rate({metric}[1m])'
                self.logger.debug(f"检测到CPU指标: {metric}")
                break
        
        # 内存使用量检测
        memory_metrics = [
            'container_memory_usage_bytes',
            'process_resident_memory_bytes',
            'node_memory_MemAvailable_bytes',
            'go_memstats_alloc_bytes'
        ]
        for metric in memory_metrics:
            if any(metric in m for m in available_metrics):
                detected['memory_usage_bytes'] = metric
                self.logger.debug(f"检测到内存指标: {metric}")
                break
        
        # 网络延迟检测 (HTTP响应时间)
        latency_metrics = [
            'http_request_duration_seconds',
            'request_duration_seconds',
            'response_time_seconds',
            'latency_seconds'
        ]
        for metric in latency_metrics:
            if any(metric in m for m in available_metrics):
                # 查找对应的bucket指标
                bucket_metric = f"{metric}_bucket"
                if any(bucket_metric in m for m in available_metrics):
                    detected['network_latency_seconds'] = f'histogram_quantile(0.95, rate({bucket_metric}[1m]))'
                    self.logger.debug(f"检测到延迟指标: {bucket_metric}")
                    break
        
        # 请求成功率检测
        request_metrics = [
            'http_requests_total',
            'requests_total',
            'http_request_total'
        ]
        for metric in request_metrics:
            if any(metric in m for m in available_metrics):
                # 构建成功率查询：总请求 - 5xx错误 / 总请求
                success_query = f'(rate({metric}[1m]) - rate({metric}{{status=~"5.."}}[1m])) / rate({metric}[1m])'
                detected['request_success_rate'] = success_query
                self.logger.debug(f"检测到请求指标: {metric}")
                break
        
        # 如果没有找到HTTP请求指标，尝试使用up指标作为服务可用性
        if 'request_success_rate' not in detected:
            if any('up' in m for m in available_metrics):
                detected['service_availability'] = 'up'
                self.logger.debug("使用up指标作为服务可用性")
        
        # 补充基础指标：如果核心指标不足，添加一些通用指标
        if len(detected) < 3:
            # 添加Prometheus自身指标作为备选
            fallback_metrics = {
                'prometheus_notifications_total': 'rate(prometheus_notifications_total[1m])',
                'prometheus_tsdb_symbol_table_size_bytes': 'prometheus_tsdb_symbol_table_size_bytes',
                'go_goroutines': 'go_goroutines'
            }
            
            for fallback_name, fallback_query in fallback_metrics.items():
                if any(fallback_name in m for m in available_metrics):
                    detected[f'fallback_{fallback_name}'] = fallback_query
                    self.logger.debug(f"添加备选指标: {fallback_name}")
                    if len(detected) >= 4:  # 确保至少有4个指标
                        break
        
        return detected

    def collect_metrics(self) -> List[Dict]:
        """采集当前时刻的所有关键指标"""
        current_time = datetime.now()
        
        # 与trace_collector保持一致的时间戳格式
        timestamp_us = int(current_time.timestamp() * 1000000)  # 微秒时间戳
        
        metrics_data = {
            'timestamp': timestamp_us,  # 微秒时间戳，与trace格式一致
            'startTime': current_time.strftime('%Y-%m-%d %H:%M:%S'),  # 人类可读时间
            'minute_key': current_time.strftime("%H_%M"),
            'date': current_time.strftime("%Y-%m-%d")
        }
        
        # 采集每个指标
        successful_queries = 0
        for metric_name, query in self.key_metrics.items():
            try:
                value = self._query_prometheus(query)
                metrics_data[metric_name] = round(value, 6)  # 保留6位小数
                successful_queries += 1
                self.logger.debug(f"采集指标 {metric_name}: {value}")
            except Exception as e:
                self.logger.warning(f"采集指标 {metric_name} 失败: {e}")
                metrics_data[metric_name] = 0.0
                self.stats["failed_queries"] += 1
        
        # 记录采集成功率
        success_rate = (successful_queries / len(self.key_metrics)) * 100
        self.logger.debug(f"本次采集成功率: {success_rate:.1f}%")
        
        return [metrics_data]

    def _query_prometheus(self, query: str) -> float:
        """查询Prometheus获取指标值 - 支持or查询"""
        params = {
            'query': query,
            'time': datetime.now().timestamp()
        }
        
        try:
            response = self.session.get(f"{self.prometheus_url}/api/v1/query", params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    # 如果有多个结果，计算平均值
                    values = []
                    for result in data['data']['result']:
                        if len(result['value']) > 1:
                            try:
                                value = float(result['value'][1])
                                # 检查是否为有效数值
                                if not (value != value) and value >= 0:  # 检查NaN和负值
                                    values.append(value)
                            except (ValueError, TypeError):
                                continue
                    
                    if values:
                        # 对于成功率，确保在0-1之间
                        avg_value = sum(values) / len(values)
                        if 'success_rate' in query and avg_value > 1:
                            avg_value = min(avg_value, 1.0)
                        return avg_value
                    
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"查询失败 {query}: {e}")
            return 0.0

    def save_metrics_data(self, metrics_data: List[Dict]):
        """保存指标数据到CSV文件 - 改为追加模式以支持高频采集"""
        if not metrics_data:
            return
        
        today, today_dir, csv_dir = self._get_current_date_dirs()
        
        current_time = datetime.now()
        filename = current_time.strftime("%H_%M")
        csv_file = os.path.join(csv_dir, f"{filename}.csv")
        
        # 动态生成字段名（基于实际可用的指标）
        base_fields = ['timestamp', 'startTime', 'minute_key', 'date']
        metric_fields = list(self.key_metrics.keys())
        fieldnames = base_fields + metric_fields
        
        # 检查文件是否存在
        file_exists = os.path.exists(csv_file)
        
        # 使用追加模式，支持同一分钟内多次写入
        with open(csv_file, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            for metrics in metrics_data:
                # 确保所有字段都存在
                row = {field: metrics.get(field, 0.0) for field in fieldnames}
                writer.writerow(row)
        
        # 修复：移除依赖undefined变量的日志输出
        self.logger.debug(f"保存指标数据: {len(metrics_data)} 样本 -> metrics/{today}/{filename}")

    def start_collection(self, duration_minutes: int = 60, interval_seconds: int = None) -> bool:
        """开始指标采集"""
        # 修改默认采集间隔为更高频率
        if interval_seconds is None:
            interval_seconds = 1
            
        self.logger.info("开始系统指标采集")
        
        if not self.test_connection():
            self.logger.error("无法连接到 Prometheus，采集终止")
            return False
        
        if duration_minutes <= 0:
            self.logger.info("持续运行模式 (duration <= 0)")
            end_time = float('inf')
        else:
            self.logger.info(f"采集配置: 持续 {duration_minutes} 分钟")
            end_time = time.time() + (duration_minutes * 60)
        
        self.logger.info(f"采集间隔: {interval_seconds} 秒")
        
        self.stats["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        sample_count = 0
        last_date = None
        
        try:
            while time.time() < end_time:
                sample_start = time.time()
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                # 检测日期变化
                if last_date and last_date != current_date:
                    self.logger.info(f"日期变更: {last_date} -> {current_date}")
                
                last_date = current_date
                
                # 采集指标
                metrics_batch = self.collect_metrics()
                
                if metrics_batch:
                    # 保存数据
                    self.save_metrics_data(metrics_batch)
                    self.collected_metrics.extend(metrics_batch)
                    sample_count += 1
                    self.stats["total_samples"] += 1
                
                # 显示进度 - 降低日志频率
                if sample_count % 4 == 0:  # 每4次采集显示一次状态
                    if duration_minutes > 0:
                        elapsed_minutes = (time.time() - start_time) / 60
                        progress_info = f"进度: {elapsed_minutes:.1f}/{duration_minutes}min"
                    else:
                        elapsed_hours = (time.time() - start_time) / 3600
                        progress_info = f"运行时间: {elapsed_hours:.1f}h"
                    
                    success_rate = ((self.stats["total_samples"] * len(self.key_metrics) - self.stats["failed_queries"]) / 
                                   max(self.stats["total_samples"] * len(self.key_metrics), 1)) * 100
                    
                    self.logger.info(f"指标采集状态: {progress_info} | "
                                   f"样本数: {sample_count} | "
                                   f"成功率: {success_rate:.1f}%")
                
                # 等待下一个采集周期
                sample_duration = time.time() - sample_start
                sleep_time = max(0, interval_seconds - sample_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.info("用户中断采集")
        except Exception as e:
            self.logger.error(f"采集异常: {e}")
        finally:
            self.stats["end_time"] = datetime.now().isoformat()
            self._print_final_stats()
        
        self.logger.info("指标采集完成")
        return True

    def _print_final_stats(self):
        """打印最终统计信息"""
        total_samples = len(self.collected_metrics)
        if total_samples == 0:
            self.logger.warning("未采集到数据")
            return
        
        total_queries = total_samples * len(self.key_metrics)
        success_queries = total_queries - self.stats["failed_queries"]
        success_rate = (success_queries / max(total_queries, 1)) * 100
        
        self.logger.info("=" * 50)
        self.logger.info("指标采集完成 - 最终统计")
        self.logger.info("=" * 50)
        self.logger.info(f"总样本数: {total_samples}")
        self.logger.info(f"指标类型: {len(self.key_metrics)} 种")
        self.logger.info(f"查询成功率: {success_rate:.1f}%")
        self.logger.info("=" * 50)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Ticket 系统指标采集器")
    parser.add_argument("--duration", type=int, default=30, help="采集持续时间（分钟），0=持续运行")
    parser.add_argument("--interval", type=int, default=15, help="采集间隔（秒），默认: 15")  # 修改默认值
    parser.add_argument("--test", action="store_true", help="测试连接")
    
    args = parser.parse_args()
    
    collector = PrometheusMetricsCollector()
    
    if args.test:
        print("测试连接中...")
        return 0 if collector.test_connection() else 1
    
    try:
        success = collector.start_collection(
            duration_minutes=args.duration,
            interval_seconds=args.interval
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        print("采集中断")
        return 0

if __name__ == "__main__":
    exit(main())
