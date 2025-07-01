# -*- coding: utf-8 -*-
"""
Train Ticket Trace Collector for Anomaly Detection
采集 Train Ticket 系统的链路追踪数据，用于异常检测分析

Author: LoveShikiNatsume
Date: 2025-06-18
Version: 2.0
"""

import requests
import json
import time
import logging
import os
import csv
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Optional
from config import Config

class AnomalyDetectionTraceCollector:
    """异常检测链路追踪数据采集器"""
    
    def __init__(self):
        self.config = Config()
        self.base_url = f"http://{self.config.JAEGER_HOST}:{self.config.JAEGER_PORT}"
        self.api_url = f"{self.base_url}/jaeger/api"
        
        self.base_output_dir = self.config.ensure_output_dir()
        
        self.logger = self._setup_logging()
        self.session = requests.Session()
        self.session.timeout = self.config.REQUEST_TIMEOUT
        
        self.operation_encoder = {}
        self.service_encoder = {}
        self.operation_counter = 1
        self.service_counter = 1
        
        self.collected_data = []
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_traces": 0,
            "total_spans": 0,
            "error_spans": 0,
            "spans_with_parent": 0,
            "db_spans": 0
        }
        
        self.logger.info("Train Ticket 链路追踪采集器初始化完成")
        self.logger.info(f"Jaeger API: {self.api_url}")
        self.logger.info(f"输出目录: {self.base_output_dir}")

    def _get_current_date_dirs(self):
        """获取当前日期的目录路径"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_dir = os.path.join(self.base_output_dir, today)
        csv_dir = os.path.join(today_dir, "csv")
        
        # 确保目录存在
        os.makedirs(csv_dir, exist_ok=True)
        
        return today, today_dir, csv_dir

    def _setup_logging(self):
        """设置日志输出"""
        logger = logging.getLogger('TraceCollector')
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
        """测试与 Jaeger 的连接"""
        try:
            response = self.session.get(f"{self.api_url}/services")
            if response.status_code == 200:
                data = response.json()
                services = data.get("data", [])
                trainticket_services = [s for s in services if "trainticket" in s]
                self.logger.info(f"连接成功，发现 Train Ticket 服务: {len(trainticket_services)} 个")
                return len(trainticket_services) > 0
            else:
                self.logger.error(f"连接失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            return False

    def get_available_services(self) -> List[str]:
        """获取可用的 Train Ticket 服务列表"""
        try:
            response = self.session.get(f"{self.api_url}/services")
            if response.status_code == 200:
                all_services = response.json().get("data", [])
                return [s for s in all_services if "trainticket" in s]
            return []
        except Exception as e:
            self.logger.error(f"获取服务列表失败: {e}")
            return []

    def collect_traces(self, service: str = None, lookback: str = "5m", limit: int = 100) -> List[Dict]:
        """从指定服务采集链路追踪数据"""
        try:
            params = {"lookback": lookback, "limit": limit}
            if service:
                params["service"] = service
            
            self.stats["total_requests"] += 1
            response = self.session.get(f"{self.api_url}/traces", params=params)
            
            if response.status_code == 200:
                traces = response.json().get("data", [])
                self.stats["successful_requests"] += 1
                self.stats["total_traces"] += len(traces)
                return traces
            else:
                self.stats["failed_requests"] += 1
                return []
        except Exception as e:
            self.stats["failed_requests"] += 1
            return []

    def _encode_operation(self, operation_name: str) -> int:
        """将操作名编码为数字"""
        if not operation_name:
            operation_name = "unknown"
        if operation_name not in self.operation_encoder:
            self.operation_encoder[operation_name] = self.operation_counter
            self.operation_counter += 1
        return self.operation_encoder[operation_name]

    def _encode_service(self, service_name: str) -> int:
        """将服务名编码为数字"""
        if not service_name:
            service_name = "unknown"
        if service_name not in self.service_encoder:
            self.service_encoder[service_name] = self.service_counter
            self.service_counter += 1
        return self.service_encoder[service_name]

    def _extract_parent_span_id(self, span: Dict) -> str:
        """从 references 字段中提取父 span ID"""
        references = span.get("references", [])
        
        for ref in references:
            if ref.get("refType") == "CHILD_OF":
                parent_span_id = ref.get("spanID", "")
                if parent_span_id:
                    return parent_span_id
        return ""

    def _extract_service_name(self, span: Dict, tags: Dict[str, str]) -> str:
        """从 Istio span 中提取服务名称"""
        service_name = tags.get("istio.canonical_service", "")
        if service_name:
            return service_name
        
        operation_name = span.get("operationName", "")
        if operation_name:
            match = re.match(r'(ts-[^.]+)', operation_name)
            if match:
                return match.group(1)
            return operation_name.split('.')[0] if '.' in operation_name else operation_name
        
        return "unknown-service"

    def _split_trace_id(self, trace_id: str) -> tuple:
        """将 32 位 trace ID 分割为高位和低位"""
        try:
            if len(trace_id) == 32:
                high = int(trace_id[:16], 16)
                low = int(trace_id[16:], 16)
                return high, low
            elif len(trace_id) == 16:
                return 0, int(trace_id, 16)
            else:
                hash_val = int(hashlib.md5(trace_id.encode()).hexdigest()[:16], 16)
                return 0, hash_val
        except:
            return 0, 0

    def _calculate_db_hash(self, tags: Dict[str, str]) -> int:
        """计算数据库哈希值，基于 HTTP 请求信息"""
        http_info = []
        for key in ['http.method', 'http.url', 'http.status_code']:
            if key in tags and tags[key]:
                http_info.append(f"{key}:{tags[key]}")
        
        if http_info:
            combined = "|".join(http_info)
            hash_val = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
            return hash_val % 1000000
        
        return 0

    def _calculate_node_latency_label(self, duration_ms: float) -> int:
        """计算节点延迟标签：0=快速(<100ms), 1=中等(<1s), 2=慢速(>=1s)"""
        if duration_ms < 100:
            return 0
        elif duration_ms < 1000:
            return 1
        else:
            return 2

    def _safe_int_from_hex(self, hex_value, default=0) -> int:
        """将十六进制字符串转换为整数"""
        try:
            if hex_value is None or hex_value == "":
                return default
            if isinstance(hex_value, (int, float)):
                return int(hex_value)
            hex_str = str(hex_value)
            if hex_str.startswith('0x'):
                return int(hex_str, 16)
            else:
                return int(hex_str, 16)
        except:
            return default

    def _extract_tags(self, tag_list: List[Dict]) -> Dict[str, str]:
        """提取 span 标签到字典"""
        tags = {}
        for tag in tag_list:
            try:
                key = str(tag.get("key", ""))
                value = str(tag.get("value", ""))
                tags[key] = value
            except:
                continue
        return tags

    def parse_traces(self, traces: List[Dict]) -> List[Dict]:
        """解析链路数据为异常检测所需格式"""
        parsed_spans = []
        
        for trace in traces:
            trace_id = trace.get("traceID", "")
            trace_id_high, trace_id_low = self._split_trace_id(trace_id)
            
            for span in trace.get("spans", []):
                try:
                    raw_span_id = span.get("spanID", "")
                    raw_parent_span_id = self._extract_parent_span_id(span)
                    
                    span_id = self._safe_int_from_hex(raw_span_id, 0)
                    parent_span_id = self._safe_int_from_hex(raw_parent_span_id, 0)
                    
                    if parent_span_id != 0:
                        self.stats["spans_with_parent"] += 1
                    
                    start_time_us = span.get("startTime", 0)
                    duration_us = span.get("duration", 0)
                    
                    start_time_formatted = ""
                    nanosecond = 0
                    try:
                        if start_time_us and start_time_us > 0:
                            start_time_us = int(start_time_us)
                            dt = datetime.fromtimestamp(start_time_us / 1000000)
                            start_time_formatted = dt.strftime('%Y-%m-%d %H:%M:%S')
                            nanosecond = (start_time_us % 1000000) * 1000
                        else:
                            start_time_formatted = "1970-01-01 00:00:00"
                            nanosecond = 0
                    except:
                        start_time_formatted = "1970-01-01 00:00:00"
                        nanosecond = 0
                    
                    try:
                        duration_us = int(duration_us) if duration_us else 0
                        duration_ms = duration_us / 1000.0
                    except:
                        duration_ms = 0.0
                    
                    span_tags = self._extract_tags(span.get("tags", []))
                    operation_name = span.get("operationName", "")
                    service_name = self._extract_service_name(span, span_tags)
                    
                    service_encoded = self._encode_service(service_name)
                    operation_encoded = self._encode_operation(operation_name)
                    
                    status = 0
                    http_status = span_tags.get("http.status_code", "200")
                    if http_status and http_status.isdigit():
                        status = int(http_status)
                    
                    has_error = False
                    if span_tags.get("error", "false").lower() == "true" or status >= 400:
                        has_error = True
                        self.stats["error_spans"] += 1
                    
                    node_latency_label = self._calculate_node_latency_label(duration_ms)
                    
                    db_hash = self._calculate_db_hash(span_tags)
                    if db_hash > 0:
                        self.stats["db_spans"] += 1
                    
                    span_data = {
                        "traceIdHigh": trace_id_high,
                        "traceIdLow": trace_id_low,
                        "parentSpanId": parent_span_id,
                        "spanId": span_id,
                        "startTime": start_time_formatted,
                        "duration": int(duration_ms),
                        "nanosecond": nanosecond,
                        "DBhash": db_hash,
                        "status": status,
                        "operationName": operation_encoded,
                        "serviceName": service_encoded,
                        "nodeLatencyLabel": node_latency_label,
                        
                        "_original_trace_id": trace_id,
                        "_original_span_id": raw_span_id,
                        "_original_parent_span_id": raw_parent_span_id,
                        "_original_service_name": service_name,
                        "_original_operation_name": operation_name,
                        "_has_error": has_error,
                        "_collection_timestamp": datetime.now().isoformat()
                    }
                    
                    parsed_spans.append(span_data)
                    
                except Exception as e:
                    continue
        
        self.stats["total_spans"] += len(parsed_spans)
        return parsed_spans

    def save_data(self, data: List[Dict], timestamp: str):
        """保存数据到文件"""
        if not data:
            return
        
        today, today_dir, csv_dir = self._get_current_date_dirs()
        
        time_part = timestamp.split("T")[1]
        hour_minute = time_part.split(":")[0] + "_" + time_part.split(":")[1]
        filename = hour_minute
        
        csv_file = os.path.join(csv_dir, f"{filename}.csv")
        self._save_csv(data, csv_file)
        
        # 保存映射表
        mapping_file = os.path.join(today_dir, f"mapping_{today.replace('-', '')}.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump({
                "operation_mapping": self.operation_encoder,
                "service_mapping": self.service_encoder,
                "reverse_operation_mapping": {v: k for k, v in self.operation_encoder.items()},
                "reverse_service_mapping": {v: k for k, v in self.service_encoder.items()},
                "last_updated": timestamp
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"保存数据: {len(data)} spans -> {today}/{filename}")

    def _save_csv(self, data: List[Dict], filepath: str):
        """保存为异常检测专用的 CSV 格式"""
        if not data:
            return
        
        fieldnames = [
            "traceIdHigh", "traceIdLow", "parentSpanId", "spanId", 
            "startTime", "duration", "nanosecond", "DBhash", "status",
            "operationName", "serviceName", "nodeLatencyLabel"
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for span in data:
                row = {field: span.get(field, 0) for field in fieldnames}
                writer.writerow(row)

    def start_collection(self, duration_minutes: int = 60, interval_seconds: int = None) -> bool:
        """开始链路追踪数据采集"""
        # 使用配置文件中的默认间隔
        if interval_seconds is None:
            interval_seconds = self.config.DEFAULT_COLLECTION_INTERVAL
            
        self.logger.info("开始链路追踪数据采集")
        
        if not self.test_connection():
            self.logger.error("无法连接到 Jaeger，采集终止")
            return False
        
        services = self.get_available_services()
        if not services:
            self.logger.error("未发现 Train Ticket 服务，采集终止")
            return False
        
        if duration_minutes <= 0:
            self.logger.info("持续运行模式 (duration <= 0)")
            end_time = float('inf')
        else:
            self.logger.info(f"采集配置: {len(services)} 个服务, 持续 {duration_minutes} 分钟")
            end_time = time.time() + (duration_minutes * 60)
        
        self.logger.info(f"采集间隔: {interval_seconds} 秒")
        
        self.stats["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        batch_number = 1
        last_date = None
        
        try:
            while time.time() < end_time:
                batch_start = time.time()
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                # 检测日期变化
                if last_date and last_date != current_date:
                    self.logger.info(f"日期变更: {last_date} -> {current_date}")
                
                last_date = current_date
                
                self.logger.debug(f"批次 {batch_number} 开始 ({current_date})")
                
                # 采集数据
                all_batch_data = []
                for service in services:
                    traces = self.collect_traces(service=service, lookback="5m", limit=50)
                    if traces:
                        parsed_data = self.parse_traces(traces)
                        all_batch_data.extend(parsed_data)
                    time.sleep(0.5)
                
                # 保存数据
                if all_batch_data:
                    current_time = datetime.now().isoformat()
                    self.save_data(all_batch_data, current_time)
                    self.collected_data.extend(all_batch_data)
                
                # 显示进度
                if duration_minutes > 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    progress_info = f"进度: {elapsed_minutes:.1f}/{duration_minutes}min"
                else:
                    elapsed_hours = (time.time() - start_time) / 3600
                    progress_info = f"运行时间: {elapsed_hours:.1f}h"
                
                total_spans = self.stats["total_spans"]
                parent_rate = (self.stats["spans_with_parent"] / max(total_spans, 1)) * 100
                db_rate = (self.stats["db_spans"] / max(total_spans, 1)) * 100
                
                self.logger.info(f"采集状态: {progress_info} | "
                               f"Span总数: {total_spans} | "
                               f"父子关系: {parent_rate:.1f}% | "
                               f"DB哈希: {db_rate:.1f}%")
                
                batch_number += 1
                
                # 等待下一个采集周期
                batch_duration = time.time() - batch_start
                sleep_time = max(0, interval_seconds - batch_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.info("用户中断采集")
        except Exception as e:
            self.logger.error(f"采集异常: {e}")
        finally:
            self.stats["end_time"] = datetime.now().isoformat()
            self._print_final_stats()
        
        self.logger.info("数据采集完成")
        return True

    def _print_final_stats(self):
        """打印最终统计信息"""
        total_spans = len(self.collected_data)
        if total_spans == 0:
            self.logger.warning("未采集到数据")
            return
        
        error_spans = len([s for s in self.collected_data if s.get("_has_error", False)])
        parent_spans = len([s for s in self.collected_data if s.get("parentSpanId", 0) > 0])
        db_hash_spans = len([s for s in self.collected_data if s.get("DBhash", 0) > 0])
        
        self.logger.info("=" * 50)
        self.logger.info("采集完成 - 最终统计")
        self.logger.info("=" * 50)
        self.logger.info(f"Span 总数: {total_spans:,}")
        self.logger.info(f"唯一服务数: {len(self.service_encoder)}")
        self.logger.info(f"唯一操作数: {len(self.operation_encoder)}")
        self.logger.info("=" * 50)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Ticket 链路追踪数据采集器")
    parser.add_argument("--duration", type=int, default=30, help="采集持续时间（分钟），0=持续运行")
    parser.add_argument("--interval", type=int, default=None, help=f"采集间隔（秒），默认: {Config().DEFAULT_COLLECTION_INTERVAL}")
    parser.add_argument("--test", action="store_true", help="测试连接")
    
    args = parser.parse_args()
    
    collector = AnomalyDetectionTraceCollector()
    
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