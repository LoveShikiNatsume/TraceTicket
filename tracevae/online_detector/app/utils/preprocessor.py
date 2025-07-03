#!/usr/bin/env python3
"""
数据预处理器
将实时trace数据转换为模型输入格式
"""

import hashlib
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

class TracePreprocessor:
    """Trace数据预处理器"""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.trace_id_mapping = {}
        self.span_id_mapping = {}
        
        # 加载配置映射
        self.operation_name_to_id = {'unknown': 0}
        self.service_name_to_id = {'unknown': 0}
        self.load_id_mappings()
    
    def load_id_mappings(self):
        """加载ID映射配置"""
        try:
            id_manager_dir = self.config_dir / "id_manager"
            
            # 检查配置文件是否存在
            if not id_manager_dir.exists():
                logger.warning("ID管理器目录不存在，使用默认映射")
                return
            
            # 尝试从现有CSV文件推断映射
            train_csv = self.config_dir / "train.csv"
            if train_csv.exists():
                self._infer_mappings_from_csv(train_csv)
            
            logger.info(f"✅ 加载了 {len(self.operation_name_to_id)} 个操作和 {len(self.service_name_to_id)} 个服务的映射")
            
        except Exception as e:
            logger.warning(f"⚠️  加载ID映射失败，使用默认映射: {e}")
    
    def _infer_mappings_from_csv(self, csv_path: Path):
        """从CSV文件推断映射关系"""
        try:
            # 读取少量数据来推断映射
            df_sample = pd.read_csv(csv_path, nrows=1000)
            
            # 获取唯一的操作和服务ID
            unique_operations = df_sample['operationName'].unique()
            unique_services = df_sample['serviceName'].unique()
            
            # 创建反向映射（这是简化版本，实际可能需要更复杂的逻辑）
            for i, op_id in enumerate(unique_operations):
                if op_id not in self.operation_name_to_id:
                    self.operation_name_to_id[f"operation_{op_id}"] = op_id
            
            for i, svc_id in enumerate(unique_services):
                if svc_id not in self.service_name_to_id:
                    self.service_name_to_id[f"service_{svc_id}"] = svc_id
                    
        except Exception as e:
            logger.warning(f"从CSV推断映射失败: {e}")
    
    def preprocess_traces(self, traces: List[Dict[str, Any]]) -> pd.DataFrame:
        """将trace数据预处理为模型输入格式"""
        records = []
        
        for trace in traces:
            trace_id = trace['traceID']
            spans = trace['spans']
            processes = trace.get('processes', {})
            
            for span in spans:
                # 获取服务名和操作名
                service_name = span.get('serviceName', 'unknown')
                operation_name = span.get('operationName', 'unknown')
                
                # 动态添加到映射中
                if operation_name not in self.operation_name_to_id:
                    self.operation_name_to_id[operation_name] = len(self.operation_name_to_id)
                
                if service_name not in self.service_name_to_id:
                    self.service_name_to_id[service_name] = len(self.service_name_to_id)
                
                # 获取状态码
                status_code = '200'
                for tag in span.get('tags', []):
                    if tag.get('key') in ['http.status_code', 'error']:
                        if tag.get('key') == 'error' and tag.get('value'):
                            status_code = 'error'
                        elif tag.get('key') == 'http.status_code':
                            status_code = str(tag.get('value', '200'))
                        break
                
                # ID映射
                trace_id_high, trace_id_low = self._get_trace_id_mapping(trace_id)
                span_id = self._get_span_id_mapping(span['spanID'])
                parent_span_id = self._get_span_id_mapping(span.get('parentSpanID', '')) if span.get('parentSpanID') else 0
                
                # 时间转换
                start_time = span.get('startTime', 0)
                duration = span.get('duration', 1000)
                
                start_time_sec = start_time / 1000000 if start_time > 0 else 0
                start_time_str = datetime.fromtimestamp(start_time_sec).strftime("%Y-%m-%d %H:%M:%S") if start_time_sec > 0 else "1970-01-01 00:00:00"
                duration_ms = max(1, int(duration / 1000))
                nanosecond = (start_time % 1000000) * 1000 if start_time > 0 else 0
                
                # 状态处理
                status_val = 0 if status_code == '200' else 1
                
                record = {
                    'traceIdHigh': trace_id_high,
                    'traceIdLow': trace_id_low,
                    'parentSpanId': parent_span_id,
                    'spanId': span_id,
                    'startTime': start_time_str,
                    'duration': duration_ms,
                    'nanosecond': int(nanosecond),
                    'DBhash': 0,
                    'status': status_val,
                    'operationName': self.operation_name_to_id[operation_name],
                    'serviceName': self.service_name_to_id[service_name],
                    'original_trace_id': trace_id  # 保留原始ID用于结果映射
                }
                
                records.append(record)
        
        return pd.DataFrame(records)
    
    def _get_trace_id_mapping(self, trace_id: str) -> Tuple[int, int]:
        """获取trace ID映射"""
        if trace_id not in self.trace_id_mapping:
            hash_obj = hashlib.md5(str(trace_id).encode())
            hash_int = int(hash_obj.hexdigest()[:16], 16)
            trace_id_high = (hash_int >> 32) & 0x7FFFFFFFFFFFFFFF
            trace_id_low = hash_int & 0x7FFFFFFFFFFFFFFF
            self.trace_id_mapping[trace_id] = (trace_id_high, trace_id_low)
        
        return self.trace_id_mapping[trace_id]
    
    def _get_span_id_mapping(self, span_id: str) -> int:
        """获取span ID映射"""
        if not span_id:
            return 0
            
        if span_id not in self.span_id_mapping:
            self.span_id_mapping[span_id] = random.randint(-2**31, 2**31-1)
        
        return self.span_id_mapping[span_id]#!/usr/bin/env python3
"""
数据预处理器
将实时trace数据转换为模型输入格式
"""

import hashlib
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

class TracePreprocessor:
    """Trace数据预处理器"""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.trace_id_mapping = {}
        self.span_id_mapping = {}
        
        # 加载配置映射
        self.operation_name_to_id = {'unknown': 0}
        self.service_name_to_id = {'unknown': 0}
        self.load_id_mappings()
    
    def load_id_mappings(self):
        """加载ID映射配置"""
        try:
            id_manager_dir = self.config_dir / "id_manager"
            
            # 检查配置文件是否存在
            if not id_manager_dir.exists():
                logger.warning("ID管理器目录不存在，使用默认映射")
                return
            
            # 尝试从现有CSV文件推断映射
            train_csv = self.config_dir / "train.csv"
            if train_csv.exists():
                self._infer_mappings_from_csv(train_csv)
            
            logger.info(f"✅ 加载了 {len(self.operation_name_to_id)} 个操作和 {len(self.service_name_to_id)} 个服务的映射")
            
        except Exception as e:
            logger.warning(f"⚠️  加载ID映射失败，使用默认映射: {e}")
    
    def _infer_mappings_from_csv(self, csv_path: Path):
        """从CSV文件推断映射关系"""
        try:
            # 读取少量数据来推断映射
            df_sample = pd.read_csv(csv_path, nrows=1000)
            
            # 获取唯一的操作和服务ID
            unique_operations = df_sample['operationName'].unique()
            unique_services = df_sample['serviceName'].unique()
            
            # 创建反向映射（这是简化版本，实际可能需要更复杂的逻辑）
            for i, op_id in enumerate(unique_operations):
                if op_id not in self.operation_name_to_id:
                    self.operation_name_to_id[f"operation_{op_id}"] = op_id
            
            for i, svc_id in enumerate(unique_services):
                if svc_id not in self.service_name_to_id:
                    self.service_name_to_id[f"service_{svc_id}"] = svc_id
                    
        except Exception as e:
            logger.warning(f"从CSV推断映射失败: {e}")
    
    def preprocess_traces(self, traces: List[Dict[str, Any]]) -> pd.DataFrame:
        """将trace数据预处理为模型输入格式"""
        records = []
        
        for trace in traces:
            trace_id = trace['traceID']
            spans = trace['spans']
            processes = trace.get('processes', {})
            
            for span in spans:
                # 获取服务名和操作名
                service_name = span.get('serviceName', 'unknown')
                operation_name = span.get('operationName', 'unknown')
                
                # 动态添加到映射中
                if operation_name not in self.operation_name_to_id:
                    self.operation_name_to_id[operation_name] = len(self.operation_name_to_id)
                
                if service_name not in self.service_name_to_id:
                    self.service_name_to_id[service_name] = len(self.service_name_to_id)
                
                # 获取状态码
                status_code = '200'
                for tag in span.get('tags', []):
                    if tag.get('key') in ['http.status_code', 'error']:
                        if tag.get('key') == 'error' and tag.get('value'):
                            status_code = 'error'
                        elif tag.get('key') == 'http.status_code':
                            status_code = str(tag.get('value', '200'))
                        break
                
                # ID映射
                trace_id_high, trace_id_low = self._get_trace_id_mapping(trace_id)
                span_id = self._get_span_id_mapping(span['spanID'])
                parent_span_id = self._get_span_id_mapping(span.get('parentSpanID', '')) if span.get('parentSpanID') else 0
                
                # 时间转换
                start_time = span.get('startTime', 0)
                duration = span.get('duration', 1000)
                
                start_time_sec = start_time / 1000000 if start_time > 0 else 0
                start_time_str = datetime.fromtimestamp(start_time_sec).strftime("%Y-%m-%d %H:%M:%S") if start_time_sec > 0 else "1970-01-01 00:00:00"
                duration_ms = max(1, int(duration / 1000))
                nanosecond = (start_time % 1000000) * 1000 if start_time > 0 else 0
                
                # 状态处理
                status_val = 0 if status_code == '200' else 1
                
                record = {
                    'traceIdHigh': trace_id_high,
                    'traceIdLow': trace_id_low,
                    'parentSpanId': parent_span_id,
                    'spanId': span_id,
                    'startTime': start_time_str,
                    'duration': duration_ms,
                    'nanosecond': int(nanosecond),
                    'DBhash': 0,
                    'status': status_val,
                    'operationName': self.operation_name_to_id[operation_name],
                    'serviceName': self.service_name_to_id[service_name],
                    'original_trace_id': trace_id  # 保留原始ID用于结果映射
                }
                
                records.append(record)
        
        return pd.DataFrame(records)
    
    def _get_trace_id_mapping(self, trace_id: str) -> Tuple[int, int]:
        """获取trace ID映射"""
        if trace_id not in self.trace_id_mapping:
            hash_obj = hashlib.md5(str(trace_id).encode())
            hash_int = int(hash_obj.hexdigest()[:16], 16)
            trace_id_high = (hash_int >> 32) & 0x7FFFFFFFFFFFFFFF
            trace_id_low = hash_int & 0x7FFFFFFFFFFFFFFF
            self.trace_id_mapping[trace_id] = (trace_id_high, trace_id_low)
        
        return self.trace_id_mapping[trace_id]
    
    def _get_span_id_mapping(self, span_id: str) -> int:
        """获取span ID映射"""
        if not span_id:
            return 0
            
        if span_id not in self.span_id_mapping:
            self.span_id_mapping[span_id] = random.randint(-2**31, 2**31-1)
        
        return self.span_id_mapping[span_id]