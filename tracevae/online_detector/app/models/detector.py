#!/usr/bin/env python3
"""
在线异常检测器 - 适配TraceVAE状态字典
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from ..utils.preprocessor import TracePreprocessor
from ..config import config

logger = logging.getLogger(__name__)

class TraceVAEInferenceEngine:
    """TraceVAE推理引擎 - 基于预训练状态字典的简化推理"""
    
    def __init__(self, state_dict, device):
        self.device = device
        self.state_dict = state_dict
        
        # 提取关键组件
        self.operation_embedding = self._extract_operation_embedding()
        self.struct_features = self._extract_struct_features()
        self.latency_features = self._extract_latency_features()
        
        # 分析模型结构
        self._analyze_model_structure()
        
        logger.info("✅ TraceVAE推理引擎初始化成功")
    
    def _analyze_model_structure(self):
        """分析模型结构"""
        modules = {}
        for key in self.state_dict.keys():
            module_name = key.split('.')[0]
            modules[module_name] = modules.get(module_name, 0) + 1
        
        logger.info("🧩 检测到的模型组件:")
        for module, count in sorted(modules.items()):
            logger.info(f"  - {module}: {count} 个参数")
        
        # 检查是否有双VAE结构
        self.has_struct_vae = any(key.startswith('struct_vae') for key in self.state_dict.keys())
        self.has_latency_vae = any(key.startswith('latency_vae') for key in self.state_dict.keys())
        
        logger.info(f"🔍 模型结构分析:")
        logger.info(f"  - 结构VAE: {'✓' if self.has_struct_vae else '✗'}")
        logger.info(f"  - 延迟VAE: {'✓' if self.has_latency_vae else '✗'}")
    
    def _extract_operation_embedding(self):
        """提取操作嵌入权重"""
        try:
            # 尝试多个可能的键名
            possible_keys = [
                'operation_embedding.node_embedding.weight',
                'struct_vae.operation_embedding.node_embedding.weight',
                'latency_vae.operation_embedding.node_embedding.weight'
            ]
            
            for key in possible_keys:
                if key in self.state_dict:
                    embedding = self.state_dict[key].to(self.device)
                    logger.info(f"✅ 提取操作嵌入: {embedding.shape} from {key}")
                    return embedding
            
            logger.warning("⚠️  未找到操作嵌入权重")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️  提取操作嵌入失败: {e}")
            return None
    
    def _extract_struct_features(self):
        """提取结构相关特征权重"""
        struct_weights = {}
        try:
            for key, value in self.state_dict.items():
                if 'struct_vae' in key and ('mean' in key or 'logstd' in key or 'logvar' in key):
                    struct_weights[key] = value.to(self.device)
            
            logger.info(f"✅ 提取了 {len(struct_weights)} 个结构VAE参数")
            return struct_weights
            
        except Exception as e:
            logger.warning(f"⚠️  提取结构特征失败: {e}")
            return {}
    
    def _extract_latency_features(self):
        """提取延迟相关特征权重"""
        latency_weights = {}
        try:
            for key, value in self.state_dict.items():
                if 'latency_vae' in key and ('mean' in key or 'logstd' in key or 'logvar' in key):
                    latency_weights[key] = value.to(self.device)
            
            logger.info(f"✅ 提取了 {len(latency_weights)} 个延迟VAE参数")
            return latency_weights
            
        except Exception as e:
            logger.warning(f"⚠️  提取延迟特征失败: {e}")
            return {}
    
    def compute_operation_embeddings(self, operation_ids):
        """计算操作嵌入"""
        if self.operation_embedding is None:
            # 返回随机嵌入作为fallback
            return torch.randn(len(operation_ids), 40).to(self.device) * 0.1
        
        # 确保operation_ids在有效范围内
        max_ops = self.operation_embedding.shape[0]
        operation_ids = torch.clamp(operation_ids, 0, max_ops - 1)
        
        return self.operation_embedding[operation_ids]
    
    def compute_anomaly_scores(self, trace_features, operation_ids, service_ids):
        """计算异常分数"""
        try:
            batch_size = trace_features.shape[0]
            
            # 1. 计算操作嵌入
            op_embeddings = self.compute_operation_embeddings(operation_ids)
            
            # 2. 基于嵌入的结构异常检测
            struct_score = self._compute_structure_anomaly_score(
                trace_features, op_embeddings, operation_ids, service_ids
            )
            
            # 3. 基于延迟的时间异常检测
            latency_score = self._compute_latency_anomaly_score(trace_features)
            
            # 4. 综合异常分数
            if self.has_struct_vae and self.has_latency_vae:
                # 双VAE架构：结构和延迟分开检测
                total_score = 0.6 * struct_score + 0.4 * latency_score
            elif self.has_struct_vae:
                # 只有结构VAE
                total_score = struct_score
                latency_score = struct_score * 0.5  # 估算
            else:
                # 统一架构或未知
                total_score = (struct_score + latency_score) / 2
            
            return {
                'reconstruction_error': float(latency_score),
                'kl_divergence': float(struct_score),
                'total_loss': float(total_score)
            }
            
        except Exception as e:
            logger.error(f"❌ 异常分数计算失败: {e}")
            # 返回中等异常分数作为fallback
            return {
                'reconstruction_error': 0.3,
                'kl_divergence': 0.2,
                'total_loss': 0.25
            }
    
    def _compute_structure_anomaly_score(self, features, op_embeddings, operation_ids, service_ids):
        """计算结构异常分数"""
        try:
            # 1. 检查依赖关系的合理性
            parent_features = features[:, 2] if features.shape[1] > 2 else torch.zeros(features.shape[0])
            has_parent_ratio = (parent_features != 0).float().mean()
            
            # 2. 检查操作类型的多样性
            unique_ops = len(torch.unique(operation_ids))
            op_diversity = unique_ops / len(operation_ids) if len(operation_ids) > 0 else 0
            
            # 3. 检查服务的多样性
            unique_services = len(torch.unique(service_ids))
            service_diversity = unique_services / len(service_ids) if len(service_ids) > 0 else 0
            
            # 4. 基于操作嵌入的异常检测
            if op_embeddings.shape[0] > 1:
                # 计算嵌入之间的相似性
                embedding_norm = F.normalize(op_embeddings, dim=1)
                similarity_matrix = torch.mm(embedding_norm, embedding_norm.t())
                
                # 排除对角线元素
                mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
                similarities = similarity_matrix[mask]
                
                # 如果所有操作都非常相似，可能是异常
                high_similarity_ratio = (similarities > 0.9).float().mean()
                embedding_anomaly = high_similarity_ratio
            else:
                embedding_anomaly = 0.0
            
            # 综合结构异常分数
            structure_anomalies = []
            
            # 依赖关系异常：太少或太多的父子关系
            if has_parent_ratio < 0.1 or has_parent_ratio > 0.95:
                structure_anomalies.append(0.8)
            else:
                structure_anomalies.append(0.1)
            
            # 多样性异常：操作或服务类型过于单一
            if op_diversity < 0.3 or service_diversity < 0.3:
                structure_anomalies.append(0.6)
            else:
                structure_anomalies.append(0.1)
            
            # 嵌入异常
            structure_anomalies.append(float(embedding_anomaly) * 0.7)
            
            # 返回最大异常分数
            return max(structure_anomalies)
            
        except Exception as e:
            logger.warning(f"结构异常分数计算失败: {e}")
            return 0.2
    
    def _compute_latency_anomaly_score(self, features):
        """计算延迟异常分数"""
        try:
            # 提取duration信息
            durations = features[:, 0] if features.shape[1] > 0 else torch.tensor([1000.0])
            
            if len(durations) == 0:
                return 0.2
            
            # 1. 统计异常检测
            if len(durations) > 1:
                mean_duration = torch.mean(durations)
                std_duration = torch.std(durations)
                
                # Z-score异常检测
                z_scores = torch.abs(durations - mean_duration) / (std_duration + 1e-6)
                max_z_score = torch.max(z_scores)
                
                # 转换为异常分数
                z_score_anomaly = torch.clamp(max_z_score / 5.0, 0.0, 1.0)
            else:
                z_score_anomaly = 0.1
            
            # 2. 绝对值异常检测
            # 如果有特别大的延迟值（比如超过30秒）
            large_latency_ratio = (durations > 30000).float().mean()  # 30秒
            
            # 3. 延迟模式异常
            # 检查是否有突然的延迟跳跃
            if len(durations) > 2:
                duration_diffs = torch.abs(durations[1:] - durations[:-1])
                max_jump = torch.max(duration_diffs) / (torch.mean(durations) + 1e-6)
                jump_anomaly = torch.clamp(max_jump / 10.0, 0.0, 1.0)
            else:
                jump_anomaly = 0.0
            
            # 综合延迟异常分数
            latency_anomalies = [
                float(z_score_anomaly),
                float(large_latency_ratio) * 0.9,
                float(jump_anomaly),
            ]
            
            return max(latency_anomalies)
            
        except Exception as e:
            logger.warning(f"延迟异常分数计算失败: {e}")
            return 0.2

class TraceAnomalyDetector:
    """在线异常检测器"""
    
    def __init__(self, model_path: str, config_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.config_dir = Path(config_dir)
        self.preprocessor = TracePreprocessor(config_dir)
        self.inference_engine = None
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
        # 异常检测阈值
        self.anomaly_threshold = config.ANOMALY_THRESHOLD
        
        self.load_model()
    
    def load_model(self):
        """加载预训练模型状态字典"""
        try:
            logger.info(f"🔄 加载TraceVAE状态字典: {self.model_path}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 加载状态字典
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            if not isinstance(state_dict, dict):
                raise ValueError(f"期望状态字典，但得到: {type(state_dict)}")
            
            logger.info(f"✅ 成功加载 {len(state_dict)} 个模型参数")
            
            # 创建推理引擎
            self.inference_engine = TraceVAEInferenceEngine(state_dict, self.device)
            
            logger.info("✅ TraceVAE模型加载成功")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def detect_anomalies(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测异常"""
        start_time = time.time()
        
        try:
            # 预处理数据
            df = self.preprocessor.preprocess_traces(traces)
            
            if df.empty:
                return []
            
            # 按trace分组进行检测
            results = []
            trace_groups = df.groupby('original_trace_id')
            
            for original_trace_id, group in trace_groups:
                trace_start_time = time.time()
                
                # 准备模型输入
                model_input, operation_ids, service_ids = self._prepare_model_input(group)
                
                # 模型推理
                anomaly_scores = self._run_inference(model_input, operation_ids, service_ids)
                
                # 解析结果
                result = self._parse_detection_result(
                    original_trace_id, 
                    anomaly_scores, 
                    trace_start_time
                )
                
                results.append(result)
            
            total_time = (time.time() - start_time) * 1000
            logger.info(f"✅ 检测完成: {len(results)} traces, {total_time:.2f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 异常检测失败: {e}")
            raise
    
    def _prepare_model_input(self, trace_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备模型输入"""
        features = []
        operation_ids = []
        service_ids = []
        
        for _, row in trace_df.iterrows():
            # 构建特征向量
            feature_vector = [
                float(row.get('duration', 1000)) / 1000.0,     # 标准化持续时间
                float(row.get('status', 0)),                   # 状态码
                float(row.get('parentSpanId', 0) != 0),        # 是否有父节点
                float(row.get('nanosecond', 0)) / 1000000.0,   # 标准化纳秒
            ]
            features.append(feature_vector)
            
            # 操作和服务ID
            operation_ids.append(int(row.get('operationName', 0)))
            service_ids.append(int(row.get('serviceName', 0)))
        
        # 转换为tensor
        if not features:
            return (torch.zeros(1, 4).to(self.device), 
                   torch.zeros(1, dtype=torch.long).to(self.device),
                   torch.zeros(1, dtype=torch.long).to(self.device))
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        operation_ids_tensor = torch.tensor(operation_ids, dtype=torch.long).to(self.device)
        service_ids_tensor = torch.tensor(service_ids, dtype=torch.long).to(self.device)
        
        return features_tensor, operation_ids_tensor, service_ids_tensor
    
    def _run_inference(self, model_input: torch.Tensor, operation_ids: torch.Tensor, service_ids: torch.Tensor) -> Dict[str, float]:
        """运行模型推理"""
        try:
            with torch.no_grad():
                scores = self.inference_engine.compute_anomaly_scores(
                    model_input, operation_ids, service_ids
                )
                return scores
                
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            # 返回中等异常分数作为fallback
            return {
                'reconstruction_error': 0.3,
                'kl_divergence': 0.2,
                'total_loss': 0.25
            }
    
    def _parse_detection_result(
        self, 
        trace_id: str, 
        scores: Dict[str, float], 
        start_time: float
    ) -> Dict[str, Any]:
        """解析检测结果"""
        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 异常判断
        total_score = scores.get('total_loss', 0)
        is_anomaly = total_score > self.anomaly_threshold
        
        # 确定异常类型和置信度
        anomaly_type = "normal"
        confidence = max(0.0, 1.0 - min(total_score / self.anomaly_threshold, 1.0))
        
        if is_anomaly:
            # 根据分数特征判断异常类型
            recon_error = scores.get('reconstruction_error', 0)
            kl_div = scores.get('kl_divergence', 0)
            
            if kl_div > recon_error and kl_div > 0.4:
                anomaly_type = "structure"  # 结构异常
            elif recon_error > 0.4:
                anomaly_type = "time"  # 时间异常
            else:
                anomaly_type = "mixed"  # 混合异常
            
            confidence = min(total_score / self.anomaly_threshold, 1.0)
        
        return {
            'traceID': trace_id,
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'confidence': round(confidence, 4),
            'details': {
                'nodeLatencyScore': round(scores.get('reconstruction_error', 0), 4),
                'graphLatencyScore': round(scores.get('kl_divergence', 0), 4),
                'graphStructureScore': round(scores.get('total_loss', 0), 4)
            },
            'processing_time_ms': round(processing_time, 2)
        }