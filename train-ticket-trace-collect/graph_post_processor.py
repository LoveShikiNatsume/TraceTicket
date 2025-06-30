# -*- coding: utf-8 -*-
"""
Train Ticket Trace Graph Post-Processor
对已采集的12列数据进行后处理，添加图分析特征扩展为14列

Author: LoveShikiNatsume
Date: 2025-06-18
"""

import json
import pandas as pd
import numpy as np
import os
import logging
import csv
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set
from datetime import datetime
import networkx as nx

class TraceGraphPostProcessor:
    """调用链图后处理器 - 扩展现有12列数据为14列"""
    
    def __init__(self, data_dir: str = "trace_output", use_dynamic_thresholds: bool = True):
        self.data_dir = data_dir
        self.use_dynamic_thresholds = use_dynamic_thresholds
        self.logger = self._setup_logging()
        
        # 调整后的图分析阈值配置 - 更符合微服务架构特征
        self.latency_thresholds = {
            "normal": 1000,     # 正常图延迟 < 1000ms (1秒)
            "medium": 5000,     # 中等图延迟 < 5000ms (5秒)
            # 高延迟 >= 5000ms
        }
        
        self.structure_thresholds = {
            "simple": 3.0,      # 简单结构复杂度 < 3.0
            "medium": 8.0,      # 中等复杂度 < 8.0
            # 复杂结构 >= 8.0
        }
        
        # 动态阈值存储
        self.dynamic_latency_thresholds = None
        self.dynamic_structure_thresholds = None
        
        # 统计信息
        self.analysis_stats = {
            "total_traces": 0,
            "latency_distribution": {"0": 0, "1": 0, "2": 0},
            "structure_distribution": {"0": 0, "1": 0, "2": 0},
            "latency_values": [],
            "complexity_values": []
        }
        
        self.logger.info("图后处理器已初始化")
        self.logger.info(f"数据目录: {data_dir}")
        self.logger.info(f"动态阈值: {'启用' if use_dynamic_thresholds else '禁用'}")
        self.logger.info("将12列数据扩展为14列（添加图分析特征）")

    def _setup_logging(self):
        """设置日志"""
        logger = logging.getLogger('GraphPostProcessor')
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def load_csv_data(self, csv_path: str) -> List[Dict]:
        """从现有的12列CSV文件加载数据"""
        try:
            df = pd.read_csv(csv_path)
            
            # 验证是否是12列格式
            expected_12_columns = [
                "traceIdHigh", "traceIdLow", "parentSpanId", "spanId", 
                "startTime", "duration", "nanosecond", "DBhash", "status",
                "operationName", "serviceName", "nodeLatencyLabel"
            ]
            
            if len(df.columns) == 14:
                self.logger.debug(f"文件 {csv_path} 已经是14列格式，跳过")
                return []
            elif len(df.columns) != 12:
                self.logger.warning(f"文件 {csv_path} 列数不是12列: {len(df.columns)}")
                return []
            
            # 转换为字典列表
            spans_data = df.to_dict('records')
            
            # 为每个span添加必要的元数据用于图分析
            for span in spans_data:
                # 重构trace_id用于分组
                span['_trace_id'] = f"{span['traceIdHigh']}-{span['traceIdLow']}"
                
                # 从映射表恢复服务名（如果可能）
                span['_service_name'] = f"service_{span['serviceName']}"
                span['_operation_name'] = f"operation_{span['operationName']}"
            
            return spans_data
            
        except Exception as e:
            self.logger.error(f"加载CSV文件失败 {csv_path}: {e}")
            return []

    def load_mapping_data(self, date_dir: str) -> Dict:
        """加载映射表以恢复原始服务名"""
        mapping_files = [f for f in os.listdir(date_dir) if f.startswith('mapping_') and f.endswith('.json')]
        
        if not mapping_files:
            self.logger.warning(f"未找到映射文件: {date_dir}")
            return {}
        
        try:
            mapping_file = os.path.join(date_dir, mapping_files[0])
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            return mapping_data
        except Exception as e:
            self.logger.warning(f"加载映射文件失败: {e}")
            return {}

    def build_call_graph(self, trace_spans: List[Dict]) -> nx.DiGraph:
        """构建调用图"""
        G = nx.DiGraph()
        
        # 按 trace 分组
        traces = defaultdict(list)
        for span in trace_spans:
            trace_id = span['_trace_id']
            traces[trace_id].append(span)
        
        for trace_id, spans in traces.items():
            # 为每个 trace 构建子图
            span_map = {span['spanId']: span for span in spans}
            
            for span in spans:
                span_id = span['spanId']
                parent_id = span.get('parentSpanId', 0)
                service = span.get('_service_name', 'unknown')
                duration = span.get('duration', 0)
                
                # 添加节点
                if not G.has_node(span_id):
                    G.add_node(span_id, 
                              service=service,
                              duration=duration,
                              trace_id=trace_id)
                
                # 添加边（父子关系）
                if parent_id and parent_id != 0 and parent_id in span_map:
                    if not G.has_edge(parent_id, span_id):
                        G.add_edge(parent_id, span_id, 
                                  call_duration=duration,
                                  trace_id=trace_id)
        
        return G

    def calculate_critical_path_latency(self, G: nx.DiGraph, trace_id: str) -> float:
        """计算关键路径延迟"""
        try:
            # 获取该 trace 的所有节点
            trace_nodes = [n for n, d in G.nodes(data=True) if d.get('trace_id') == trace_id]
            
            if not trace_nodes:
                return 0.0
            
            # 创建子图
            subgraph = G.subgraph(trace_nodes)
            
            # 找到根节点（没有前驱的节点）
            root_nodes = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0]
            
            if not root_nodes:
                return 0.0
            
            # 计算从根节点到所有叶节点的最长路径
            max_path_latency = 0.0
            
            for root in root_nodes:
                try:
                    # 计算到所有节点的最长距离
                    distances = {}
                    distances[root] = G.nodes[root].get('duration', 0)
                    
                    # 拓扑排序
                    topo_order = list(nx.topological_sort(subgraph))
                    
                    for node in topo_order:
                        if node not in distances:
                            distances[node] = 0
                        
                        # 更新后继节点的距离
                        for successor in subgraph.successors(node):
                            successor_duration = G.nodes[successor].get('duration', 0)
                            new_distance = distances[node] + successor_duration
                            
                            if successor not in distances or new_distance > distances[successor]:
                                distances[successor] = new_distance
                    
                    # 找到最大距离
                    if distances:
                        max_path_latency = max(max_path_latency, max(distances.values()))
                        
                except nx.NetworkXError:
                    # 如果有环，使用简单求和
                    total_duration = sum(G.nodes[n].get('duration', 0) for n in trace_nodes)
                    max_path_latency = max(max_path_latency, total_duration)
            
            return max_path_latency
            
        except Exception as e:
            self.logger.debug(f"计算关键路径延迟失败: {e}")
            return 0.0

    def calculate_structure_complexity(self, G: nx.DiGraph, trace_id: str) -> float:
        """优化的结构复杂度计算"""
        try:
            # 获取该 trace 的所有节点
            trace_nodes = [n for n, d in G.nodes(data=True) if d.get('trace_id') == trace_id]
            
            if not trace_nodes:
                return 0.0
            
            subgraph = G.subgraph(trace_nodes)
            node_count = len(trace_nodes)
            
            # 1. 计算调用链深度
            try:
                max_depth = 0
                root_nodes = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0]
                
                if root_nodes:
                    for root in root_nodes:
                        try:
                            paths = nx.single_source_shortest_path_length(subgraph, root)
                            if paths:
                                max_depth = max(max_depth, max(paths.values()) + 1)
                        except:
                            continue
                else:
                    # 如果没有明确的根节点，使用节点数作为深度
                    max_depth = min(node_count, 10)  # 限制最大深度
                    
            except:
                max_depth = min(node_count, 10)
            
            # 2. 计算扇出度（并行度）
            out_degrees = [subgraph.out_degree(n) for n in subgraph.nodes()]
            max_fan_out = max(out_degrees, default=1)
            avg_fan_out = sum(out_degrees) / max(len(out_degrees), 1)
            
            # 3. 计算服务多样性
            services = set()
            for node in trace_nodes:
                service = G.nodes[node].get('service', 'unknown')
                services.add(service)
            service_diversity = len(services)
            
            # 4. 检测循环
            try:
                cycles = list(nx.simple_cycles(subgraph))
                cycle_count = len(cycles)
            except:
                cycle_count = 0
            
            # 5. 计算网络密度
            possible_edges = node_count * (node_count - 1)
            actual_edges = subgraph.number_of_edges()
            density = actual_edges / max(possible_edges, 1) if possible_edges > 0 else 0
            
            # 6. 优化的复杂度计算公式
            complexity_score = (
                max_depth * 0.25 +          # 调用链深度
                max_fan_out * 0.20 +        # 最大并行度
                avg_fan_out * 0.15 +        # 平均扇出度
                service_diversity * 0.15 +   # 服务多样性
                cycle_count * 0.15 +        # 循环复杂度
                density * 10 * 0.10         # 网络密度(放大10倍)
            )
            
            return complexity_score
            
        except Exception as e:
            self.logger.debug(f"计算结构复杂度失败: {e}")
            return 0.0

    def calculate_dynamic_thresholds(self, latency_values: List[float], complexity_values: List[float]):
        """基于数据分布动态计算阈值"""
        if not latency_values or not complexity_values:
            self.logger.warning("数据不足，使用静态阈值")
            return
        
        # 计算延迟阈值（使用分位数）
        latency_p33 = np.percentile(latency_values, 33)
        latency_p67 = np.percentile(latency_values, 67)
        
        # 计算复杂度阈值
        complexity_p33 = np.percentile(complexity_values, 33)
        complexity_p67 = np.percentile(complexity_values, 67)
        
        self.dynamic_latency_thresholds = {
            "normal": latency_p33,
            "medium": latency_p67
        }
        
        self.dynamic_structure_thresholds = {
            "simple": complexity_p33,
            "medium": complexity_p67
        }
        
        self.logger.info(f"动态延迟阈值: 正常<{latency_p33:.0f}ms, 中等<{latency_p67:.0f}ms")
        self.logger.info(f"动态复杂度阈值: 简单<{complexity_p33:.2f}, 中等<{complexity_p67:.2f}")

    def calculate_graph_latency_label(self, critical_path_latency: float) -> int:
        """计算图延迟标签（支持动态阈值）"""
        thresholds = self.dynamic_latency_thresholds if self.use_dynamic_thresholds and self.dynamic_latency_thresholds else self.latency_thresholds
        
        if critical_path_latency < thresholds["normal"]:
            return 0  # 正常
        elif critical_path_latency < thresholds["medium"]:
            return 1  # 中等延迟
        else:
            return 2  # 高延迟

    def calculate_graph_structure_label(self, complexity_score: float) -> int:
        """计算图结构标签（支持动态阈值）"""
        thresholds = self.dynamic_structure_thresholds if self.use_dynamic_thresholds and self.dynamic_structure_thresholds else self.structure_thresholds
        
        if complexity_score < thresholds["simple"]:
            return 0  # 简单结构
        elif complexity_score < thresholds["medium"]:
            return 1  # 中等复杂结构
        else:
            return 2  # 复杂结构

    def analyze_batch_data(self, spans_data: List[Dict], mapping_data: Dict) -> List[Dict]:
        """分析一批数据并添加图特征"""
        if not spans_data:
            return []
        
        # 使用映射表恢复真实服务名
        reverse_service_mapping = mapping_data.get('reverse_service_mapping', {})
        
        for span in spans_data:
            service_id = span.get('serviceName', 0)
            if str(service_id) in reverse_service_mapping:
                span['_service_name'] = reverse_service_mapping[str(service_id)]
        
        # 构建调用图
        G = self.build_call_graph(spans_data)
        
        # 按 trace 分组计算图特征
        traces = defaultdict(list)
        for span in spans_data:
            trace_id = span['_trace_id']
            traces[trace_id].append(span)
        
        # 第一轮：收集所有数据用于动态阈值计算
        if self.use_dynamic_thresholds:
            batch_latency_values = []
            batch_complexity_values = []
            
            for trace_id in traces.keys():
                critical_path_latency = self.calculate_critical_path_latency(G, trace_id)
                complexity_score = self.calculate_structure_complexity(G, trace_id)
                
                if critical_path_latency > 0:
                    batch_latency_values.append(critical_path_latency)
                if complexity_score > 0:
                    batch_complexity_values.append(complexity_score)
            
            # 累积到全局统计
            self.analysis_stats["latency_values"].extend(batch_latency_values)
            self.analysis_stats["complexity_values"].extend(batch_complexity_values)
        
        # 为每个 trace 计算图特征
        trace_features = {}
        
        for trace_id, spans in traces.items():
            try:
                # 计算关键路径延迟
                critical_path_latency = self.calculate_critical_path_latency(G, trace_id)
                
                # 计算结构复杂度
                complexity_score = self.calculate_structure_complexity(G, trace_id)
                
                # 计算标签
                graph_latency_label = self.calculate_graph_latency_label(critical_path_latency)
                graph_structure_label = self.calculate_graph_structure_label(complexity_score)
                
                # 更新统计
                self.analysis_stats["total_traces"] += 1
                self.analysis_stats["latency_distribution"][str(graph_latency_label)] += 1
                self.analysis_stats["structure_distribution"][str(graph_structure_label)] += 1
                
                trace_features[trace_id] = {
                    'graph_latency_label': graph_latency_label,
                    'graph_structure_label': graph_structure_label,
                    'critical_path_latency': critical_path_latency,
                    'complexity_score': complexity_score
                }
                
            except Exception as e:
                self.logger.debug(f"分析 trace {trace_id} 失败: {e}")
                trace_features[trace_id] = {
                    'graph_latency_label': 0,
                    'graph_structure_label': 0,
                    'critical_path_latency': 0.0,
                    'complexity_score': 0.0
                }
        
        # 更新 span 数据，添加图特征
        updated_spans = []
        for span in spans_data:
            trace_id = span['_trace_id']
            features = trace_features.get(trace_id, {
                'graph_latency_label': 0,
                'graph_structure_label': 0
            })
            
            # 添加新字段
            span['graphLatencyLabel'] = features['graph_latency_label']
            span['graphStructureLabel'] = features['graph_structure_label']
            
            updated_spans.append(span)
        
        return updated_spans

    def save_enhanced_csv(self, spans_data: List[Dict], output_path: str):
        """保存扩展后的14列CSV文件"""
        if not spans_data:
            return
        
        # 14列字段顺序
        fieldnames = [
            "traceIdHigh", "traceIdLow", "parentSpanId", "spanId", 
            "startTime", "duration", "nanosecond", "DBhash", "status",
            "operationName", "serviceName", "nodeLatencyLabel",
            "graphLatencyLabel", "graphStructureLabel"  # 新增字段
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for span in spans_data:
                row = {field: span.get(field, 0) for field in fieldnames}
                writer.writerow(row)

    def process_date_directory(self, date_dir: str) -> bool:
        """处理某个日期目录的所有数据"""
        csv_dir = os.path.join(date_dir, "csv")
        
        if not os.path.exists(csv_dir):
            self.logger.warning(f"CSV 目录不存在: {csv_dir}")
            return False
        
        # 加载映射数据
        mapping_data = self.load_mapping_data(date_dir)
        
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            self.logger.warning(f"未找到 CSV 文件: {csv_dir}")
            return False
        
        # 创建enhanced目录存放14列文件
        enhanced_dir = os.path.join(date_dir, "csv_enhanced")
        os.makedirs(enhanced_dir, exist_ok=True)
        
        total_files = len(csv_files)
        self.logger.info(f"开始处理 {total_files} 个CSV文件...")
        
        # 重置统计信息
        self.analysis_stats = {
            "total_traces": 0,
            "latency_distribution": {"0": 0, "1": 0, "2": 0},
            "structure_distribution": {"0": 0, "1": 0, "2": 0},
            "latency_values": [],
            "complexity_values": []
        }
        
        processed_count = 0
        enhanced_count = 0
        
        # 如果使用动态阈值，先处理部分文件收集数据
        if self.use_dynamic_thresholds and total_files > 5:
            sample_count = min(5, total_files // 3)  # 使用1/3的文件作为样本
            self.logger.info(f"动态阈值模式：先处理 {sample_count} 个样本文件...")
            
            for i, csv_file in enumerate(sorted(csv_files)[:sample_count]):
                csv_path = os.path.join(csv_dir, csv_file)
                spans_data = self.load_csv_data(csv_path)
                if spans_data:
                    self.analyze_batch_data(spans_data, mapping_data)
            
            # 计算动态阈值
            if self.analysis_stats["latency_values"] and self.analysis_stats["complexity_values"]:
                self.calculate_dynamic_thresholds(
                    self.analysis_stats["latency_values"],
                    self.analysis_stats["complexity_values"]
                )
            
            # 重置统计以重新开始
            self.analysis_stats = {
                "total_traces": 0,
                "latency_distribution": {"0": 0, "1": 0, "2": 0},
                "structure_distribution": {"0": 0, "1": 0, "2": 0},
                "latency_values": [],
                "complexity_values": []
            }
        
        # 处理所有文件
        for csv_file in sorted(csv_files):
            try:
                csv_path = os.path.join(csv_dir, csv_file)
                
                # 加载12列数据
                spans_data = self.load_csv_data(csv_path)
                if not spans_data:
                    continue
                
                # 进行图分析，添加图特征
                enhanced_spans = self.analyze_batch_data(spans_data, mapping_data)
                
                # 保存14列CSV文件
                enhanced_path = os.path.join(enhanced_dir, csv_file)
                self.save_enhanced_csv(enhanced_spans, enhanced_path)
                
                processed_count += 1
                enhanced_count += len(enhanced_spans)
                
                if processed_count % 10 == 0:
                    self.logger.info(f"已处理 {processed_count}/{total_files} 个文件")
                
            except Exception as e:
                self.logger.error(f"处理文件 {csv_file} 失败: {e}")
                continue
        
        # 输出详细统计信息
        self._print_analysis_stats(date_dir)
        
        self.logger.info(f"完成处理: {processed_count}/{total_files} 个文件")
        self.logger.info(f"增强了 {enhanced_count} 条span记录")
        self.logger.info(f"14列CSV文件保存在: {enhanced_dir}")
        return processed_count > 0

    def _print_analysis_stats(self, date_dir: str):
        """输出分析统计信息"""
        stats = self.analysis_stats
        total_traces = stats["total_traces"]
        
        if total_traces == 0:
            self.logger.warning("没有处理到任何trace数据")
            return
        
        self.logger.info("=== 图分析统计信息 ===")
        self.logger.info(f"处理的trace总数: {total_traces}")
        
        # 延迟标签分布
        lat_dist = stats["latency_distribution"]
        self.logger.info("延迟标签分布:")
        self.logger.info(f"  正常(0): {lat_dist['0']} ({lat_dist['0']/total_traces*100:.1f}%)")
        self.logger.info(f"  中等(1): {lat_dist['1']} ({lat_dist['1']/total_traces*100:.1f}%)")
        self.logger.info(f"  高延迟(2): {lat_dist['2']} ({lat_dist['2']/total_traces*100:.1f}%)")
        
        # 结构标签分布
        struct_dist = stats["structure_distribution"]
        self.logger.info("结构标签分布:")
        self.logger.info(f"  简单(0): {struct_dist['0']} ({struct_dist['0']/total_traces*100:.1f}%)")
        self.logger.info(f"  中等(1): {struct_dist['1']} ({struct_dist['1']/total_traces*100:.1f}%)")
        self.logger.info(f"  复杂(2): {struct_dist['2']} ({struct_dist['2']/total_traces*100:.1f}%)")
        
        # 数值分布统计
        if stats["latency_values"]:
            lat_values = stats["latency_values"]
            self.logger.info(f"延迟分布: min={min(lat_values):.0f}ms, "
                           f"median={np.median(lat_values):.0f}ms, "
                           f"max={max(lat_values):.0f}ms")
        
        if stats["complexity_values"]:
            comp_values = stats["complexity_values"]
            self.logger.info(f"复杂度分布: min={min(comp_values):.2f}, "
                           f"median={np.median(comp_values):.2f}, "
                           f"max={max(comp_values):.2f}")
        
        # 保存统计信息到文件
        stats_file = os.path.join(date_dir, "graph_analysis_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "analysis_date": datetime.now().isoformat(),
                "total_traces": total_traces,
                "latency_distribution": lat_dist,
                "structure_distribution": struct_dist,
                "thresholds_used": {
                    "latency": self.dynamic_latency_thresholds if self.use_dynamic_thresholds and self.dynamic_latency_thresholds else self.latency_thresholds,
                    "structure": self.dynamic_structure_thresholds if self.use_dynamic_thresholds and self.dynamic_structure_thresholds else self.structure_thresholds
                },
                "dynamic_thresholds_enabled": self.use_dynamic_thresholds
            }, f, indent=2, ensure_ascii=False)

    def run_post_processing(self, specific_date: str = None) -> bool:
        """运行后处理"""
        if not os.path.exists(self.data_dir):
            self.logger.error(f"数据目录不存在: {self.data_dir}")
            return False
        
        if specific_date:
            # 处理特定日期
            date_dir = os.path.join(self.data_dir, specific_date)
            if not os.path.exists(date_dir):
                self.logger.error(f"日期目录不存在: {date_dir}")
                return False
            
            self.logger.info(f"后处理日期: {specific_date}")
            return self.process_date_directory(date_dir)
        else:
            # 处理所有日期
            date_dirs = [d for d in os.listdir(self.data_dir) 
                        if os.path.isdir(os.path.join(self.data_dir, d)) 
                        and d.count('-') == 2]  # YYYY-MM-DD 格式
            
            if not date_dirs:
                self.logger.error("未找到日期目录")
                return False
            
            self.logger.info(f"发现 {len(date_dirs)} 个日期目录")
            
            success_count = 0
            for date_dir in sorted(date_dirs):
                self.logger.info(f"后处理日期: {date_dir}")
                date_path = os.path.join(self.data_dir, date_dir)
                if self.process_date_directory(date_path):
                    success_count += 1
            
            self.logger.info(f"成功处理 {success_count}/{len(date_dirs)} 个日期")
            return success_count > 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Ticket 调用链图后处理器 - 扩展12列为14列",
        epilog="""
使用示例:
  python graph_post_processor.py                     # 处理所有日期（动态阈值）
  python graph_post_processor.py --date 2025-06-18  # 处理特定日期
  python graph_post_processor.py --static-thresholds # 使用静态阈值
  python graph_post_processor.py --data-dir ./traces # 指定数据目录
  
输出说明:
  原始12列文件: trace_output/YYYY-MM-DD/csv/
  增强14列文件: trace_output/YYYY-MM-DD/csv_enhanced/
  分析统计: trace_output/YYYY-MM-DD/graph_analysis_stats.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=str, 
                       help="处理特定日期 (YYYY-MM-DD)")
    parser.add_argument("--data-dir", type=str, default="trace_output",
                       help="数据目录路径，默认: trace_output")
    parser.add_argument("--static-thresholds", action="store_true",
                       help="使用静态阈值而不是动态阈值")
    
    args = parser.parse_args()
    
    try:
        # 检查依赖
        import networkx
        import pandas
        import numpy
    except ImportError as e:
        print(f"缺少依赖包: {e}")
        print("请运行: pip install networkx pandas numpy")
        return 1
    
    use_dynamic = not args.static_thresholds
    processor = TraceGraphPostProcessor(
        data_dir=args.data_dir, 
        use_dynamic_thresholds=use_dynamic
    )
    
    try:
        success = processor.run_post_processing(specific_date=args.date)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("后处理已中断")
        return 0
    except Exception as e:
        print(f"后处理失败: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())