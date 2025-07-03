#!/usr/bin/env python3
"""
生成结构异常和时间异常数据
基于 spans.json 和 fault.json 生成异常样本
"""

import json
import random
import copy
import os
from pathlib import Path
from collections import defaultdict, deque

class AnomalyGenerator:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, time_period):
        """加载指定时间段的数据"""
        spans_file = f"{self.data_dir}/TT.{time_period}/spans.json"
        fault_file = f"{self.data_dir}/TT.fault-{time_period}.json"
        
        spans_data = []
        fault_data = []
        
        if os.path.exists(spans_file):
            with open(spans_file, 'r') as f:
                spans_data = json.load(f)
        
        if os.path.exists(fault_file):
            with open(fault_file, 'r') as f:
                fault_data = json.load(f)
        
        return spans_data, fault_data
    
    def build_trace_tree(self, trace):
        """构建追踪树结构"""
        spans = trace['spans']
        
        # 构建 span 映射
        span_map = {span['spanID']: span for span in spans}
        
        # 构建父子关系
        children = defaultdict(list)
        root_spans = []
        
        for span in spans:
            parent_id = span.get('parentSpanID', '')
            if parent_id and parent_id in span_map:
                children[parent_id].append(span['spanID'])
            else:
                root_spans.append(span['spanID'])
        
        return span_map, children, root_spans
    
    def generate_structure_anomaly(self, trace, drop_probability=None):
        """生成结构异常 - 随机丢弃节点及其子节点"""
        if drop_probability is None:
            drop_probability = random.uniform(0.1, 0.5)
        
        # 深拷贝原始追踪
        anomaly_trace = copy.deepcopy(trace)
        
        # 构建树结构
        span_map, children, root_spans = self.build_trace_tree(trace)
        
        # 选择要丢弃的节点
        all_span_ids = list(span_map.keys())
        num_to_drop = max(1, int(len(all_span_ids) * drop_probability))
        
        # 随机选择节点进行丢弃
        spans_to_drop = set()
        
        for _ in range(num_to_drop):
            if len(all_span_ids) > len(spans_to_drop):
                # 选择一个未被丢弃的节点
                available_spans = [sid for sid in all_span_ids if sid not in spans_to_drop]
                if available_spans:
                    selected_span = random.choice(available_spans)
                    
                    # 丢弃该节点及其所有子节点
                    to_drop = self._get_subtree_nodes(selected_span, children)
                    spans_to_drop.update(to_drop)
        
        # 从追踪中移除丢弃的节点
        anomaly_trace['spans'] = [
            span for span in anomaly_trace['spans'] 
            if span['spanID'] not in spans_to_drop
        ]
        
        # 添加异常标记
        anomaly_trace['anomaly_type'] = 'structure'
        anomaly_trace['dropped_spans'] = list(spans_to_drop)
        anomaly_trace['drop_probability'] = drop_probability
        
        return anomaly_trace
    
    def _get_subtree_nodes(self, span_id, children):
        """获取以指定节点为根的所有子树节点"""
        result = {span_id}
        queue = deque([span_id])
        
        while queue:
            current = queue.popleft()
            for child in children.get(current, []):
                if child not in result:
                    result.add(child)
                    queue.append(child)
        
        return result
    
    def generate_time_anomaly(self, trace, latency_multiplier=None):
        """生成时间异常 - 增加节点耗时并传播到父节点"""
        if latency_multiplier is None:
            latency_multiplier = random.uniform(2, 10)
        
        # 深拷贝原始追踪
        anomaly_trace = copy.deepcopy(trace)
        
        # 构建树结构
        span_map, children, root_spans = self.build_trace_tree(trace)
        
        # 构建父节点映射
        parents = {}
        for span in trace['spans']:
            parent_id = span.get('parentSpanID', '')
            if parent_id and parent_id in span_map:
                parents[span['spanID']] = parent_id
        
        # 随机选择一个节点增加延迟
        all_span_ids = list(span_map.keys())
        if not all_span_ids:
            return anomaly_trace
        
        selected_span_id = random.choice(all_span_ids)
        
        # 找到对应的span并修改其duration
        affected_spans = []
        
        for span in anomaly_trace['spans']:
            if span['spanID'] == selected_span_id:
                original_duration = span['duration']
                additional_duration = int(original_duration * (latency_multiplier - 1))
                span['duration'] += additional_duration
                affected_spans.append(span['spanID'])
                
                # 将增加的时间传播到所有父节点
                current_span_id = span['spanID']
                while current_span_id in parents:
                    parent_id = parents[current_span_id]
                    
                    # 找到父节点并增加其duration
                    for parent_span in anomaly_trace['spans']:
                        if parent_span['spanID'] == parent_id:
                            parent_span['duration'] += additional_duration
                            affected_spans.append(parent_span['spanID'])
                            break
                    
                    current_span_id = parent_id
                
                break
        
        # 添加异常标记
        anomaly_trace['anomaly_type'] = 'time'
        anomaly_trace['affected_span'] = selected_span_id
        anomaly_trace['affected_spans'] = affected_spans
        anomaly_trace['latency_multiplier'] = latency_multiplier
        
        return anomaly_trace
    
    def generate_anomalies_for_period(self, time_period, structure_ratio=0.3, time_ratio=0.3):
        """为指定时间段生成异常数据"""
        print(f"📅 处理时间段: {time_period}")
        
        # 加载数据
        spans_data, fault_data = self.load_data(time_period)
        
        if not spans_data:
            print(f"  ⚠️  没有找到spans数据")
            return
        
        # 生成异常数据
        normal_traces = []
        structure_anomalies = []
        time_anomalies = []
        
        num_traces = len(spans_data)
        num_structure = int(num_traces * structure_ratio)
        num_time = int(num_traces * time_ratio)
        
        print(f"  总traces: {num_traces}")
        print(f"  生成结构异常: {num_structure}")
        print(f"  生成时间异常: {num_time}")
        
        # 随机选择traces生成异常
        trace_indices = list(range(num_traces))
        random.shuffle(trace_indices)
        
        # 生成结构异常
        structure_indices = trace_indices[:num_structure]
        for idx in structure_indices:
            anomaly_trace = self.generate_structure_anomaly(spans_data[idx])
            structure_anomalies.append(anomaly_trace)
        
        # 生成时间异常
        time_indices = trace_indices[num_structure:num_structure + num_time]
        for idx in time_indices:
            anomaly_trace = self.generate_time_anomaly(spans_data[idx])
            time_anomalies.append(anomaly_trace)
        
        # 剩余的作为正常数据
        normal_indices = trace_indices[num_structure + num_time:]
        for idx in normal_indices:
            normal_trace = copy.deepcopy(spans_data[idx])
            normal_trace['anomaly_type'] = 'normal'
            normal_traces.append(normal_trace)
        
        # 保存结果
        self.save_anomaly_data(time_period, normal_traces, structure_anomalies, time_anomalies)
        
        return {
            'normal': len(normal_traces),
            'structure': len(structure_anomalies),
            'time': len(time_anomalies)
        }
    
    def save_anomaly_data(self, time_period, normal_traces, structure_anomalies, time_anomalies):
        """保存异常数据"""
        period_dir = Path(self.output_dir) / f"TT.{time_period}"
        period_dir.mkdir(exist_ok=True)
        
        # 保存正常数据
        with open(period_dir / "normal_spans.json", 'w') as f:
            json.dump(normal_traces, f, indent=2)
        
        # 保存结构异常数据
        with open(period_dir / "structure_anomaly_spans.json", 'w') as f:
            json.dump(structure_anomalies, f, indent=2)
        
        # 保存时间异常数据
        with open(period_dir / "time_anomaly_spans.json", 'w') as f:
            json.dump(time_anomalies, f, indent=2)
        
        # 保存合并的数据（包含所有类型）
        all_traces = normal_traces + structure_anomalies + time_anomalies
        random.shuffle(all_traces)  # 打乱顺序
        
        with open(period_dir / "all_spans_with_anomalies.json", 'w') as f:
            json.dump(all_traces, f, indent=2)
        
        print(f"  ✅ 数据已保存到: {period_dir}")
    
    def generate_all_periods(self, structure_ratio=0.3, time_ratio=0.3):
        """为所有时间段生成异常数据"""
        print("🚀 开始生成异常数据...")
        
        # 找到所有时间段
        time_periods = []
        for item in os.listdir(self.data_dir):
            if item.startswith('TT.') and os.path.isdir(os.path.join(self.data_dir, item)):
                period = item.replace('TT.', '')
                time_periods.append(period)
        
        time_periods.sort()
        
        total_stats = {
            'normal': 0,
            'structure': 0,
            'time': 0
        }
        
        for period in time_periods:
            stats = self.generate_anomalies_for_period(period, structure_ratio, time_ratio)
            if stats:
                for key in total_stats:
                    total_stats[key] += stats[key]
        
        print(f"\n📊 总体统计:")
        print(f"  正常traces: {total_stats['normal']}")
        print(f"  结构异常traces: {total_stats['structure']}")
        print(f"  时间异常traces: {total_stats['time']}")
        print(f"  总计: {sum(total_stats.values())}")
        
        # 保存统计信息
        with open(Path(self.output_dir) / "generation_stats.json", 'w') as f:
            json.dump(total_stats, f, indent=2)
        
        return total_stats

def main():
    """主函数"""
    # 配置路径
    data_dir = "/home/fuxian/tracevae/TT_Dataset/TT_Dataset/data"
    output_dir = "/home/fuxian/tracevae/TT_Dataset/TT_Dataset/anomaly_data"
    
    # 创建生成器
    generator = AnomalyGenerator(data_dir, output_dir)
    
    # 生成异常数据
    # structure_ratio: 结构异常的比例
    # time_ratio: 时间异常的比例
    # 剩余的traces作为正常数据
    stats = generator.generate_all_periods(
        structure_ratio=0.3,  # 30%的traces作为结构异常
        time_ratio=0.3        # 30%的traces作为时间异常
    )
    
    print(f"\n🎯 异常数据生成完成!")
    print(f"📂 输出目录: {output_dir}")
    print("\n📋 生成的文件结构:")
    print("  TT.{period}/")
    print("    ├── normal_spans.json           # 正常traces")
    print("    ├── structure_anomaly_spans.json # 结构异常traces")
    print("    ├── time_anomaly_spans.json     # 时间异常traces")
    print("    └── all_spans_with_anomalies.json # 所有traces混合")

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    random.seed(42)
    main()