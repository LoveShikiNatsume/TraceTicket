#!/usr/bin/env python3
"""
将异常数据转换为CSV格式
基于之前生成的异常数据，转换为与原始时间偏移代码相同格式的输出
"""

import json
import pandas as pd
import numpy as np
import random
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import os

def load_anomaly_data(anomaly_data_dir):
    """加载所有异常数据"""
    print("📂 加载异常数据...")
    
    all_spans = []
    
    # 遍历所有时间段目录
    for period_dir in os.listdir(anomaly_data_dir):
        period_path = Path(anomaly_data_dir) / period_dir
        
        if not period_path.is_dir() or not period_dir.startswith('TT.'):
            continue
            
        time_period = period_dir.replace('TT.', '')
        print(f"  处理时间段: {time_period}")
        
        # 加载所有类型的数据文件
        data_files = [
            'all_spans_with_anomalies.json'  # 使用混合的数据文件
        ]
        
        for data_file in data_files:
            file_path = period_path / data_file
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    traces_data = json.load(f)
                
                # 处理每个trace
                for trace in traces_data:
                    trace_id = trace['traceID']
                    processes = trace['processes']
                    anomaly_type = trace.get('anomaly_type', 'normal')
                    
                    for span in trace['spans']:
                        # 获取基本信息
                        process_id = span['processID']
                        service_name = processes[process_id]['serviceName']
                        operation_name = span['operationName']
                        
                        # 获取状态码
                        status_code = '200'
                        for tag in span.get('tags', []):
                            if tag['key'] in ['http.status_code', 'error']:
                                if tag['key'] == 'error' and tag['value']:
                                    status_code = 'error'
                                elif tag['key'] == 'http.status_code':
                                    status_code = str(tag['value'])
                                break
                        
                        # 根据异常类型设置标签
                        labels = {
                            'nodeLatencyLabel': 0,
                            'graphLatencyLabel': 0,
                            'graphStructureLabel': 0
                        }
                        
                        if anomaly_type == 'structure':
                            labels['graphStructureLabel'] = 1
                        elif anomaly_type == 'time':
                            labels['nodeLatencyLabel'] = 1
                            labels['graphLatencyLabel'] = 1
                        # normal 类型保持全0
                        
                        # 创建span记录
                        span_record = {
                            'traceID': trace_id,
                            'spanID': span['spanID'],
                            'parentSpanID': span.get('parentSpanID', ''),
                            'operationName': operation_name,
                            'serviceName': service_name,
                            'startTime': span['startTime'],
                            'duration': span['duration'],
                            'status': status_code,
                            'time_period': time_period,
                            'anomaly_type': anomaly_type,
                            **labels
                        }
                        
                        all_spans.append(span_record)
                        
            except Exception as e:
                print(f"    ❌ 处理文件 {data_file} 失败: {e}")
    
    print(f"✅ 总计加载 {len(all_spans)} 个spans")
    return all_spans

def convert_to_csv_with_labels(spans, output_dir):
    """转换为CSV格式并保存"""
    print("📊 转换为CSV格式...")
    
    # 生成ID映射
    operation_names = set(s['operationName'] for s in spans)
    service_names = set(s['serviceName'] for s in spans)
    
    operation_name_to_id = {op: i+1 for i, op in enumerate(sorted(operation_names))}
    service_name_to_id = {svc: i+1 for i, svc in enumerate(sorted(service_names))}
    
    # 转换数据
    csv_records = []
    trace_id_mapping = {}
    span_id_mapping = {}
    
    for span in spans:
        # ID映射
        if span['traceID'] not in trace_id_mapping:
            hash_obj = hashlib.md5(str(span['traceID']).encode())
            hash_int = int(hash_obj.hexdigest()[:16], 16)
            trace_id_high = (hash_int >> 32) & 0x7FFFFFFFFFFFFFFF
            trace_id_low = hash_int & 0x7FFFFFFFFFFFFFFF
            trace_id_mapping[span['traceID']] = (trace_id_high, trace_id_low)
        
        if span['spanID'] not in span_id_mapping:
            span_id_mapping[span['spanID']] = random.randint(-2**31, 2**31-1)
        
        trace_id_high, trace_id_low = trace_id_mapping[span['traceID']]
        span_id = span_id_mapping[span['spanID']]
        
        # 处理父span ID
        parent_span_id = 0
        if span['parentSpanID']:
            if span['parentSpanID'] not in span_id_mapping:
                span_id_mapping[span['parentSpanID']] = random.randint(-2**31, 2**31-1)
            parent_span_id = span_id_mapping[span['parentSpanID']]
        
        # 时间转换
        start_time_sec = span['startTime'] / 1000000
        start_time_str = datetime.fromtimestamp(start_time_sec).strftime("%Y-%m-%d %H:%M:%S")
        duration_ms = max(1, int(span['duration'] / 1000))
        nanosecond = (span['startTime'] % 1000000) * 1000
        
        # 状态处理
        status_val = 0 if span['status'] == '200' else 1
        
        # CSV记录
        csv_record = {
            'traceIdHigh': trace_id_high,
            'traceIdLow': trace_id_low,
            'parentSpanId': parent_span_id,
            'spanId': span_id,
            'startTime': start_time_str,
            'duration': duration_ms,
            'nanosecond': int(nanosecond),
            'DBhash': 0,
            'status': status_val,
            'operationName': operation_name_to_id.get(span['operationName'], 0),
            'serviceName': service_name_to_id.get(span['serviceName'], 0),
            'nodeLatencyLabel': span['nodeLatencyLabel'],
            'graphLatencyLabel': span['graphLatencyLabel'],
            'graphStructureLabel': span['graphStructureLabel']
        }
        
        csv_records.append(csv_record)
    
    df = pd.DataFrame(csv_records)
    
    # 分割数据集
    print("🔄 分割数据集...")
    
    # 按trace分组
    trace_groups = df.groupby(['traceIdHigh', 'traceIdLow'])
    
    # 分离正常和异常traces
    normal_traces = []
    anomaly_traces = []
    
    for trace_id, group in trace_groups:
        has_anomaly = (group['nodeLatencyLabel'].sum() > 0 or 
                      group['graphLatencyLabel'].sum() > 0 or 
                      group['graphStructureLabel'].sum() > 0)
        
        if has_anomaly:
            anomaly_traces.append(trace_id)
        else:
            normal_traces.append(trace_id)
    
    print(f"正常traces: {len(normal_traces)}, 异常traces: {len(anomaly_traces)}")
    
    # 分别分割
    random.shuffle(normal_traces)
    random.shuffle(anomaly_traces)
    
    # 正常traces分割 (70%, 15%, 15%)
    normal_total = len(normal_traces)
    normal_train_size = int(0.7 * normal_total)
    normal_val_size = int(0.15 * normal_total)
    
    normal_train = normal_traces[:normal_train_size]
    normal_val = normal_traces[normal_train_size:normal_train_size + normal_val_size]
    normal_test = normal_traces[normal_train_size + normal_val_size:]
    
    # 异常traces分割
    anomaly_total = len(anomaly_traces)
    anomaly_train_size = int(0.7 * anomaly_total)
    anomaly_val_size = int(0.15 * anomaly_total)
    
    anomaly_train = anomaly_traces[:anomaly_train_size]
    anomaly_val = anomaly_traces[anomaly_train_size:anomaly_train_size + anomaly_val_size]
    anomaly_test = anomaly_traces[anomaly_train_size + anomaly_val_size:]
    
    # 合并
    train_traces = normal_train + anomaly_train
    val_traces = normal_val + anomaly_val
    test_traces = normal_test + anomaly_test
    
    # 创建数据框
    train_df = df[df.set_index(['traceIdHigh', 'traceIdLow']).index.isin(train_traces)].reset_index(drop=True)
    val_df = df[df.set_index(['traceIdHigh', 'traceIdLow']).index.isin(val_traces)].reset_index(drop=True)
    test_df = df[df.set_index(['traceIdHigh', 'traceIdLow']).index.isin(test_traces)].reset_index(drop=True)
    
    # 保存文件
    Path(output_dir).mkdir(exist_ok=True)

    # 训练集和验证集删除异常标签列
    train_df_clean = train_df.drop(columns=['nodeLatencyLabel', 'graphLatencyLabel', 'graphStructureLabel'])
    val_df_clean = val_df.drop(columns=['nodeLatencyLabel', 'graphLatencyLabel', 'graphStructureLabel'])

    train_df_clean.to_csv(Path(output_dir) / "train.csv", index=False)
    val_df_clean.to_csv(Path(output_dir) / "val.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test.csv", index=False)  # 测试集保留标签列
    
    # 统计信息
    print("✅ 异常数据转换完成!")
    
    # 统计异常类型分布
    print("\n📊 异常类型统计:")
    anomaly_stats = defaultdict(int)
    for span in spans:
        anomaly_stats[span['anomaly_type']] += 1
    
    for anomaly_type, count in anomaly_stats.items():
        print(f"  {anomaly_type}: {count} spans")
    
    # 统计各数据集
    for split_name, split_df in [("训练", train_df), ("验证", val_df), ("测试", test_df)]:
        node_anomalies = split_df['nodeLatencyLabel'].sum()
        graph_lat_anomalies = split_df['graphLatencyLabel'].sum()
        graph_struct_anomalies = split_df['graphStructureLabel'].sum()
        total_spans = len(split_df)
        
        max_anomalies = max(node_anomalies, graph_lat_anomalies, graph_struct_anomalies)
        anomaly_ratio = max_anomalies / total_spans * 100 if total_spans > 0 else 0
        
        print(f"{split_name}集: {total_spans} spans, 异常率: {anomaly_ratio:.2f}%")
        print(f"  - 节点延迟: {node_anomalies}, 图延迟: {graph_lat_anomalies}, 图结构: {graph_struct_anomalies}")
    
    return operation_name_to_id, service_name_to_id

def save_yaml_files(operation_name_to_id, service_name_to_id, output_dir):
    """保存YAML配置文件"""
    print("📝 保存YAML配置文件...")
    
    id_manager_dir = Path(output_dir) / "id_manager"
    id_manager_dir.mkdir(parents=True, exist_ok=True)
    
    # operation_id.yml
    operation_file = id_manager_dir / "operation_id.yml"
    with open(operation_file, 'w', encoding='utf-8') as f:
        f.write("? ''\n: 0\n")
        
        global_id = 1
        service_ids = sorted(service_name_to_id.values())
        operation_ids = sorted(operation_name_to_id.values())
        
        for service_id in service_ids:
            if service_id > 0:
                for operation_id in operation_ids:
                    if operation_id > 0:
                        composite_key = f"{service_id}/{operation_id}"
                        f.write(f"{composite_key}: {global_id}\n")
                        global_id += 1
    
    # service_id.yml
    service_file = id_manager_dir / "service_id.yml"
    with open(service_file, 'w', encoding='utf-8') as f:
        f.write("? ''\n: 0\n")
        for svc_id in sorted(service_name_to_id.values()):
            if svc_id != 0:
                f.write(f"'{svc_id}': {svc_id}\n")
    
    # status_id.yml
    status_file = id_manager_dir / "status_id.yml"
    with open(status_file, 'w', encoding='utf-8') as f:
        f.write("? ''\n: 0\n")
        f.write("'0': 1\n")
        f.write("'1': 2\n")
        f.write("'200': 3\n")
    
    # latency_range.yml (简化版本)
    latency_file = id_manager_dir / "latency_range.yml"
    with open(latency_file, 'w', encoding='utf-8') as f:
        global_id = 1
        service_ids = sorted(service_name_to_id.values())
        operation_ids = sorted(operation_name_to_id.values())
        
        for service_id in service_ids:
            if service_id > 0:
                for operation_id in operation_ids:
                    if operation_id > 0:
                        f.write(f"{global_id}:\n")
                        f.write(f"  mean: 10.0\n")
                        f.write(f"  std: 5.0\n")
                        f.write(f"  p99: 50.0\n")
                        global_id += 1
    
    print(f"✅ YAML文件已保存到: {id_manager_dir}")

def main():
    """主函数"""
    # 配置路径
    anomaly_data_dir = "/home/fuxian/tracevae/TT_Dataset/TT_Dataset/anomaly_data"  # 之前生成的异常数据目录
    output_dir = "/home/fuxian/tracevae/TT_Dataset/TT_Dataset/convert_structure_anomaly_data"  # 输出CSV目录
    
    print("🔄 开始转换异常数据为CSV格式...")
    
    try:
        # 1. 加载异常数据
        spans = load_anomaly_data(anomaly_data_dir)
        
        if not spans:
            print("❌ 没有加载到任何spans数据")
            return
        
        # 2. 转换为CSV并保存
        operation_name_to_id, service_name_to_id = convert_to_csv_with_labels(spans, output_dir)
        
        # 3. 保存YAML文件
        save_yaml_files(operation_name_to_id, service_name_to_id, output_dir)
        
        print(f"\n🎯 CSV数据集已保存到: {output_dir}")
        print("📋 生成的文件:")
        print(f"  ├── train.csv  (训练集，无标签列)")
        print(f"  ├── val.csv    (验证集，无标签列)")
        print(f"  ├── test.csv   (测试集，含标签列)")
        print(f"  └── id_manager/ (YAML配置文件)")
        print("\n📋 下一步操作:")
        print(f"  bash test.sh results/train/models/final.pt {output_dir}")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    random.seed(42)
    main()