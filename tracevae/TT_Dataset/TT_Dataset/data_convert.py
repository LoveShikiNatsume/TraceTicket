#!/usr/bin/env python3
"""
修复的TT数据集转换脚本 - 正确使用故障注入数据
"""

import json
import pandas as pd
import yaml
import random
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import os

def load_fault_injection_data(data_dir):
    """加载所有故障注入数据"""
    print("加载故障注入数据...")
    
    fault_data = {}
    fault_files = [f for f in os.listdir(data_dir) if f.startswith('TT.fault-') and f.endswith('.json')]
    
    for fault_file in fault_files:
        print(f"  - 加载: {fault_file}")
        with open(os.path.join(data_dir, fault_file)) as f:
            fault_info = json.load(f)
        
        # 提取时间段标识
        time_period = fault_file.replace('TT.fault-', '').replace('.json', '')
        fault_data[time_period] = fault_info
        
        print(f"    实验开始: {datetime.fromtimestamp(fault_info['start'])}")
        print(f"    故障数量: {len(fault_info['faults'])}")
        
        for fault in fault_info['faults']:
            fault_start = datetime.fromtimestamp(fault['start'])
            fault_end = datetime.fromtimestamp(fault['start'] + fault['duration'])
            print(f"      {fault['fault']} on {fault['name']}: {fault_start} - {fault_end}")
    
    return fault_data

def load_json_traces_with_time_period(data_dir):
    """加载所有时间段的trace数据"""
    print("加载trace数据...")
    
    all_traces = []
    spans_files = []
    
    # 查找所有spans.json文件
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            spans_file = os.path.join(item_path, 'spans.json')
            if os.path.exists(spans_file):
                spans_files.append((item, spans_file))
    
    for time_period, spans_file in spans_files:
        print(f"  - 加载: {time_period}/spans.json")
        with open(spans_file) as f:
            traces = json.load(f)
        
        # 为每个trace添加时间段标识
        for trace in traces:
            trace['time_period'] = time_period
        
        all_traces.extend(traces)
        print(f"    trace数量: {len(traces)}")
    
    print(f"总trace数量: {len(all_traces)}")
    return all_traces

def match_spans_with_faults(spans, fault_data):
    """将spans与故障数据匹配，生成真实标签"""
    print("匹配spans与故障数据...")
    
    labeled_spans = []
    
    for span in spans:
        time_period = span.get('time_period', '')
        
        # 默认标签
        labels = {
            'nodeLatencyLabel': 0,
            'graphLatencyLabel': 0, 
            'graphStructureLabel': 0,
            'is_fault_period': 0,
            'fault_types': []
        }
        
        # 检查是否在故障时间段内
        if time_period in fault_data:
            fault_info = fault_data[time_period]
            span_start_time = span['startTime'] / 1000000  # 微秒转秒
            
            labels['is_fault_period'] = 1  # 标记为故障实验期间
            
            # 检查每个故障
            for fault in fault_info['faults']:
                fault_start = fault['start']
                fault_end = fault['start'] + fault['duration']
                
                # 检查span是否在故障时间窗口内
                if fault_start <= span_start_time <= fault_end:
                    fault_type = fault['fault']
                    labels['fault_types'].append(fault_type)
                    
                    # 根据故障类型设置标签
                    if fault_type in ['cpu_load', 'memory_stress']:
                        labels['nodeLatencyLabel'] = 1
                        labels['graphLatencyLabel'] = 1
                    elif fault_type in ['network_delay', 'network_loss']:
                        labels['nodeLatencyLabel'] = 1
                        labels['graphLatencyLabel'] = 1
                    elif fault_type in ['pod_failure', 'container_kill']:
                        labels['graphStructureLabel'] = 1
                    else:
                        # 未知故障类型，标记为延迟异常
                        labels['nodeLatencyLabel'] = 1
                        labels['graphLatencyLabel'] = 1
        
        # 合并span数据和标签
        span_with_labels = {**span, **labels}
        labeled_spans.append(span_with_labels)
    
    # 统计标签分布
    node_anomalies = sum(1 for s in labeled_spans if s['nodeLatencyLabel'] == 1)
    graph_lat_anomalies = sum(1 for s in labeled_spans if s['graphLatencyLabel'] == 1)
    graph_struct_anomalies = sum(1 for s in labeled_spans if s['graphStructureLabel'] == 1)
    fault_period_spans = sum(1 for s in labeled_spans if s['is_fault_period'] == 1)
    
    print(f"标签统计:")
    print(f"  - 故障期间spans: {fault_period_spans}")
    print(f"  - 节点延迟异常: {node_anomalies}")
    print(f"  - 图延迟异常: {graph_lat_anomalies}")
    print(f"  - 图结构异常: {graph_struct_anomalies}")
    print(f"  - 正常spans: {len(labeled_spans) - max(node_anomalies, graph_lat_anomalies, graph_struct_anomalies)}")
    
    return labeled_spans

def extract_spans_from_labeled_traces(labeled_traces):
    """从带标签的traces中提取spans"""
    print("提取spans...")
    
    all_spans = []
    operation_names = set()
    service_names = set()
    status_codes = set()
    
    for trace in labeled_traces:
        trace_id = trace['traceID']
        processes = trace['processes']
        time_period = trace.get('time_period', '')
        
        for span in trace['spans']:
            # 获取服务名
            process_id = span['processID']
            service_name = processes[process_id]['serviceName']
            
            # 获取操作名
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
            
            # 创建span记录（包含标签信息）
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
                # 故障标签信息需要从trace级别传递到span级别
                'nodeLatencyLabel': 0,  # 稍后设置
                'graphLatencyLabel': 0,
                'graphStructureLabel': 0
            }
            
            all_spans.append(span_record)
            operation_names.add(operation_name)
            service_names.add(service_name)
            status_codes.add(status_code)
    
    return all_spans, operation_names, service_names, status_codes

def apply_fault_labels_to_spans(spans, fault_data):
    """为spans应用故障标签"""
    print("应用故障标签到spans...")
    
    for span in spans:
        time_period = span['time_period']
        
        if time_period in fault_data:
            fault_info = fault_data[time_period]
            span_start_time = span['startTime'] / 1000000
            
            # 检查每个故障
            for fault in fault_info['faults']:
                fault_start = fault['start']
                fault_end = fault['start'] + fault['duration']
                
                if fault_start <= span_start_time <= fault_end:
                    fault_type = fault['fault']
                    
                    if fault_type in ['cpu_load', 'memory_stress', 'network_delay', 'network_loss']:
                        span['nodeLatencyLabel'] = 1
                        span['graphLatencyLabel'] = 1
                    elif fault_type in ['pod_failure', 'container_kill']:
                        span['graphStructureLabel'] = 1
    
    return spans

# 保持其他函数不变，但修改main函数
def main():
    """修复的主函数"""
    data_dir = "/home/fuxian/tracevae/TT_Dataset/TT_Dataset/data"
    output_dir = "/home/fuxian/tracevae/TT_Dataset/TT_Dataset/convert_data_fixed"
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("🚀 开始修复的TT数据集转换...")
    
    try:
        # 1. 加载故障注入数据
        fault_data = load_fault_injection_data(data_dir)
        
        # 2. 加载所有traces
        traces = load_json_traces_with_time_period(data_dir)
        
        # 3. 提取spans
        spans, operations, services, statuses = extract_spans_from_labeled_traces(traces)
        
        # 4. 应用真实的故障标签
        spans = apply_fault_labels_to_spans(spans, fault_data)
        
        # 5. 生成ID映射（保持原有逻辑）
        operation_name_to_id, service_name_to_id, status_name_to_id, operation_durations = generate_id_mappings(
            operations, services, statuses, spans
        )
        
        # 6. 转换为CSV格式
        df = convert_spans_to_csv_format_with_labels(spans, operation_name_to_id, service_name_to_id, status_name_to_id)
        
        # 7. 分割数据集（确保保留标签）
        train_df, val_df, test_df = split_dataset_preserve_labels(df)
        
        # 8. 保存文件
        train_df.to_csv(Path(output_dir) / "train.csv", index=False)
        val_df.to_csv(Path(output_dir) / "val.csv", index=False)
        test_df.to_csv(Path(output_dir) / "test.csv", index=False)
        
        # 9. 保存YAML文件（保持原有逻辑）
        save_yaml_files(operation_name_to_id, service_name_to_id, status_name_to_id, operation_durations, output_dir)
        
        print("✅ 修复的转换完成!")
        print(f"训练集: {len(train_df)} 条记录")
        print(f"验证集: {len(val_df)} 条记录")
        print(f"测试集: {len(test_df)} 条记录")
        
        # 显示标签分布
        for split_name, split_df in [("训练", train_df), ("验证", val_df), ("测试", test_df)]:
            node_anomalies = split_df['nodeLatencyLabel'].sum()
            graph_lat_anomalies = split_df['graphLatencyLabel'].sum()
            graph_struct_anomalies = split_df['graphStructureLabel'].sum()
            print(f"{split_name}集异常分布: 节点延迟={node_anomalies}, 图延迟={graph_lat_anomalies}, 图结构={graph_struct_anomalies}")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

def convert_spans_to_csv_format_with_labels(spans, operation_name_to_id, service_name_to_id, status_name_to_id):
    """转换spans为CSV格式，保留标签信息"""
    print("转换spans为CSV格式（包含真实标签）...")
    
    csv_records = []
    trace_id_mapping = {}
    span_id_mapping = {}
    
    for span in spans:
        # 生成ID映射（保持原有逻辑）
        if span['traceID'] not in trace_id_mapping:
            trace_id_mapping[span['traceID']] = generate_realistic_ids(span['traceID'])
        
        if span['spanID'] not in span_id_mapping:
            span_id_mapping[span['spanID']] = generate_span_id()
        
        trace_id_high, trace_id_low = trace_id_mapping[span['traceID']]
        span_id = span_id_mapping[span['spanID']]
        
        # 处理父span ID
        parent_span_id = 0
        if span['parentSpanID']:
            if span['parentSpanID'] not in span_id_mapping:
                span_id_mapping[span['parentSpanID']] = generate_span_id()
            parent_span_id = span_id_mapping[span['parentSpanID']]
        
        # 转换时间
        start_time_sec = span['startTime'] / 1000000
        start_time_str = datetime.fromtimestamp(start_time_sec).strftime("%Y-%m-%d %H:%M:%S")
        nanosecond = (span['startTime'] % 1000000) * 1000
        
        # 转换duration
        duration_ms = max(1, int(span['duration'] / 1000))
        
        # 处理状态
        status_val = 0 if span['status'] == '200' else 1
        
        # 获取ID
        operation_id = operation_name_to_id.get(span['operationName'], 0)
        service_id = service_name_to_id.get(span['serviceName'], 0)

        # CSV记录包含真实标签
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
            'operationName': operation_id,
            'serviceName': service_id,
            'nodeLatencyLabel': span['nodeLatencyLabel'],
            'graphLatencyLabel': span['graphLatencyLabel'],
            'graphStructureLabel': span['graphStructureLabel']
        }
        
        csv_records.append(csv_record)
    
    return pd.DataFrame(csv_records)

def split_dataset_preserve_labels(df):
    """分割数据集，保留标签分布"""
    print("分割数据集（保留标签分布）...")
    
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
    
    # 分别分割正常和异常traces
    random.shuffle(normal_traces)
    random.shuffle(anomaly_traces)
    
    # 正常traces分割
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
    
    # 合并训练、验证、测试集
    train_traces = normal_train + anomaly_train
    val_traces = normal_val + anomaly_val
    test_traces = normal_test + anomaly_test
    
    # 创建数据框
    train_df = df[df.set_index(['traceIdHigh', 'traceIdLow']).index.isin(train_traces)].reset_index(drop=True)
    val_df = df[df.set_index(['traceIdHigh', 'traceIdLow']).index.isin(val_traces)].reset_index(drop=True)
    test_df = df[df.set_index(['traceIdHigh', 'traceIdLow']).index.isin(test_traces)].reset_index(drop=True)
    
    return train_df, val_df, test_df

# 保持其他辅助函数不变
def generate_id_mappings(operations, services, statuses, spans):
    """生成ID映射文件"""
    print("生成ID映射...")
    
    # 操作映射
    operation_name_to_id = {}
    for i, op in enumerate(sorted(operations), 1):
        operation_name_to_id[str(op)] = i
    
    # 服务映射
    service_name_to_id = {}
    for i, svc in enumerate(sorted(services), 1):
        service_name_to_id[str(svc)] = i
    
    # 状态映射
    status_name_to_id = {}
    for i, status in enumerate(sorted(statuses), 1):
        status_name_to_id[str(status)] = i
    
    # 计算延迟统计
    operation_durations = defaultdict(list)
    for span in spans:
        duration_ms = span['duration'] / 1000
        operation_name = span['operationName']
        operation_durations[operation_name].append(duration_ms)
    
    return operation_name_to_id, service_name_to_id, status_name_to_id, operation_durations

def generate_realistic_ids(original_id):
    """生成现实的trace ID"""
    hash_obj = hashlib.md5(str(original_id).encode())
    hash_int = int(hash_obj.hexdigest()[:16], 16)
    
    trace_id_high = (hash_int >> 32) & 0x7FFFFFFFFFFFFFFF
    trace_id_low = hash_int & 0x7FFFFFFFFFFFFFFF
    
    return trace_id_high, trace_id_low

def generate_span_id():
    """生成span ID"""
    return random.randint(-2**31, 2**31-1)

def save_yaml_files(operation_name_to_id, service_name_to_id, status_name_to_id, operation_durations, output_dir):
    """保存YAML映射文件"""
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
    
    # latency_range.yml
    latency_file = id_manager_dir / "latency_range.yml"
    with open(latency_file, 'w', encoding='utf-8') as f:
        global_id = 1
        service_ids = sorted(service_name_to_id.values())
        operation_ids = sorted(operation_name_to_id.values())
        
        for service_id in service_ids:
            if service_id > 0:
                for operation_id in operation_ids:
                    if operation_id > 0:
                        original_op_name = None
                        for op_name, mapped_id in operation_name_to_id.items():
                            if mapped_id == operation_id:
                                original_op_name = op_name
                                break
                        
                        if original_op_name and original_op_name in operation_durations and operation_durations[original_op_name]:
                            durations = operation_durations[original_op_name]
                            mean_val = float(np.mean(durations))
                            std_val = float(np.std(durations)) if len(durations) > 1 else 1.0
                            p99_val = float(np.percentile(durations, 99)) if len(durations) > 1 else mean_val * 3
                        else:
                            mean_val, std_val, p99_val = 1.0, 1.0, 3.0
                        
                        f.write(f"{global_id}:\n")
                        f.write(f"  mean: {mean_val}\n")
                        f.write(f"  std: {std_val}\n")
                        f.write(f"  p99: {p99_val}\n")
                        global_id += 1
    
    print(f"✅ YAML文件已生成到: {id_manager_dir}")

if __name__ == "__main__":
    main()