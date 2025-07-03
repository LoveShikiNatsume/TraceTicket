#!/usr/bin/env python3
"""
CSV文件监控器 - 监控指定路径的CSV文件，实时进行异常检测

使用方法：
# 一次性检测CSV文件
python csv_file_monitor.py your_data.csv --mode once

# 持续监控CSV文件
python csv_file_monitor.py your_data.csv --mode monitor
"""

import asyncio
import aiohttp
import pandas as pd
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVFileHandler(FileSystemEventHandler):
    """CSV文件变化处理器"""
    
    def __init__(self, file_monitor):
        self.file_monitor = file_monitor
    
    def on_modified(self, event):
        """文件修改时触发"""
        if not event.is_directory and event.src_path.endswith('.csv'):
            logger.info(f"📁 检测到CSV文件变化: {event.src_path}")
            asyncio.create_task(self.file_monitor.process_csv_file(event.src_path))

class CSVFileMonitor:
    """CSV文件监控器"""
    
    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url
        self.processed_lines = {}  # 记录每个文件已处理的行数
        
        # 操作和服务映射
        self.operation_mappings = {
            3: "GET /api/gateway",
            6: "POST /api/auth", 
            17: "SELECT database_query"
        }
        
        self.service_mappings = {
            2: "gateway-service",
            4: "database-service"
        }
    
    async def monitor_csv_file(self, csv_file_path: str, check_interval: int = 5):
        """监控单个CSV文件，定期检查新增的数据"""
        csv_path = Path(csv_file_path)
        
        if not csv_path.exists():
            logger.error(f"❌ CSV文件不存在: {csv_path}")
            return
        
        logger.info(f"🔍 开始监控CSV文件: {csv_path}")
        
        # 记录初始文件大小
        last_size = csv_path.stat().st_size
        last_modified = csv_path.stat().st_mtime
        
        while True:
            try:
                # 检查文件是否有变化
                current_size = csv_path.stat().st_size
                current_modified = csv_path.stat().st_mtime
                
                if current_size > last_size or current_modified > last_modified:
                    logger.info(f"📊 检测到文件变化，开始处理新数据...")
                    
                    # 处理新增的数据 - 返回整个CSV的状态
                    csv_status = await self.process_new_data(csv_path)
                    
                    # 输出CSV整体状态
                    if csv_status:
                        logger.info(f"📁 CSV文件状态: {csv_status}")
                    
                    last_size = current_size
                    last_modified = current_modified
                
                # 等待下次检查
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"❌ 监控文件时出错: {e}")
                await asyncio.sleep(check_interval)
    
    async def process_new_data(self, csv_path: Path):
        """处理CSV文件中的新数据，返回整个CSV的状态"""
        try:
            # 读取完整文件
            df = pd.read_csv(csv_path)
            
            # 获取文件的处理记录
            file_key = str(csv_path)
            last_processed = self.processed_lines.get(file_key, 0)
            
            # 只处理新增的行
            if len(df) > last_processed:
                new_data = df.iloc[last_processed:]
                logger.info(f"📈 发现 {len(new_data)} 行新数据")
                
                # 转换为traces格式
                traces = self.convert_csv_to_traces(new_data)
                
                if traces:
                    # 发送到检测服务，获取每个trace的检测结果
                    trace_results = await self.send_for_detection(traces)
                    
                    # 判断整个CSV文件的状态
                    csv_status = self.determine_csv_status(trace_results, csv_path.name)
                    
                    # 更新处理记录
                    self.processed_lines[file_key] = len(df)
                    
                    return csv_status
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 处理新数据失败: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def process_entire_csv_file(self, csv_file_path: str, batch_size: int = 20):
        """一次性处理整个CSV文件，返回整个CSV的状态"""
        csv_path = Path(csv_file_path)
        
        if not csv_path.exists():
            logger.error(f"❌ CSV文件不存在: {csv_path}")
            return {"status": "ERROR", "error": "文件不存在"}
        
        logger.info(f"📂 开始处理CSV文件: {csv_path}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            logger.info(f"📊 文件包含 {len(df)} 行数据")
            
            # 转换为traces
            traces = self.convert_csv_to_traces(df)
            logger.info(f"🔄 转换得到 {len(traces)} 个traces")
            
            if not traces:
                logger.warning("⚠️  没有有效的trace数据")
                return {"status": "NORMAL", "reason": "无有效数据"}
            
            # 分批处理
            all_trace_results = []
            for i in range(0, len(traces), batch_size):
                batch = traces[i:i + batch_size]
                logger.info(f"🔍 处理批次 {i//batch_size + 1}: {len(batch)} traces")
                
                batch_results = await self.send_for_detection(batch)
                if batch_results:
                    all_trace_results.extend(batch_results)
                
                # 避免请求过快
                await asyncio.sleep(0.5)
            
            # 判断整个CSV文件的状态
            csv_status = self.determine_csv_status(all_trace_results, csv_path.name)
            
            # 输出结果
            self.print_csv_result(csv_status)
            
            return csv_status
            
        except Exception as e:
            logger.error(f"❌ 处理CSV文件失败: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def determine_csv_status(self, trace_results: List[Dict], csv_filename: str) -> Dict[str, Any]:
        """
        根据trace检测结果判断整个CSV文件的状态
        只要有一个trace异常，整个CSV就是异常
        """
        if not trace_results:
            return {
                "csv_file": csv_filename,
                "status": "NORMAL", 
                "reason": "无检测结果",
                "total_traces": 0,
                "anomaly_traces": 0,
                "normal_traces": 0,
                "anomaly_percentage": 0.0
            }
        
        total_traces = len(trace_results)
        anomaly_traces = sum(1 for r in trace_results if r.get('is_anomaly', False))
        normal_traces = total_traces - anomaly_traces
        anomaly_percentage = (anomaly_traces / total_traces) * 100
        
        # 关键逻辑：只要有一个trace异常，整个CSV就是异常
        csv_is_anomaly = anomaly_traces > 0
        
        # 统计异常类型
        anomaly_types = {}
        anomaly_details = []
        
        for result in trace_results:
            if result.get('is_anomaly', False):
                anomaly_type = result.get('anomaly_type', 'unknown')
                anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
                
                anomaly_details.append({
                    "traceID": result.get('traceID'),
                    "anomaly_type": anomaly_type,
                    "confidence": result.get('confidence', 0)
                })
        
        return {
            "csv_file": csv_filename,
            "status": "ANOMALY" if csv_is_anomaly else "NORMAL",
            "total_traces": total_traces,
            "anomaly_traces": anomaly_traces,
            "normal_traces": normal_traces,
            "anomaly_percentage": anomaly_percentage,
            "anomaly_types": anomaly_types,
            "anomaly_details": anomaly_details[:10],  # 只保留前10个异常详情
            "detection_timestamp": datetime.utcnow().isoformat()
        }
    
    def print_csv_result(self, csv_status: Dict[str, Any]):
        """打印CSV检测结果"""
        print("\n" + "="*60)
        print("📊 CSV文件异常检测结果")
        print("="*60)
        
        status_icon = "🚨" if csv_status["status"] == "ANOMALY" else "✅"
        print(f"{status_icon} CSV文件: {csv_status['csv_file']}")
        print(f"📋 整体状态: {csv_status['status']}")
        print(f"📈 总traces: {csv_status['total_traces']}")
        print(f"✅ 正常traces: {csv_status['normal_traces']}")
        print(f"🚨 异常traces: {csv_status['anomaly_traces']}")
        print(f"📊 异常比例: {csv_status['anomaly_percentage']:.1f}%")
        
        if csv_status["status"] == "ANOMALY":
            print(f"\n🏷️  异常类型分布:")
            for anomaly_type, count in csv_status.get('anomaly_types', {}).items():
                print(f"  - {anomaly_type}: {count}")
            
            print(f"\n🔍 异常trace示例:")
            for detail in csv_status.get('anomaly_details', [])[:5]:
                print(f"  - {detail['traceID']}: {detail['anomaly_type']} "
                      f"(置信度: {detail['confidence']:.3f})")
        
        # 保存结果
        output_file = f"csv_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(csv_status, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细结果已保存到: {output_file}")
    
    # ... 其他方法保持不变 ...
    def convert_csv_to_traces(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """将CSV数据转换为traces格式"""
        traces = []
        
        try:
            # 数据类型转换
            df = self._convert_data_types(df)
            
            # 按trace分组
            trace_groups = df.groupby(['traceIdHigh', 'traceIdLow'])
            
            for (trace_high, trace_low), group in trace_groups:
                trace = self._convert_group_to_trace(group, trace_high, trace_low)
                if trace and len(trace['spans']) > 0:
                    traces.append(trace)
            
            return traces
            
        except Exception as e:
            logger.error(f"❌ 转换CSV数据失败: {e}")
            return []
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        try:
            # 转换数值列
            numeric_columns = ['traceIdHigh', 'traceIdLow', 'parentSpanId', 'spanId', 
                             'duration', 'nanosecond', 'DBhash', 'status', 'operationName', 'serviceName']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 转换时间列
            if 'startTime' in df.columns:
                df['startTime'] = pd.to_datetime(df['startTime'], errors='coerce')
            
            return df
        except Exception as e:
            logger.warning(f"数据类型转换失败: {e}")
            return df
    
    def _convert_group_to_trace(self, group: pd.DataFrame, trace_high: int, trace_low: int) -> Dict[str, Any]:
        """将grouped数据转换为trace格式"""
        try:
            trace_id = f"trace_{trace_high}_{trace_low}"
            spans = []
            processes = {}
            
            # 按时间排序spans
            group_sorted = group.sort_values(['startTime', 'nanosecond'])
            
            for _, row in group_sorted.iterrows():
                span = self._convert_row_to_span(row)
                if span:
                    spans.append(span)
                    
                    # 添加到processes
                    service_name = span['serviceName']
                    process_key = f"p{len(processes)}"
                    if service_name not in [p.get('serviceName') for p in processes.values()]:
                        processes[process_key] = {"serviceName": service_name}
            
            if not spans:
                return None
            
            return {
                "traceID": trace_id,
                "spans": spans,
                "processes": processes
            }
            
        except Exception as e:
            logger.warning(f"转换trace失败: {e}")
            return None
    
    def _convert_row_to_span(self, row) -> Dict[str, Any]:
        """将单行数据转换为span格式"""
        try:
            # Span IDs
            span_id = str(int(row['spanId']))
            parent_span_id = ""
            if row['parentSpanId'] != 0:
                parent_span_id = str(int(row['parentSpanId']))
            
            # 操作和服务名称
            operation_id = int(row['operationName'])
            service_id = int(row['serviceName'])
            
            operation_name = self.operation_mappings.get(operation_id, f"operation_{operation_id}")
            service_name = self.service_mappings.get(service_id, f"service_{service_id}")
            
            # 时间转换
            start_time = self._convert_start_time(row)
            
            # 持续时间转换（毫秒到微秒）
            duration_ms = int(row['duration'])
            duration_us = duration_ms * 1000
            
            # 状态转换
            status = int(row['status'])
            status_code = 500 if status == 1 else 200
            
            # 构建span
            span = {
                "spanID": span_id,
                "parentSpanID": parent_span_id,
                "operationName": operation_name,
                "serviceName": service_name,
                "startTime": start_time,
                "duration": duration_us,
                "tags": [
                    {"key": "http.status_code", "value": status_code},
                    {"key": "operation.id", "value": operation_id},
                    {"key": "service.id", "value": service_id}
                ]
            }
            
            return span
            
        except Exception as e:
            logger.warning(f"转换span失败: {e}")
            return None
    
    def _convert_start_time(self, row) -> int:
        """转换开始时间为微秒时间戳"""
        try:
            start_time = row['startTime']
            nanosecond = int(row['nanosecond'])
            
            if pd.isna(start_time):
                return int(datetime.now().timestamp() * 1000000)
            
            # 转换为微秒时间戳
            timestamp_us = int(start_time.timestamp() * 1000000)
            timestamp_us += nanosecond // 1000
            
            return timestamp_us
            
        except Exception as e:
            return int(datetime.now().timestamp() * 1000000)
    
    async def send_for_detection(self, traces: List[Dict]) -> List[Dict]:
        """发送traces到检测服务"""
        try:
            request_data = {
                "traces": traces,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.post(
                    f"{self.service_url}/detect",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    request_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        anomaly_count = result.get('total_anomalies', 0)
                        total_count = result.get('total_processed', 0)
                        
                        logger.info(f"✅ 检测完成: {total_count} traces, "
                                  f"{anomaly_count} 异常, {request_time:.0f}ms")
                        
                        return result.get('results', [])
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 检测请求失败: {response.status} - {error_text}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ 发送检测请求失败: {e}")
            return []

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CSV文件异常检测监控器")
    parser.add_argument("csv_file", help="要监控的CSV文件路径")
    parser.add_argument("--mode", choices=["once", "monitor"], default="once",
                       help="运行模式: once=一次性处理, monitor=持续监控")
    parser.add_argument("--service-url", default="http://localhost:8000",
                       help="检测服务URL")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="批处理大小")
    parser.add_argument("--check-interval", type=int, default=5,
                       help="监控模式下的检查间隔(秒)")
    
    args = parser.parse_args()
    
    # 检查CSV文件是否存在
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ CSV文件不存在: {csv_path}")
        return
    
    # 创建监控器
    monitor = CSVFileMonitor(args.service_url)
    
    # 检查检测服务
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.service_url}/health") as response:
                if response.status != 200:
                    print(f"❌ 检测服务不可用: {args.service_url}")
                    print("请先启动检测服务: python run.py")
                    return
    except Exception as e:
        print(f"❌ 无法连接到检测服务: {e}")
        print("请先启动检测服务: python run.py")
        return
    
    if args.mode == "once":
        print(f"🔄 一次性处理CSV文件: {csv_path}")
        csv_result = await monitor.process_entire_csv_file(args.csv_file, args.batch_size)
        
        # 输出最终结果
        print(f"\n🎯 最终结果: CSV文件 '{csv_path.name}' 状态为 {csv_result['status']}")
        
    elif args.mode == "monitor":
        print(f"👁️  持续监控CSV文件: {csv_path}")
        print("按 Ctrl+C 停止监控")
        await monitor.monitor_csv_file(args.csv_file, args.check_interval)

if __name__ == "__main__":
    # 安装依赖检查
    try:
        import watchdog
    except ImportError:
        print("需要安装watchdog: pip install watchdog")
        exit(1)
    
    asyncio.run(main())