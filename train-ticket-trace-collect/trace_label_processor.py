# -*- coding: utf-8 -*-
"""
Train Ticket Trace Label Processor
对已采集的11列数据生成异常标签，扩展为14列数据用于异常检测

Author: LoveShikiNatsume
Date: 2025-06-18
Version: 1.0 基于故障注入记录生成异常标签
"""

import json
import pandas as pd
import os
import logging
import csv
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class TraceAnomalyLabelProcessor:
    """调用链异常标签生成器"""
    
    def __init__(self, data_dir: str = "trace", fault_records_dir: str = "fault_injection_records"):
        self.data_dir = data_dir
        self.fault_records_dir = fault_records_dir
        self.logger = self._setup_logging()
        
        # 标签生成统计
        self.processing_stats = {
            "total_files": 0,
            "total_spans": 0,
            "labeled_spans": 0,
            "anomaly_spans": 0,
            "normal_spans": 0,
            "fault_records_found": 0,
            "fault_records_used": 0
        }
        
        self.logger.info("链路异常标签处理器初始化完成")
        self.logger.info(f"数据目录: {data_dir}")
        self.logger.info(f"故障记录目录: {fault_records_dir}")

    def _setup_logging(self):
        """设置日志系统"""
        logger = logging.getLogger('TraceLabelProcessor')
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def load_csv_data(self, csv_path: str) -> List[Dict]:
        """从11列CSV文件加载基础数据"""
        try:
            df = pd.read_csv(csv_path)
            
            # 验证列数和格式
            expected_11_columns = [
                "traceIdHigh", "traceIdLow", "parentSpanId", "spanId", 
                "startTime", "duration", "nanosecond", "DBhash", "status",
                "operationName", "serviceName"
            ]
            
            if len(df.columns) == 14:
                self.logger.debug(f"文件 {csv_path} 已经是14列格式，跳过处理")
                return []
            elif len(df.columns) != 11:
                self.logger.warning(f"文件 {csv_path} 列数不匹配: {len(df.columns)} 列，期望11列")
                return []
            
            spans_data = df.to_dict('records')
            
            # 为每个span添加trace标识用于分组
            for span in spans_data:
                span['_trace_id'] = f"{span['traceIdHigh']}-{span['traceIdLow']}"
            
            self.processing_stats["total_spans"] += len(spans_data)
            return spans_data
            
        except Exception as e:
            self.logger.error(f"加载CSV文件失败 {csv_path}: {e}")
            return []

    def load_fault_injection_records(self, target_date: str) -> Dict[str, Dict]:
        """加载指定日期的故障注入记录"""
        fault_file = Path(self.fault_records_dir) / f"fault_records_{target_date.replace('-', '')}.json"
        
        if not fault_file.exists():
            self.logger.debug(f"故障记录文件不存在: {fault_file}")
            return {}
        
        try:
            with open(fault_file, 'r', encoding='utf-8') as f:
                fault_records = json.load(f)
            
            # 将记录按minute_key索引
            indexed_records = {}
            for record in fault_records:
                minute_key = record.get("minute_key")
                if minute_key:
                    indexed_records[minute_key] = record
            
            self.processing_stats["fault_records_found"] = len(indexed_records)
            self.logger.info(f"加载故障记录: {len(indexed_records)} 条 (日期: {target_date})")
            
            return indexed_records
            
        except Exception as e:
            self.logger.error(f"加载故障记录失败 {fault_file}: {e}")
            return {}

    def extract_minute_key_from_filename(self, csv_filename: str) -> Optional[str]:
        """从CSV文件名提取minute_key"""
        try:
            # 文件名格式: HH_MM.csv
            basename = os.path.basename(csv_filename)
            if basename.endswith('.csv'):
                minute_key = basename[:-4]  # 移除.csv后缀
                
                # 验证格式 HH_MM
                if '_' in minute_key and len(minute_key.split('_')) == 2:
                    hour, minute = minute_key.split('_')
                    if hour.isdigit() and minute.isdigit():
                        return minute_key
            
            return None
        except Exception:
            return None

    def determine_trace_time_range(self, spans_data: List[Dict]) -> tuple:
        """确定trace数据的时间范围"""
        try:
            start_times = []
            for span in spans_data:
                start_time_str = span.get('startTime', '')
                if start_time_str and start_time_str != "1970-01-01 00:00:00":
                    try:
                        dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                        start_times.append(dt)
                    except ValueError:
                        continue
            
            if start_times:
                earliest = min(start_times)
                latest = max(start_times)
                return earliest, latest
            
            return None, None
        except Exception:
            return None, None

    def check_fault_injection_match(self, spans_data: List[Dict], fault_records: Dict[str, Dict], 
                                   minute_key: str) -> tuple:
        """检查trace数据是否匹配故障注入记录"""
        # 首先检查minute_key是否有对应的故障记录
        fault_record = fault_records.get(minute_key)
        
        if not fault_record:
            # 没有找到故障记录，标记为正常
            return False, {
                "has_fault": False,
                "reason": "no_fault_record",
                "minute_key": minute_key
            }
        
        # 找到故障记录，进一步验证时间范围
        try:
            fault_start_time = datetime.strptime(fault_record.get("start_time", ""), '%Y-%m-%d %H:%M:%S')
            fault_end_time = datetime.strptime(fault_record.get("end_time", ""), '%Y-%m-%d %H:%M:%S')
            
            # 检查trace时间是否在故障时间范围内
            earliest_trace, latest_trace = self.determine_trace_time_range(spans_data)
            
            if earliest_trace and latest_trace:
                # 判断trace时间与故障时间是否有重叠
                trace_in_fault_range = (
                    earliest_trace <= fault_end_time and
                    latest_trace >= fault_start_time
                )
                
                if trace_in_fault_range:
                    self.processing_stats["fault_records_used"] += 1
                    return True, {
                        "has_fault": True,
                        "fault_type": fault_record.get("fault_type", "unknown"),
                        "description": fault_record.get("description", ""),
                        "intensity": fault_record.get("intensity", "unknown"),
                        "fault_start": fault_record.get("start_time"),
                        "fault_end": fault_record.get("end_time"),
                        "minute_key": minute_key
                    }
                else:
                    return False, {
                        "has_fault": False,
                        "reason": "time_mismatch",
                        "minute_key": minute_key
                    }
            else:
                # 如果无法确定trace时间，但有故障记录，默认认为是异常
                self.processing_stats["fault_records_used"] += 1
                return True, {
                    "has_fault": True,
                    "fault_type": fault_record.get("fault_type", "unknown"),
                    "description": fault_record.get("description", ""),
                    "intensity": fault_record.get("intensity", "unknown"),
                    "minute_key": minute_key,
                    "note": "time_verification_failed"
                }
                
        except Exception as e:
            self.logger.debug(f"故障记录时间解析失败: {e}")
            # 解析失败但有记录，保守地标记为异常
            self.processing_stats["fault_records_used"] += 1
            return True, {
                "has_fault": True,
                "fault_type": fault_record.get("fault_type", "unknown"),
                "description": fault_record.get("description", ""),
                "minute_key": minute_key,
                "note": "time_parse_error"
            }

    def generate_anomaly_labels(self, spans_data: List[Dict], fault_match_result: tuple) -> List[Dict]:
        """为所有span生成异常标签"""
        has_fault, fault_info = fault_match_result
        
        # 根据故障注入情况确定标签
        if has_fault:
            # 有故障注入 -> 异常标签
            node_latency_label = 1
            graph_latency_label = 1
            graph_structure_label = 0  # 结构标签始终为0（不考虑结构异常）
            
            self.processing_stats["anomaly_spans"] += len(spans_data)
            
        else:
            # 无故障注入 -> 正常标签
            node_latency_label = 0
            graph_latency_label = 0
            graph_structure_label = 0
            
            self.processing_stats["normal_spans"] += len(spans_data)
        
        # 为所有span添加标签
        labeled_spans = []
        for span in spans_data:
            labeled_span = span.copy()
            labeled_span['nodeLatencyLabel'] = node_latency_label
            labeled_span['graphLatencyLabel'] = graph_latency_label
            labeled_span['graphStructureLabel'] = graph_structure_label
            
            # 添加调试信息（不写入CSV）
            labeled_span['_fault_info'] = fault_info
            labeled_span['_label_generation_time'] = datetime.now().isoformat()
            
            labeled_spans.append(labeled_span)
        
        self.processing_stats["labeled_spans"] += len(labeled_spans)
        return labeled_spans

    def save_labeled_csv(self, spans_data: List[Dict], output_path: str):
        """保存带异常标签的14列CSV文件"""
        if not spans_data:
            return
        
        # 定义14列输出字段
        fieldnames = [
            "traceIdHigh", "traceIdLow", "parentSpanId", "spanId", 
            "startTime", "duration", "nanosecond", "DBhash", "status",
            "operationName", "serviceName", "nodeLatencyLabel",
            "graphLatencyLabel", "graphStructureLabel"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for span in spans_data:
                row = {field: span.get(field, 0) for field in fieldnames}
                writer.writerow(row)
        
        self.logger.debug(f"已保存标签化CSV文件: {os.path.basename(output_path)} ({len(spans_data)} spans)")

    def process_single_csv_file(self, csv_path: str, target_date: str) -> bool:
        """处理单个CSV文件并生成异常标签"""
        try:
            # 检查是否已处理
            processed_flag = csv_path.replace('.csv', '.label_processed')
            if os.path.exists(processed_flag):
                self.logger.debug(f"文件 {os.path.basename(csv_path)} 已处理，跳过")
                return True
            
            # 加载11列基础数据
            spans_data = self.load_csv_data(csv_path)
            if not spans_data:
                return False
            
            # 加载故障注入记录
            fault_records = self.load_fault_injection_records(target_date)
            
            # 从文件名提取时间键
            minute_key = self.extract_minute_key_from_filename(csv_path)
            if not minute_key:
                self.logger.warning(f"无法从文件名提取时间键: {csv_path}")
                return False
            
            # 检查故障注入匹配
            fault_match_result = self.check_fault_injection_match(spans_data, fault_records, minute_key)
            
            # 生成异常标签
            labeled_spans = self.generate_anomaly_labels(spans_data, fault_match_result)
            
            # 保存标签化数据
            self.save_labeled_csv(labeled_spans, csv_path)
            
            # 创建处理标志
            self._create_processing_flag(csv_path, fault_match_result)
            
            # 记录处理状态
            has_fault, fault_info = fault_match_result
            if has_fault:
                self.logger.info(f"已处理文件: {os.path.basename(csv_path)} (异常 - {fault_info.get('fault_type', 'unknown')})")
            else:
                self.logger.info(f"已处理文件: {os.path.basename(csv_path)} (正常)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理文件失败 {csv_path}: {e}")
            return False

    def _create_processing_flag(self, csv_path: str, fault_match_result: tuple):
        """创建处理完成标志文件"""
        flag_file = csv_path.replace('.csv', '.label_processed')
        has_fault, fault_info = fault_match_result
        
        with open(flag_file, 'w', encoding='utf-8') as f:
            f.write(f"Label processing completed at: {datetime.now().isoformat()}\n")
            f.write(f"CSV file enhanced to 14 columns with anomaly labels\n")
            f.write(f"Fault injection status: {'ANOMALY' if has_fault else 'NORMAL'}\n")
            if has_fault:
                f.write(f"Fault type: {fault_info.get('fault_type', 'unknown')}\n")
                f.write(f"Description: {fault_info.get('description', '')}\n")
            f.write(f"Ready for anomaly detection\n")
        
        self.logger.debug(f"已创建处理标志: {os.path.basename(flag_file)}")

    def process_date_directory(self, date_dir: str) -> bool:
        """处理某个日期目录的所有数据"""
        csv_dir = os.path.join(date_dir, "csv")
        
        if not os.path.exists(csv_dir):
            self.logger.warning(f"CSV 目录不存在: {csv_dir}")
            return False
        
        # 从路径提取日期
        target_date = os.path.basename(date_dir)
        
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            self.logger.warning(f"未找到 CSV 文件: {csv_dir}")
            return False
        
        total_files = len(csv_files)
        self.logger.info(f"开始处理 {total_files} 个CSV文件 (日期: {target_date})")
        
        # 重置统计
        self.processing_stats.update({
            "total_files": 0,
            "total_spans": 0,
            "labeled_spans": 0,
            "anomaly_spans": 0,
            "normal_spans": 0,
            "fault_records_found": 0,
            "fault_records_used": 0
        })
        
        processed_count = 0
        
        for csv_file in sorted(csv_files):
            csv_path = os.path.join(csv_dir, csv_file)
            if self.process_single_csv_file(csv_path, target_date):
                processed_count += 1
                self.processing_stats["total_files"] += 1
            
            # 每处理10个文件显示一次进度
            if processed_count % 10 == 0:
                self.logger.info(f"处理进度: {processed_count}/{total_files}")
        
        # 打印处理统计
        self._print_processing_stats(date_dir)
        
        self.logger.info(f"处理完成: {processed_count}/{total_files} 个文件")
        return processed_count > 0

    def process_specific_csv_file(self, csv_file_path: str) -> bool:
        """处理指定的CSV文件"""
        if not os.path.exists(csv_file_path):
            self.logger.error(f"CSV文件不存在: {csv_file_path}")
            return False
        
        # 从路径提取日期
        try:
            path_parts = Path(csv_file_path).parts
            date_index = -1
            for i, part in enumerate(path_parts):
                if len(part) == 10 and part.count('-') == 2:  # YYYY-MM-DD格式
                    date_index = i
                    break
            
            if date_index == -1:
                self.logger.error(f"无法从路径提取日期: {csv_file_path}")
                return False
            
            target_date = path_parts[date_index]
            return self.process_single_csv_file(csv_file_path, target_date)
            
        except Exception as e:
            self.logger.error(f"处理指定文件失败: {e}")
            return False

    def run_label_processing(self, specific_date: str = None) -> bool:
        """运行标签处理流程"""
        if specific_date:
            date_dir = os.path.join(self.data_dir, specific_date)
            if not os.path.exists(date_dir):
                self.logger.error(f"日期目录不存在: {date_dir}")
                return False
            
            self.logger.info(f"处理日期: {specific_date}")
            return self.process_date_directory(date_dir)
        else:
            if not os.path.exists(self.data_dir):
                self.logger.error(f"数据目录不存在: {self.data_dir}")
                return False
            
            date_dirs = [d for d in os.listdir(self.data_dir) 
                        if os.path.isdir(os.path.join(self.data_dir, d)) and 
                        len(d) == 10 and d.count('-') == 2]  # YYYY-MM-DD格式
            
            if not date_dirs:
                self.logger.warning(f"未找到日期目录: {self.data_dir}")
                return False
            
            self.logger.info(f"发现 {len(date_dirs)} 个日期目录")
            
            success_count = 0
            for date_str in sorted(date_dirs):
                date_dir = os.path.join(self.data_dir, date_str)
                self.logger.info(f"处理日期: {date_str}")
                if self.process_date_directory(date_dir):
                    success_count += 1
            
            self.logger.info(f"处理完成: {success_count}/{len(date_dirs)} 个日期")
            return success_count > 0

    def _print_processing_stats(self, date_dir: str):
        """打印处理统计信息"""
        stats = self.processing_stats
        
        if stats["total_spans"] == 0:
            self.logger.warning("未处理任何span数据")
            return
        
        self.logger.info("=" * 50)
        self.logger.info("标签生成完成 - 统计结果")
        self.logger.info("=" * 50)
        self.logger.info(f"处理的文件数: {stats['total_files']}")
        self.logger.info(f"处理的span总数: {stats['total_spans']}")
        self.logger.info(f"生成标签的span数: {stats['labeled_spans']}")
        
        if stats["labeled_spans"] > 0:
            anomaly_rate = (stats["anomaly_spans"] / stats["labeled_spans"]) * 100
            normal_rate = (stats["normal_spans"] / stats["labeled_spans"]) * 100
            
            self.logger.info("标签分布:")
            self.logger.info(f"  异常标签: {stats['anomaly_spans']} ({anomaly_rate:.1f}%)")
            self.logger.info(f"  正常标签: {stats['normal_spans']} ({normal_rate:.1f}%)")
        
        self.logger.info(f"故障记录: 发现 {stats['fault_records_found']} 条，使用 {stats['fault_records_used']} 条")
        self.logger.info("=" * 50)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Ticket 调用链异常标签处理器")
    parser.add_argument("--date", type=str, help="处理特定日期 (YYYY-MM-DD)")
    parser.add_argument("--file", type=str, help="处理特定CSV文件")
    parser.add_argument("--data-dir", type=str, default="trace", help="数据目录路径，默认: trace")
    parser.add_argument("--fault-dir", type=str, default="fault_injection_records", help="故障记录目录，默认: fault_injection_records")
    
    args = parser.parse_args()
    
    try:
        import pandas
    except ImportError as e:
        print(f"缺少依赖包: {e}")
        print("请运行: pip install pandas")
        return 1
    
    processor = TraceAnomalyLabelProcessor(
        data_dir=args.data_dir,
        fault_records_dir=args.fault_dir
    )
    
    try:
        if args.file:
            success = processor.process_specific_csv_file(args.file)
        else:
            success = processor.run_label_processing(specific_date=args.date)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("标签处理中断")
        return 0
    except Exception as e:
        print(f"标签处理失败: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
