#!/usr/bin/env python3
"""
检测结果查看器 - 分析和可视化异常检测结果
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionResultViewer:
    """检测结果查看器"""
    
    def __init__(self, result_file: str):
        self.result_file = Path(result_file)
        self.results = self.load_results()
    
    def load_results(self) -> List[Dict]:
        """加载检测结果"""
        try:
            with open(self.result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"✅ 加载了 {len(results)} 个检测结果")
            return results
        except Exception as e:
            logger.error(f"❌ 加载结果文件失败: {e}")
            return []
    
    def show_summary(self):
        """显示检测结果摘要"""
        if not self.results:
            print("❌ 没有检测结果")
            return
        
        total_traces = len(self.results)
        anomaly_traces = sum(1 for r in self.results if r.get('is_anomaly', False))
        normal_traces = total_traces - anomaly_traces
        
        # 计算平均置信度
        avg_confidence = sum(r.get('confidence', 0) for r in self.results) / total_traces
        avg_processing_time = sum(r.get('processing_time_ms', 0) for r in self.results) / total_traces
        
        print("\n" + "="*60)
        print(f"📊 检测结果摘要 - {self.result_file.name}")
        print("="*60)
        print(f"📈 总trace数量: {total_traces}")
        print(f"✅ 正常traces: {normal_traces} ({normal_traces/total_traces*100:.1f}%)")
        print(f"🚨 异常traces: {anomaly_traces} ({anomaly_traces/total_traces*100:.1f}%)")
        print(f"🎯 平均置信度: {avg_confidence:.3f}")
        print(f"⏱️  平均处理时间: {avg_processing_time:.1f}ms")
        
        # 异常类型统计
        anomaly_types = {}
        for result in self.results:
            if result.get('is_anomaly', False):
                anomaly_type = result.get('anomaly_type', 'unknown')
                anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
        
        if anomaly_types:
            print(f"\n🏷️  异常类型分布:")
            for anomaly_type, count in sorted(anomaly_types.items()):
                percentage = count / anomaly_traces * 100 if anomaly_traces > 0 else 0
                print(f"  - {anomaly_type}: {count} ({percentage:.1f}%)")
    
    def show_anomalies(self, limit: int = 10):
        """显示异常详情"""
        anomalies = [r for r in self.results if r.get('is_anomaly', False)]
        
        if not anomalies:
            print("\n🎉 恭喜！没有检测到任何异常")
            return
        
        print(f"\n🚨 检测到的异常 (前{min(limit, len(anomalies))}个):")
        print("-" * 80)
        
        # 按置信度排序
        anomalies_sorted = sorted(anomalies, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, anomaly in enumerate(anomalies_sorted[:limit], 1):
            trace_id = anomaly.get('traceID', 'N/A')
            anomaly_type = anomaly.get('anomaly_type', 'unknown')
            confidence = anomaly.get('confidence', 0)
            
            details = anomaly.get('details', {})
            latency_score = details.get('nodeLatencyScore', 0)
            structure_score = details.get('graphStructureScore', 0)
            
            print(f"\n{i}. TraceID: {trace_id}")
            print(f"   异常类型: {anomaly_type}")
            print(f"   置信度: {confidence:.3f}")
            print(f"   延迟分数: {latency_score:.3f}")
            print(f"   结构分数: {structure_score:.3f}")
            
            # 如果有spans信息，显示问题spans
            if 'spans' in anomaly:
                spans = anomaly['spans']
                print(f"   Spans数量: {len(spans)}")
                
                # 找出最可能有问题的span
                if spans and isinstance(spans, list):
                    max_duration_span = max(spans, key=lambda s: s.get('duration', 0))
                    duration_ms = max_duration_span.get('duration', 0) / 1000
                    print(f"   最长耗时Span: {max_duration_span.get('operationName', 'N/A')} ({duration_ms:.1f}ms)")
    
    def show_normal_traces(self, limit: int = 5):
        """显示正常traces样例"""
        normal_traces = [r for r in self.results if not r.get('is_anomaly', False)]
        
        if not normal_traces:
            print("\n⚠️  没有正常的traces")
            return
        
        print(f"\n✅ 正常traces样例 (前{min(limit, len(normal_traces))}个):")
        print("-" * 60)
        
        for i, trace in enumerate(normal_traces[:limit], 1):
            trace_id = trace.get('traceID', 'N/A')
            confidence = trace.get('confidence', 0)
            processing_time = trace.get('processing_time_ms', 0)
            
            print(f"{i}. TraceID: {trace_id}")
            print(f"   置信度: {confidence:.3f}")
            print(f"   处理时间: {processing_time:.1f}ms")
    
    def export_to_csv(self, output_file: str = None):
        """导出结果到CSV"""
        if not self.results:
            print("❌ 没有数据可导出")
            return
        
        if not output_file:
            output_file = f"analysis_{self.result_file.stem}.csv"
        
        # 准备数据
        data = []
        for result in self.results:
            row = {
                'traceID': result.get('traceID', ''),
                'is_anomaly': result.get('is_anomaly', False),
                'anomaly_type': result.get('anomaly_type', 'normal'),
                'confidence': result.get('confidence', 0),
                'processing_time_ms': result.get('processing_time_ms', 0),
                'nodeLatencyScore': result.get('details', {}).get('nodeLatencyScore', 0),
                'graphStructureScore': result.get('details', {}).get('graphStructureScore', 0)
            }
            data.append(row)
        
        # 保存CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"📊 结果已导出到: {output_file}")
        
        # 显示统计信息
        print(f"\n📈 导出统计:")
        print(f"  - 总记录数: {len(df)}")
        print(f"  - 异常记录: {df['is_anomaly'].sum()}")
        print(f"  - 平均置信度: {df['confidence'].mean():.3f}")
        print(f"  - 最高置信度: {df['confidence'].max():.3f}")
        print(f"  - 最低置信度: {df['confidence'].min():.3f}")
    
    def create_visualization(self):
        """创建可视化图表"""
        if not self.results:
            print("❌ 没有数据可视化")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 准备数据
            df = pd.DataFrame(self.results)
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'异常检测结果分析 - {self.result_file.name}', fontsize=16)
            
            # 1. 异常vs正常分布
            anomaly_counts = df['is_anomaly'].value_counts()
            axes[0, 0].pie(anomaly_counts.values, labels=['正常', '异常'], autopct='%1.1f%%', 
                          colors=['lightgreen', 'lightcoral'])
            axes[0, 0].set_title('异常vs正常分布')
            
            # 2. 置信度分布
            axes[0, 1].hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('置信度分布')
            axes[0, 1].set_xlabel('置信度')
            axes[0, 1].set_ylabel('频次')
            
            # 3. 异常类型分布
            if 'anomaly_type' in df.columns:
                anomaly_types = df[df['is_anomaly'] == True]['anomaly_type'].value_counts()
                if len(anomaly_types) > 0:
                    axes[1, 0].bar(anomaly_types.index, anomaly_types.values, color='orange')
                    axes[1, 0].set_title('异常类型分布')
                    axes[1, 0].set_xlabel('异常类型')
                    axes[1, 0].set_ylabel('数量')
                    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # 4. 处理时间分布
            if 'processing_time_ms' in df.columns:
                axes[1, 1].hist(df['processing_time_ms'], bins=20, alpha=0.7, color='lightcyan', edgecolor='black')
                axes[1, 1].set_title('处理时间分布')
                axes[1, 1].set_xlabel('处理时间 (ms)')
                axes[1, 1].set_ylabel('频次')
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = f"analysis_chart_{self.result_file.stem}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"📈 可视化图表已保存到: {chart_file}")
            
            # 显示图表
            plt.show()
            
        except ImportError:
            print("⚠️  需要安装matplotlib和seaborn: pip install matplotlib seaborn")
        except Exception as e:
            print(f"❌ 创建可视化失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="检测结果查看器")
    parser.add_argument("result_file", help="检测结果JSON文件路径")
    parser.add_argument("--export-csv", action="store_true", help="导出到CSV文件")
    parser.add_argument("--visualize", action="store_true", help="创建可视化图表")
    parser.add_argument("--anomaly-limit", type=int, default=10, help="显示异常的数量限制")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.result_file).exists():
        print(f"❌ 结果文件不存在: {args.result_file}")
        return
    
    # 创建查看器
    viewer = DetectionResultViewer(args.result_file)
    
    # 显示摘要
    viewer.show_summary()
    
    # 显示异常详情
    viewer.show_anomalies(args.anomaly_limit)
    
    # 显示正常traces样例
    viewer.show_normal_traces(5)
    
    # 导出CSV
    if args.export_csv:
        viewer.export_to_csv()
    
    # 创建可视化
    if args.visualize:
        viewer.create_visualization()

if __name__ == "__main__":
    main()