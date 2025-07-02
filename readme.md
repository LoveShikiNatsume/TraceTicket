# TraceTicket

Train Ticket 系统Trace采集与异常检测工具，用于微服务异常检测研究。

## 项目结构

```
TraceTicket/
├── train-ticket-trace-collect/       # 数据采集与标签生成
│   ├── trace_collector.py            # 链路数据采集
│   ├── trace_label_processor.py      # 异常标签生成
│   ├── metrics_collector.py          # 系统指标采集
│   └── config.py                     # 采集配置
├── train-ticket-auto-query/          # 压测流量生成
│   ├── config.py                     # 压测配置
│   ├── queries.py                    # 核心查询类
│   ├── atomic_queries.py             # 原子操作
│   ├── scenarios.py                  # 业务场景
│   ├── normal_request_manager.py     # 请求管理
│   ├── run.py                        # 压测启动
│   └── query_*.py                    # 业务查询模块
├── fault_injection_records/          # 故障注入记录
│   └── fault_records_YYYYMMDD.json   # 按日期存储
├── main_controller.py                # 主控制器
├── trace/                            # 链路数据输出
│   └── YYYY-MM-DD/
│       └── csv/
│           ├── HH_MM.csv             # 分钟级数据
│           └── HH_MM.label_processed # 处理标志
└── metrics/                          # 指标数据输出
    └── YYYY-MM-DD/
        └── csv/
            └── HH_MM.csv             # 分钟级指标
```

## 功能特性

主控制器协调所有组件，实现一键启动的实时异常检测系统。系统包含压测流量生成、链路数据采集、系统指标采集、异常标签生成和准确性验证等功能。

压测模块可配置轻度、中度、重度三种强度，模拟真实用户行为。数据采集模块从Jaeger采集链路数据，从Prometheus采集系统指标。标签生成器基于故障注入记录自动生成异常标签，系统会实时对比模型检测结果与标签，统计检测准确率。

## 数据格式

基础采集数据为11列格式：
```csv
traceIdHigh,traceIdLow,parentSpanId,spanId,startTime,duration,nanosecond,DBhash,status,operationName,serviceName
```

经过标签处理后扩展为14列，增加三个异常标签：nodeLatencyLabel（节点延迟异常）、graphLatencyLabel（图延迟异常）、graphStructureLabel（图结构异常，固定为0）。标签值为0表示正常，1表示异常。

## 故障注入记录

要获得有意义的异常检测准确率统计，必须提供故障注入记录文件。因为系统默认将所有无故障记录的数据标记为正常（0,0,0），只有通过故障注入记录才能生成异常标签，进而计算检测准确率。

故障记录文件位于 `fault_injection_records/fault_records_YYYYMMDD.json`，格式如下：

```json
[
  {
    "start_time": "2025-06-18 14:30:00",
    "end_time": "2025-06-18 14:32:00",
    "minute_key": "14_30",
    "fault_type": "latency_injection",
    "description": "注入网络延迟故障"
  }
]
```

系统根据trace时间是否落在故障时间范围内来生成标签。有故障记录的时间段标记为异常（1,1,0），无故障记录的时间段标记为正常（0,0,0）。

## 快速使用

安装依赖：`pip install requests pandas numpy networkx`

配置连接地址：编辑 `train-ticket-trace-collect/config.py` 中的Jaeger和Prometheus地址，以及 `train-ticket-auto-query/*.py` 中的Train Ticket系统地址。

运行系统：`python main_controller.py`

可选配置包括编辑 `train-ticket-auto-query/config.py` 调整压测强度，创建 `fault_injection_records/` 目录和对应的故障记录文件。

## 配置调整

**压测强度**：编辑 `train-ticket-auto-query/config.py` 中的 `DEFAULT_PRESSURE_LEVEL` 和 `PRESSURE_TEST_PRESETS`

**指标采集间隔**：修改 `main_controller.py` 中 `start_metrics_collection()` 方法的 `--interval` 参数（当前1秒）

**异常检测阈值**：修改 `main_controller.py` 中的 `detection_threshold`（当前0.15）

## 异常检测模型接口

系统通过 `main_controller.py` 中的 `call_anomaly_detection_model()` 方法调用检测模型。当前为模拟实现，要接入真实模型需要替换该方法。

**输入参数**：14列CSV文件路径（包含异常标签的完整数据）

**期望返回格式**：
```json
{
  "file_name": "14_30.csv",
  "anomaly_score": 0.8765,
  "threshold": 0.15,
  "anomaly_detected": true,
  "model_confidence": 0.7265,
  "anomaly_types": ["high_latency", "service_degradation"]
}
```

**调用示例**：
```python
# 真实模型调用示例
cmd = [sys.executable, "path/to/model.py", "--input", csv_file, "--output-format", "json"]
result = subprocess.run(cmd, capture_output=True, text=True)
return json.loads(result.stdout)
```

## 运行模式

基本监控模式下，所有数据标记为正常，只能监控模型检测结果，无法计算准确率。完整验证模式需要提供故障注入记录文件，系统会根据故障记录生成异常标签并计算检测准确率。

## 输出示例

实时监控状态显示运行时间、检测次数、异常次数、准确率以及各组件状态：
```
监控状态: 运行时间: 2.5h | 检测次数: 35 | 异常次数: 8 | 准确率: 85.7% | 压测:运行(1250/1400, 89.3%) | 采集:运行 | 指标:运行
```

异常检测结果会详细显示模型输出、标签期望、准确性评估等信息。正常检测结果以简洁形式显示。最终统计包含总检测次数、检测到异常数量、准确率分解（真阳性、真阴性、假阳性、假阴性）等信息。

## 架构特点

系统采用模块化设计，各组件独立运行，易于扩展和调试。支持分钟级实时处理，可长期运行。基于ground truth的准确性验证，支持不同压测强度和检测参数调整。

当前异常检测模型为模拟实现。要接入真实模型，只需替换 `main_controller.py` 中的 `call_anomaly_detection_model()` 方法即可。

## 故障记录获取

可配合Chaos Engineering工具（如ChaosMesh、Gremlin等）自动生成故障记录，或在人工故障测试时手动记录故障时间，也可将故障注入脚本与本系统集成。建议故障类型多样化，故障强度分级，记录时间精确到分钟级别，保持故障记录与实际注入的一致性。