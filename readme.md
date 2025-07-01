# TraceTicket

Train Ticket 系统Trace采集与分析工具，用于异常检测研究。

## 项目结构

```
TraceTicket/
├── train-ticket-trace-collect/    # Trace数据采集与标签生成
│   ├── trace_collector.py         # Trace数据采集（输出11列基础数据）
│   ├── trace_label_processor.py   # 异常标签生成器（基于故障注入记录）
│   ├── metrics_collector.py       # 系统指标采集
│   └── config.py                  # 配置文件
├── train-ticket-auto-query/       # Train-Ticket自动查询压测
│   ├── config.py                  # 压测强度配置
│   ├── queries.py                 # 核心查询类
│   ├── atomic_queries.py          # 原子操作查询
│   ├── scenarios.py               # 业务场景组合
│   ├── normal_request_manager.py  # 请求管理器
│   ├── run.py                     # 启动脚本
│   └── query_*.py                 # 各种业务查询脚本
├── fault_injection_records/       # 故障注入记录目录
│   └── fault_records_YYYYMMDD.json # 按日期存储的故障记录
├── main_controller.py             # 主控制器（一键启动）
└── trace/                         # 数据输出目录
    └── YYYY-MM-DD/
        └── csv/                   # 按分钟存储的数据文件
```

## 功能概述

### 主控制器 (main_controller.py)
- 一键启动整个异常检测系统
- 协调压测、数据采集、标签生成和异常检测
- 实时监控系统状态和检测结果
- 自动验证检测准确性

### 业务流量生成模块 (train-ticket-auto-query)
- config.py: 压测强度配置（轻度/中度/重度三种预设）
- queries.py: 核心查询类，封装所有业务 API 调用
- atomic_queries.py: 原子级别的 API 操作函数
- scenarios.py: 组合业务场景（查票→订票→支付→取票等）
- normal_request_manager.py: 多线程请求管理，模拟真实用户行为
- query_*.py: 各种专项业务查询脚本

### 数据采集 (trace_collector.py)
- 连接 Jaeger API 采集 Train Ticket 系统链路数据
- 输出 11 列基础格式 CSV（移除了标签计算）
- 支持跨日期持续运行

### 异常标签生成 (trace_label_processor.py)
- 基于故障注入记录生成异常标签
- 将 11 列基础数据扩展为 14 列标签化数据
- 支持时间范围匹配和准确性验证
- 如果没有故障记录，默认标记为正常

### 系统指标采集 (metrics_collector.py)
- 从 Prometheus 采集系统性能指标
- 智能检测可用指标并自动配置
- 高频采集（默认15秒间隔）

## 数据格式

### 11列基础格式
```csv
traceIdHigh,traceIdLow,parentSpanId,spanId,startTime,duration,nanosecond,DBhash,status,operationName,serviceName
```

### 14列标签化格式
在11列基础上增加：
- `nodeLatencyLabel`: 节点延迟异常标签 (0=正常, 1=异常)
- `graphLatencyLabel`: 图延迟异常标签 (0=正常, 1=异常)  
- `graphStructureLabel`: 图结构异常标签 (固定为0，不考虑结构异常)

## 故障注入记录文件

### 目录结构
```
fault_injection_records/
└── fault_records_YYYYMMDD.json    # 按日期命名的故障记录文件
```

### 文件命名规则
- 格式：`fault_records_YYYYMMDD.json`
- 例如：`fault_records_20250618.json`（对应2025-06-18的故障记录）

### 文件内容格式
```json
[
  {
    "start_time": "2025-06-18 14:30:00",
    "end_time": "2025-06-18 14:32:00",
    "minute_key": "14_30",
    "fault_type": "latency_injection",
    "description": "注入网络延迟故障",
    "intensity": "high"
  },
  {
    "start_time": "2025-06-18 15:45:00", 
    "end_time": "2025-06-18 15:47:00",
    "minute_key": "15_45",
    "fault_type": "service_unavailable",
    "description": "模拟服务不可用故障",
    "intensity": "medium"
  }
]
```

### 字段说明
- `start_time`: 故障开始时间（格式：YYYY-MM-DD HH:MM:SS）
- `end_time`: 故障结束时间（格式：YYYY-MM-DD HH:MM:SS）
- `minute_key`: 分钟键（格式：HH_MM），用于快速匹配数据文件
- `fault_type`: 故障类型（如：latency_injection, service_unavailable等）
- `description`: 故障描述信息
- `intensity`: 故障强度（如：low, medium, high）

### 标签生成规则
1. **有故障记录**：trace时间落在故障时间范围内 → `nodeLatencyLabel=1, graphLatencyLabel=1, graphStructureLabel=0`
2. **无故障记录**：找不到对应的故障记录文件或记录 → `nodeLatencyLabel=0, graphLatencyLabel=0, graphStructureLabel=0`

## 快速使用

### 1. 配置环境
```bash
pip install requests pandas numpy networkx
```

### 2. 配置压测强度（可选）
编辑 `train-ticket-auto-query/config.py` 修改压测参数：
```python
DEFAULT_PRESSURE_LEVEL = "medium"  # light/medium/heavy
```

### 3. 准备故障记录（可选）
如需验证异常检测准确性，创建故障记录文件：
```bash
mkdir fault_injection_records
# 创建对应日期的故障记录文件
```

### 4. 一键启动系统
```bash
python main_controller.py
```

### 5. 或分步骤运行

#### 生成业务流量
```bash
python train-ticket-auto-query/run.py
```

#### 采集数据
```bash
# 测试连接
python train-ticket-trace-collect/trace_collector.py --test

# 采集数据
python train-ticket-trace-collect/trace_collector.py --duration 60
```

#### 生成异常标签
```bash
# 处理所有数据
python train-ticket-trace-collect/trace_label_processor.py

# 处理特定日期
python train-ticket-trace-collect/trace_label_processor.py --date 2025-06-18

# 处理特定文件
python train-ticket-trace-collect/trace_label_processor.py --file trace/2025-06-18/csv/14_30.csv
```

## 输出目录

```
trace/
└── YYYY-MM-DD/
    ├── csv/                      # 分钟级数据文件
    │   ├── HH_MM.csv            # 11列→14列数据
    │   └── HH_MM.label_processed # 标签处理完成标志
    └── mapping_YYYYMMDD.json    # 服务和操作名映射表

metrics/
└── YYYY-MM-DD/
    └── csv/
        └── HH_MM.csv            # 系统指标数据
```

## 配置说明

### Jaeger 连接 (train-ticket-trace-collect/config.py)
```python
JAEGER_HOST = "192.168.1.102"
JAEGER_PORT = 31686
```

### Train Ticket 服务地址 (train-ticket-auto-query)
```python
base_address = "http://192.168.1.102:32677"
auth_address = "http://192.168.1.102:30530"
```

### 压测强度配置 (train-ticket-auto-query/config.py)
```python
PRESSURE_TEST_PRESETS = {
    "light": {"thread_count": 3, "request_interval": 2.0},
    "medium": {"thread_count": 5, "request_interval": 1.0},
    "heavy": {"thread_count": 8, "request_interval": 0.5}
}
```