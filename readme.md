# TraceTicket

Train Ticket 系统Trace采集与分析工具，用于异常检测研究。

## 项目结构

```
TraceTicket/
├── train-ticket-trace-collect/    # Trace数据采集与后处理
│   ├── trace_collector.py         # Trace数据采集
│   ├── graph_post_processor.py    # 图分析后处理
│   └── config.py                  # 配置文件
├── train-ticket-auto-query/       # Train-Ticket自动查询
│   ├── queries.py                 # 核心查询类
│   ├── atomic_queries.py          # 原子操作查询
│   ├── scenarios.py               # 业务场景组合
│   ├── normal_request_manager.py  # 请求管理器
│   ├── run.py                     # 启动脚本
│   └── query_*.py                 # 各种业务查询脚本
└── trace_output/                  # 数据输出
```

## 功能概述
### 业务流量生成模块 (train-ticket-auto-query)
- queries.py: 核心查询类，封装所有业务 API 调用
- atomic_queries.py: 原子级别的 API 操作函数
- scenarios.py: 组合业务场景（查票→订票→支付→取票等）
- normal_request_manager.py: 多线程请求管理，模拟真实用户行为
- query_*.py: 各种专项业务查询脚本

### 数据采集 (trace_collector.py)
- 连接 Jaeger API 采集 Train Ticket 系统链路数据
- 输出 12 列标准格式 CSV
- 支持跨日期持续运行

### 图分析后处理 (graph_post_processor.py)
- 基于调用图分析添加延迟和结构复杂度特征
- 将 12 列数据扩展为 14 列
- 支持动态阈值优化标签分布

## 数据格式

### 12列基础格式
```csv
traceIdHigh,traceIdLow,parentSpanId,spanId,startTime,duration,nanosecond,DBhash,status,operationName,serviceName,nodeLatencyLabel
```

### 14列增强格式
在12列基础上增加：
- `graphLatencyLabel`: 图延迟标签 (0=正常, 1=中等, 2=高延迟)
- `graphStructureLabel`: 图结构标签 (0=简单, 1=中等, 2=复杂)

## 快速使用

### 1. 配置环境
```bash
pip install requests pandas numpy networkx
```

### 2. 生成业务流量
```bash
# 启动多线程压测（生成真实业务流量）
python train-ticket-auto-query/run.py
```

### 3. 采集数据
```bash
# 测试连接
python train-ticket-trace-collect/trace_collector.py --test

# 采集1小时数据
python train-ticket-trace-collect/trace_collector.py --duration 60
```

### 4. 图分析后处理
```bash
# 处理所有数据
python train-ticket-trace-collect/graph_post_processor.py

# 处理特定日期
python train-ticket-trace-collect/graph_post_processor.py --date 2025-06-18
```

## 输出目录

```
trace_output/
└── YYYY-MM-DD/
    ├── csv/                    # 12列原始数据
    ├── csv_enhanced/           # 14列增强数据
    ├── json/                   # JSON格式数据
    ├── mapping_YYYYMMDD.json   # 编码映射表
    └── graph_analysis_stats.json # 分析统计
```

## 配置说明
Jaeger 连接 (train-ticket-trace-collect/config.py)
```python
JAEGER_HOST = "192.168.1.102"
JAEGER_PORT = 31686
```
Train Ticket 服务地址 (train-ticket-auto-query)
```python
base_address = "http://192.168.1.102:32677"
auth_address = "http://192.168.1.102:30530"
```