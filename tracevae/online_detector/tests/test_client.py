#!/usr/bin/env python3
"""
在线异常检测服务测试客户端
"""

import asyncio
import aiohttp
import json
import time
import random
from typing import List, Dict

class TraceGenerator:
    """测试用trace数据生成器"""
    
    def generate_normal_trace(self) -> Dict:
        """生成正常trace"""
        trace_id = f"trace_{random.randint(1000, 9999)}"
        base_time = int(time.time() * 1000000)
        
        spans = []
        for i in range(random.randint(2, 5)):
            span = {
                "spanID": f"span_{i}_{random.randint(100, 999)}",
                "parentSpanID": f"span_{i-1}_{random.randint(100, 999)}" if i > 0 else "",
                "operationName": random.choice([
                    "GET /api/users", "POST /api/orders", "GET /api/products",
                    "PUT /api/inventory", "DELETE /api/cache"
                ]),
                "serviceName": random.choice([
                    "user-service", "order-service", "product-service",
                    "inventory-service", "cache-service"
                ]),
                "startTime": base_time + i * 1000,
                "duration": random.randint(1000, 50000),
                "tags": [
                    {"key": "http.status_code", "value": 200},
                    {"key": "http.method", "value": "GET"}
                ]
            }
            spans.append(span)
        
        return {
            "traceID": trace_id,
            "spans": spans,
            "processes": {
                f"p{i}": {"serviceName": span["serviceName"]}
                for i, span in enumerate(spans)
            }
        }
    
    def generate_anomaly_trace(self) -> Dict:
        """生成异常trace"""
        trace = self.generate_normal_trace()
        
        # 随机创建异常
        anomaly_type = random.choice(["latency", "error", "structure"])
        
        if anomaly_type == "latency":
            # 增加延迟
            for span in trace["spans"]:
                if random.random() < 0.5:
                    span["duration"] *= random.randint(5, 20)
        
        elif anomaly_type == "error":
            # 添加错误状态
            for span in trace["spans"]:
                if random.random() < 0.3:
                    span["tags"] = [
                        {"key": "http.status_code", "value": random.choice([500, 503, 404])},
                        {"key": "error", "value": True}
                    ]
        
        elif anomaly_type == "structure":
            # 删除一些spans（结构异常）
            if len(trace["spans"]) > 1:
                num_to_remove = random.randint(1, len(trace["spans"]) // 2)
                for _ in range(num_to_remove):
                    if trace["spans"]:
                        trace["spans"].pop(random.randint(0, len(trace["spans"]) - 1))
        
        return trace

async def test_health_check(base_url: str = "http://localhost:8000"):
    """测试健康检查"""
    print("🏥 测试健康检查...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 服务健康: {result}")
                    return True
                else:
                    print(f"❌ 健康检查失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 健康检查连接失败: {e}")
            return False

async def test_detection(base_url: str = "http://localhost:8000", num_traces: int = 10):
    """测试异常检测"""
    print(f"🔍 测试异常检测 ({num_traces} traces)...")
    
    generator = TraceGenerator()
    
    # 生成测试数据
    traces = []
    normal_count = int(num_traces * 0.7)
    anomaly_count = num_traces - normal_count
    
    for _ in range(normal_count):
        traces.append(generator.generate_normal_trace())
    for _ in range(anomaly_count):
        traces.append(generator.generate_anomaly_trace())
    
    # 打乱顺序
    random.shuffle(traces)
    
    # 请求数据
    request_data = {
        "traces": traces,
        "batch_size": 32
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            start_time = time.time()
            
            async with session.post(
                f"{base_url}/detect",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                request_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    
                    print("🎯 检测结果:")
                    print(f"  总处理: {result['total_processed']} traces")
                    print(f"  异常数量: {result['total_anomalies']}")
                    print(f"  服务端处理时间: {result['total_processing_time_ms']:.2f}ms")
                    print(f"  总请求时间: {request_time:.2f}ms")
                    
                    print("\n📊 详细结果:")
                    for i, res in enumerate(result['results'][:5]):  # 只显示前5个
                        status = "🚨 异常" if res['is_anomaly'] else "✅ 正常"
                        print(f"  Trace {i+1}: {status} - {res['anomaly_type']} "
                              f"(置信度: {res['confidence']:.3f})")
                    
                    if len(result['results']) > 5:
                        print(f"  ... 还有 {len(result['results']) - 5} 个结果")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 检测请求失败: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            print(f"❌ 检测请求连接失败: {e}")
            return False

async def test_stats(base_url: str = "http://localhost:8000"):
    """测试统计信息"""
    print("📊 测试统计信息...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/stats") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 服务状态: {result}")
                    return True
                else:
                    print(f"❌ 统计信息获取失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 统计信息连接失败: {e}")
            return False

async def run_all_tests():
    """运行所有测试"""
    print("🧪 开始测试TraceVAE在线异常检测服务...")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # 1. 健康检查
    health_ok = await test_health_check(base_url)
    if not health_ok:
        print("❌ 服务未就绪，请先启动服务")
        return
    
    print()
    
    # 2. 统计信息
    await test_stats(base_url)
    print()
    
    # 3. 异常检测测试
    await test_detection(base_url, num_traces=10)
    print()
    
    # 4. 批量测试
    print("📦 批量测试...")
    await test_detection(base_url, num_traces=50)
    
    print("\n✅ 所有测试完成!")

if __name__ == "__main__":
    asyncio.run(run_all_tests())