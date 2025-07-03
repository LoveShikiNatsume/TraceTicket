#!/usr/bin/env python3
"""
在线异常检测服务主文件
"""

import asyncio
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .models.detector import TraceAnomalyDetector
from .config import config

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic模型
class TraceSpan(BaseModel):
    spanID: str
    parentSpanID: Optional[str] = ""
    operationName: str
    serviceName: str
    startTime: int
    duration: int
    tags: List[dict] = []

class TraceData(BaseModel):
    traceID: str
    spans: List[TraceSpan]
    processes: dict = {}

class DetectionRequest(BaseModel):
    traces: List[TraceData]
    batch_size: Optional[int] = Field(default=32, ge=1, le=config.MAX_BATCH_SIZE)

class AnomalyResult(BaseModel):
    traceID: str
    is_anomaly: bool
    anomaly_type: str
    confidence: float
    details: dict
    processing_time_ms: float

class DetectionResponse(BaseModel):
    results: List[AnomalyResult]
    total_processed: int
    total_anomalies: int
    total_processing_time_ms: float

# FastAPI应用
app = FastAPI(
    title="TraceVAE Online Anomaly Detector",
    description="基于TraceVAE的实时微服务追踪异常检测服务",
    version="1.0.0",
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检测器实例
detector: Optional[TraceAnomalyDetector] = None

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global detector
    
    logger.info("🚀 启动在线异常检测服务...")
    
    try:
        # 验证配置
        if not config.validate_paths():
            raise RuntimeError("配置验证失败")
        
        # 获取绝对路径
        model_path = config.get_absolute_path(config.MODEL_PATH)
        config_dir = config.get_absolute_path(config.CONFIG_DIR)
        
        logger.info(f"模型路径: {model_path}")
        logger.info(f"配置目录: {config_dir}")
        
        # 初始化检测器
        detector = TraceAnomalyDetector(
            model_path=model_path,
            config_dir=config_dir,
            device=config.DEVICE
        )
        
        logger.info("✅ 异常检测服务启动成功")
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    global detector
    if detector and hasattr(detector, 'executor'):
        detector.executor.shutdown(wait=True)
    logger.info("🔽 异常检测服务已关闭")

@app.get("/health")
async def health_check():
    """健康检查"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "TraceVAE Online Detector",
        "version": "1.0.0"
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_anomalies(request: DetectionRequest):
    """异常检测接口"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # 转换输入数据
        traces_dict = []
        for trace in request.traces:
            trace_dict = {
                'traceID': trace.traceID,
                'spans': [span.dict() for span in trace.spans],
                'processes': trace.processes
            }
            traces_dict.append(trace_dict)
        
        # 执行异步检测
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            detector.executor,
            detector.detect_anomalies,
            traces_dict
        )
        
        # 统计结果
        total_processed = len(results)
        total_anomalies = sum(1 for r in results if r['is_anomaly'])
        total_processing_time = sum(r['processing_time_ms'] for r in results)
        
        logger.info(
            f"✅ 检测完成: {total_processed} traces, "
            f"{total_anomalies} anomalies, "
            f"{total_processing_time:.2f}ms"
        )
        
        return DetectionResponse(
            results=[AnomalyResult(**r) for r in results],
            total_processed=total_processed,
            total_anomalies=total_anomalies,
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ 检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """获取服务统计信息"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "model_loaded": detector.model is not None,
        "device": str(detector.device),
        "threshold": detector.anomaly_threshold,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )