# 压测配置预设
PRESSURE_TEST_PRESETS = {
    "light": {
        "name": "轻度压测",
        "thread_count": 3,
        "request_interval": 2.0,
        "description": "适用于开发环境和基础功能测试"
    },
    "medium": {
        "name": "中度压测", 
        "thread_count": 5,
        "request_interval": 1.0,
        "description": "适用于集成测试和异常检测实验"
    },
    "heavy": {
        "name": "重度压测",
        "thread_count": 8,
        "request_interval": 0.5,
        "description": "适用于性能测试和压力测试场景"
    }
}

# 默认压测等级
DEFAULT_PRESSURE_LEVEL = "medium"

def get_pressure_config(level=None):
    """
    获取指定等级的压测配置。
    
    参数:
        level (str): 压测等级 (light/medium/heavy)
                    如果为None，则使用DEFAULT_PRESSURE_LEVEL
    
    返回:
        dict: 包含thread_count和request_interval的配置字典
    
    异常:
        ValueError: 当指定的等级在预设中不存在时抛出
    """
    if level is None:
        level = DEFAULT_PRESSURE_LEVEL
    
    if level not in PRESSURE_TEST_PRESETS:
        available_levels = list(PRESSURE_TEST_PRESETS.keys())
        raise ValueError(f"无效的压测等级 '{level}'. 可用等级: {available_levels}")
    
    return PRESSURE_TEST_PRESETS[level]

def list_available_levels():
    """
    列出所有可用的压测等级及其描述。
    
    返回:
        dict: 等级名称到描述的映射字典
    """
    return {level: config["description"] for level, config in PRESSURE_TEST_PRESETS.items()}
