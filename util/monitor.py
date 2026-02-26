import json
import time
from functools import wraps
from datetime import datetime
import os

def monitor_function(log_file="function_monitor.json"):
    """
    监控函数执行的装饰器
    记录函数的输入、输出、起始时间、终止时间和执行时间
    
    Args:
        log_file (str): 日志文件路径，默认为 function_monitor.json
    """
    def decorator(func):
        # 确保日志文件存在
        if not os.path.exists(log_file):
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 记录开始时间
            start_time = time.perf_counter()
            start_timestamp = datetime.now().isoformat()
            
            try:
                # 执行函数并获取返回值
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                end_timestamp = datetime.now().isoformat()
                execution_time = end_time - start_time
                error = None
            except Exception as e:
                # 如果函数执行出错
                end_time = time.perf_counter()
                end_timestamp = datetime.now().isoformat()
                execution_time = end_time - start_time
                result = None
                error = str(e)
                raise  # 重新抛出异常
            
            # 构建日志记录
            log_entry = {
                "function_name": func.__name__,
                "timestamp": datetime.now().isoformat(),
                "start_time": start_timestamp,
                "end_time": end_timestamp,
                "execution_time_seconds": round(execution_time, 6),
                "input_arguments": {
                    "args": args,
                    "kwargs": kwargs
                },
                "output_result": result,
                "error": error,
                "status": "success" if error is None else "error"
            }
            
            # 将记录保存到JSON文件
            save_to_json(log_entry, log_file)
            
            return result
        
        return wrapper
    return decorator

def save_to_json(log_entry, log_file):
    """将日志记录保存到JSON文件"""
    try:
        # 读取现有数据
        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
        
        # 添加新记录
        data.append(log_entry)
        
        # 写回文件
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    except Exception as e:
        print(f"保存日志时出错: {e}")

# 高级版本：支持自定义序列化器
def advanced_monitor_function(log_file="advanced_monitor.json", max_file_size_mb=10):
    """
    高级监控装饰器，支持文件大小限制和自定义序列化
    
    Args:
        log_file (str): 日志文件路径
        max_file_size_mb (int): 最大文件大小(MB)，超过则归档
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 检查文件大小
            check_file_size(log_file, max_file_size_mb)
            
            start_time = time.perf_counter()
            start_timestamp = datetime.now().isoformat()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                error = None
            except Exception as e:
                end_time = time.perf_counter()
                result = None
                error = str(e)
                raise
            
            # 自定义序列化函数参数
            def custom_serializer(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                    return list(obj)
                else:
                    return str(obj)
            
            log_entry = {
                "function_name": func.__name__,
                "module": func.__module__,
                "timestamp": datetime.now().isoformat(),
                "start_time": start_timestamp,
                "end_time": datetime.now().isoformat(),
                "execution_time_seconds": round(end_time - start_time, 6),
                "input_arguments": {
                    "args": [custom_serializer(arg) for arg in args],
                    "kwargs": {k: custom_serializer(v) for k, v in kwargs.items()}
                },
                "output_result": custom_serializer(result) if result is not None else None,
                "error": error,
                "status": "success" if error is None else "error"
            }
            
            save_to_json(log_entry, log_file)
            return result
        
        return wrapper
    return decorator

def check_file_size(log_file, max_size_mb):
    """检查文件大小，如果超过限制则归档"""
    if os.path.exists(log_file):
        file_size = os.path.getsize(log_file) / (1024 * 1024)  # 转换为MB
        if file_size > max_size_mb:
            # 归档旧文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = f"{log_file}.archive.{timestamp}"
            os.rename(log_file, archive_file)
            # 创建新文件
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)