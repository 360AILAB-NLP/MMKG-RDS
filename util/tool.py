from hashlib import md5


def compute_args_hash(*args) -> str:
    """
    Compute a unique hash for a given set of arguments.

    The hash is computed by concatenating the arguments and then hashing the resulting string.
    """
    args_str = "".join([str(arg) for arg in args])
    args_str = args_str.lower()
    
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        # Handle surrogate characters and other encoding issues
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.

    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    
    return prefix + compute_args_hash(content)

from contextlib import contextmanager
import time

@contextmanager
def stage_context(name: str, stage_num: int):
    start_time = time.time()
    print(f"🚀 阶段 {stage_num}: {name}")
    try:
        yield
        elapsed = time.time() - start_time
        print(f"✅ 阶段 {stage_num} 完成 | 耗时: {elapsed:.2f}秒")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ 阶段 {stage_num} 失败 | 耗时: {elapsed:.2f}秒")
        raise

# import time
# from functools import wraps
# from typing import Dict, Any

# class StageTracker:
#     def __init__(self):
#         self.stage_count = 0
#         self.start_time = None
    
#     def stage(self, name: str):
#         """装饰器：跟踪阶段执行和耗时"""
#         def decorator(func):
#             @wraps(func)
#             def sync_wrapper(*args, **kwargs):
#                 return self._execute_stage(name, func, *args, **kwargs)
            
#             @wraps(func)
#             async def async_wrapper(*args, **kwargs):
#                 return await self._execute_stage_async(name, func, *args, **kwargs)
            
#             return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
#         return decorator
    
#     def _execute_stage(self, name: str, func, *args, **kwargs):
#         self.stage_count += 1
#         print(f"🚀 阶段 {self.stage_count}: {name}")
        
#         start_time = time.time()
#         try:
#             result = func(*args, **kwargs)
#             elapsed = time.time() - start_time
#             print(f"✅ 阶段 {self.stage_count} 完成 | 耗时: {elapsed:.2f}秒")
#             return result
#         except Exception as e:
#             elapsed = time.time() - start_time
#             print(f"❌ 阶段 {self.stage_count} 失败 | 耗时: {elapsed:.2f}秒 | 错误: {e}")
#             raise
    
#     async def _execute_stage_async(self, name: str, func, *args, **kwargs):
#         self.stage_count += 1
#         print(f"🚀 阶段 {self.stage_count}: {name}")
        
#         start_time = time.time()
#         try:
#             result = await func(*args, **kwargs)
#             elapsed = time.time() - start_time
#             print(f"✅ 阶段 {self.stage_count} 完成 | 耗时: {elapsed:.2f}秒")
#             return result
#         except Exception as e:
#             elapsed = time.time() - start_time
#             print(f"❌ 阶段 {self.stage_count} 失败 | 耗时: {elapsed:.2f}秒 | 错误: {e}")
#             raise
