import json
from typing import *
import logging
import re
import ast
logger = logging.getLogger(__name__)

def parse_json(text: str, samrt_json_parsers: List=[lambda x: json.loads(x),lambda x: ast.literal_eval(x),lambda x: parse_json_like(x)]) -> Dict[str, Any]:
    """
    尝试使用多种JSON解析策略解析文本。
    
    Args:
        text (str): 要解析为JSON的文本
        samrt_json_parsers (List): JSON解析策略列表
        
    Return:
        Dict[str, Any]: 解析后的字典，如果所有解析器都失败则返回空字典
    """
    func_len = len(samrt_json_parsers)
    for idx, parser in enumerate(samrt_json_parsers):
        try:
            return parser(text)
        except Exception as e:
            logger.warning(f"函数 {idx+1}/{func_len} 解析JSON失败, {e}")
            if idx == func_len - 1:  # 最后一个解析器失败
                logger.error(f"解析JSON失败: {text}")
            continue
    return {}

def parse_json_like(text: str) -> Dict[str, Any]:
    """
    将类似JSON的字符串解析为键值对。
    
    函数用途:
    解析看起来像JSON但可能格式不严格的内容为字典。
    每行应包含由第一个冒号分隔的键值对。
    去除键和值的空白字符、引号和尾随逗号。
    
    Args:
        text (str): 要解析的文本，每行包含一个键值对
        
    Return:
        Dict[str, Any]: 包含解析后键值对的字典
    """
    lines = text.splitlines()
    result: Dict[str, Any] = {}

    # 正则表达式用于去除开头/结尾的引号或逗号
    strip_re = re.compile(r'^["\']?|["\']?,?$')

    for raw in lines:
        raw = raw.strip()
        if not raw or raw in '{}[]':        # 跳过空行或纯括号
            continue

        if ':' not in raw:                  # 跳过没有冒号的行
            continue

        key_part, val_part = raw.split(':', 1)   # 只分割一次
        key = strip_re.sub('', key_part.strip())
        val = strip_re.sub('', val_part.strip())

        if key:                             # 如果键为空则丢弃
            result[key] = val
    return result