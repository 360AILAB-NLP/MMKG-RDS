"""
信息模糊化器
用于对采样的信息进行模糊化处理
如实体模糊化、属性模糊化
包含基于模板和基于LLM改写
"""

# 模糊化，我针对输入的文本进行模糊化，如时间、数字、地点、人名、组织名、实体名、属性、实体关系、实体描述等
import re
import json
import ast
import logging
import random
from typing import *
from datetime import datetime, timedelta
from data_synthesis.trace_generate import TraceSelectOutput, Trace  # 替换成你实际的模块路径

logger = logging.getLogger(__name__)

class InformationBlur:
    """信息模糊化器"""
    def __init__(self, blur_probability: float = 0.5):
        self.blur_probability = blur_probability
        # self.time_templates = {
        #     'year': [
        #         '21世纪初期', '21世纪10年代', '21世纪20年代初',
        #         '近十年', '过去几年', '最近几年',
        #         '本世纪初', '本世纪中期', '当代'
        #     ],
        #     'month': [
        #         '春季', '夏季', '秋季', '冬季',
        #         '上半年', '下半年', '年中', '年末',
        #         '春夏之交', '秋冬之际', '某个季度'
        #     ],
        #     'day': [
        #         '月初', '月中', '月末', '某一天',
        #         '周初', '周中', '周末', '某个时期'
        #     ]
        # }
        
        # 数字模糊化模板
        self.number_templates = [
            '约{}', '大约{}', '接近{}', '超过{}',
            '不到{}', '几{}', '多个{}', '少数{}'
        ]
        # 人名模糊化模板
        self.person_templates = [
            '某位{}', '一位{}', '著名的{}', '知名{}',
            '相关{}', '业内{}', '专业{}'
        ]

        # 组织模糊化模板
        self.organization_templates = [
            '某{}', '一家{}', '知名{}', '大型{}',
            '国际{}', '领先{}', '专业{}'
        ]

        # 地点模糊化模板
        self.location_templates = [
            '某个{}', '一个{}', '位于{}的', '来自{}的',
            '{}地区', '{}附近', '{}周边'
        ]

    def blur(self, trace_output: TraceSelectOutput) -> TraceSelectOutput:
        """模糊化 TraceSelectOutput 中的实体名称"""
        try:
            for trace in trace_output.paths:
                self._blur_trace(trace)
            logger.info("完成实体名称模糊化")
            return trace_output
        except Exception as e:
            logger.error(f"模糊化失败: {e}")
            return trace_output

    def _blur_trace(self, trace: Trace):
        """原地逐节点打标签+blur"""
        for node in trace.nodes.values():
            original_name = node.get('name')
            should_blur = random.random() < self.blur_probability
            node['is_blurry'] = should_blur
            if should_blur and node.get('type', 'unknown')=='Entity':
                node['blurry_name'] = self._blur_name(original_name, node.get('entity_type', 'unknown'))
            # 不模糊就啥也不写，保持原样

    def _blur_name(self, name: str, entity_types: List[str]) -> str:
        """根据实体类型模糊化名称"""
        try: 
            if isinstance(entity_types, str):  # 尝试将字符串转换为列表
                entity_types = ast.literal_eval(entity_types)  # json.loads(name)
            entity_types = set(entity_types)
            
        except:
            entity_types = [entity_types]
        if entity_types & {'person', 'researcher', 'scientist', 'author'}:
            templates = self.person_templates
            category = '研究者'
        elif entity_types & {'organization', 'company', 'institution'}:
            templates = self.organization_templates
            category = '机构'
        elif entity_types & {'location', 'place', 'city', 'country'}:
            templates = self.location_templates
            category = '地区'
        # elif entity_types & {'time', 'date', 'year', 'month', 'day'}:
        #     templates = self.time_templates
            category = '时间'
        elif entity_types & {'number', 'count', 'amount', 'quantity'}:
            templates = self.number_templates
            category = '数量'
        else:
            return f"某个{entity_type}"
        template = random.choice(templates)
        return template.format(category)
    
    # 大模型模糊改写,还没有写完
    def llm_blur(self, trace_output: TraceSelectOutput) -> TraceSelectOutput:
        pass