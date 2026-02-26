"""
关系数据存储处理模块
"""


import os
import json
from typing import List, Dict, Any
from .node import NodeType
from .modal import *
import re

class EdgeData:
    def __init__(self):
        """
        初始化EdgeData对象，包含空的关系列表和预定义的正则表达式模式
        用于匹配文本中的表格、图像和公式引用
        """
        self.relation_list = []
        self.table_label_match_pattern = [r'\b(?:tab|table|表|表格)\s*(\d+)\b']
        self.image_label_match_pattern = [r'\b(?:fig|figure|图|图片)\s*(\d+)\b']
        self.formula_label_match_pattern = [
        # 中文前缀: 公式1, 等式1, 公式(1), 等式(1)
        r'(?:公式|等式)\s*[（(]?\s*(\d+(?:\.\d+)?[a-zA-Z]?)\s*[）)]?',
        # 英文前缀: Equation 1, Eq. 1, Eq (1)
        r'(?:Equation|Eq\.?)\s*[（(]?\s*(\d+(?:\.\d+)?[a-zA-Z]?)\s*[）)]?',
        # latex格式的 \label{1}
        r'\\label\s*\{([^}]*)\}',
        # latex格式的 \tag{1}
        r'\\tag\s*\{([^}]*)\}',
        # latex格式的引用
        r'\\eqref\s*\{([^}]*)\}',
        r'\\ref\s*\{([^}]*)\}.*?(?:equation|align|eqnarray)',
    ]

    def load(self, edge_list_or_path: str | List[Dict[str, Any]]):
        """
        从JSON文件路径或字典列表加载关系数据
        
        Args:
            edge_list_or_path (str | List[Dict[str, Any]]): JSON文件路径或表示关系的字典列表
        """
        if isinstance(edge_list_or_path, str):
            path = edge_list_or_path
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = edge_list_or_path
        
        self.relation_list = data
    
    def add_edge(self, source_id: str, target_id: str, relation: str, desc: str = "",attr: Dict = {}):
        """
        在两个节点/实体之间添加新的边/关系
        
        Args:
            source_id (str): 源节点/实体的ID
            target_id (str): 目标节点/实体的ID
            relation (str): 源和目标之间的关系类型
            desc (str, optional): 关系的描述. 默认为 ""
            attr (Dict, optional): 边的附加属性. 默认为 {}
        """
        self.relation_list.append({"head": source_id, "tail": target_id, "relation": relation, "desc": desc, "attr": attr})

    def regex_match(self, text: str, patterns: List[str]):
        """
        在提供的文本中查找给定正则表达式模式的所有匹配项
        
        Args:
            text (str): 要搜索模式的文本
            patterns (List[str]): 要匹配的正则表达式模式列表
            
        Returns:
            List[str]: 唯一匹配字符串的列表
        """
        text = str(text)
        match_list = []
        for p in patterns:
            match = re.findall(p, text, re.IGNORECASE)
            match_list.extend([m.strip() for m in match])
        return list(set(match_list))
    
    # 修整引用了被合并了的实体的entity_id, 针对边
    def correct_merged_id(self, entity_id_map: Dict[str, str]):
        """
        根据映射字典更新关系中的头和尾ID
        用于修正已合并的实体ID
        
        Args:
            entity_id_map (Dict[str, str]): 将旧实体ID映射到新实体ID的字典
        """
        for idx, relation in enumerate(self.relation_list):
            self.relation_list[idx]["head"] = entity_id_map.get(relation["head"], relation["head"])
            self.relation_list[idx]["tail"] = entity_id_map.get(relation["tail"], relation["tail"])

    def correct_merged_relation(self, relation2std_rel: Dict[str, str]):
        """
        根据映射字典更新关系中的关系类型
        用于修正已合并的关系类型
        
        Args:
            relation2std_rel (Dict[str, str]): 将旧关系类型映射到新关系类型的字典
        """
        for idx, relation in enumerate(self.relation_list):
            self.relation_list[idx]["relation"] = relation2std_rel.get(relation["relation"], relation["relation"])

    def save(self, path: str = "./edge_data.json"):
        """
        将关系列表保存到JSON文件
        
        Args:
            path (str, optional): 保存数据的文件路径. 默认为 "./edge_data.json"
        """
        self.deduplication()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.relation_list, f, ensure_ascii=False, indent=2)

    # 边去重
    def deduplication(self):
        """
        去除重复的边
        """
        rel_dict = {}
        for relation in self.relation_list:
            rel_dict[f"{relation['head']}{relation['relation']}{relation['tail']}"] = relation
        
        self.relation_list = list(rel_dict.values())
