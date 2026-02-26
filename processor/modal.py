"""
处理器基类，用于处理各类数据并调用LLM模型。
"""

from util.tool import compute_mdhash_id
from .node import PREFIX_MAP
from prompts.dataprocess_prompt import PROMPTS
from .node import NodeType, NodeData
from typing import Dict, List, Any, Optional, Callable
import json
import ast
import asyncio
import logging
import re
logger = logging.getLogger(__name__)
from tqdm.asyncio import tqdm_asyncio
from util.jsonparser import parse_json_like, parse_json





class BaseProcessor:
    def __init__(self, llm_model_func, vision_model_func, nodedata: NodeData):
        """
        使用所需函数和数据结构初始化基础处理器。
        
        Args:
            llm_model_func: 调用LLM模型的函数
            vision_model_func: 调用视觉模型处理图像的函数
            nodedata (NodeData): 存储处理节点的数据结构
        """
        self.llm_model_func = llm_model_func
        self.vision_model_func = vision_model_func
        self.nodedata = nodedata
        self.json_tag = "myjson"
        self.json_tag_start = f"<{self.json_tag}>"
        self.json_tag_end = f"</{self.json_tag}>"
        self.samrt_json_parsers = [
            lambda x: json.loads(x),
            lambda x: ast.literal_eval(x),
            lambda x: parse_json_like(x)
        ]
        self.need_field_call_llm = [
            "type",
        ]
    def cheak_input(self, text: str) -> bool:
        """
        检查输入是否满足调用LLM的要求（占位符）。
        
        Args:
            text (str): 要检查的输入文本
            
        Return:
            bool: 总是返回True（目前未实现）
        """
        # todo
        # 检查llm调用是否满足要求，包含type字段
        return True

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        尝试使用多种JSON解析策略解析文本。
        
        Args:
            text (str): 要解析为JSON的文本
            
        Return:
            Dict[str, Any]: 解析后的字典，如果所有解析器都失败则返回空字典
        """
        return parse_json(text=text, samrt_json_parsers=self.samrt_json_parsers)
        # try:
        #     return parse_json(text=text, samrt_json_parsers=self.samrt_json_parsers)
        # except Exception as e:
        #     logger.error(f"解析JSON失败: {text}")
        #     return {}
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数据（抽象方法，需由子类实现）。
        
        Args:
            data (Dict[str, Any]): 要处理的数据
            
        Return:
            Dict[str, Any]: 处理后的数据
            
        Exception:
            NotImplementedError: 必须由子类实现
        """
        raise NotImplementedError("子类必须实现此方法")
    
    async def call_llm(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用提供的数据调用LLM（抽象方法，需由子类实现）。
        
        Args:
            data (Dict[str, Any]): 发送到LLM的数据
            
        Return:
            Dict[str, Any]: LLM响应
            
        Exception:
            NotImplementedError: 必须由子类实现
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def post_process(self, doc: Dict[str, Any], llm_out: str = "", merge: bool = True) -> Dict[str, Any]:
        """
        在LLM调用后对文档进行后处理（抽象方法，需由子类实现）。
        
        Args:
            doc (Dict[str, Any]): 要后处理的文档
            llm_out (str): LLM输出字符串
            merge (bool): 是否将数据合入储存中，如nodelist和edgelist
            
        Return:
            Dict[str, Any]: 后处理后的文档
            
        Exception:
            NotImplementedError: 必须由子类实现
        """
        raise NotImplementedError("子类必须实现此方法")
    
    async def aprocess(self, data: Dict[str, Any], *args, **kwargs) -> List[Dict[str, Any]]:
        """
        异步处理数据，通过LLM调用和后处理。
        
        Args:
            data (Dict[str, Any]): 要处理的数据
            *args: 其他位置参数
            **kwargs: 其他关键字参数
            
        Return:
            List[Dict[str, Any]]: 处理结果列表
        """
        llm_out = await self.call_llm(data)
        return self.post_process(chunk, llm_out)

    def _extract_json_between_tags(self, text: str):
        """
        提取预定义标签之间的JSON内容。
        
        Args:
            text (str): 包含标签间JSON的文本
            
        Return:
            str: 提取的JSON内容，如果未找到标签则返回空字符串
        """
        start_tag = self.json_tag_start
        end_tag = self.json_tag_end
        start_index = text.find(start_tag)
        if start_index == -1:
            return ""
        end_index = text.find(end_tag, start_index + len(start_tag))
        return text[start_index + len(start_tag):end_index]

# {id, name, content, desc, attr}
class DocumentProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        """
        使用特定节点类型初始化DocumentProcessor。
        """
        super().__init__(*args, **kwargs) 
        self.type = NodeType.Document

    """文档名称与基础信息提取"""
    def add_mdhash_id(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据文档的名称/标题/路径添加唯一ID。
        
        Args:
            doc (Dict[str, Any]): 文档字典
            
        Return:
            Dict[str, Any]: 添加了ID的文档
        """
        name = doc.get("name") or doc.get("title") or doc.get("path")
        node_id = compute_mdhash_id(str(name or doc), prefix=PREFIX_MAP[self.type])
        doc["id"] = node_id
        return doc
    def _process(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        通过添加ID、类型并存储到节点数据中来处理文档。
        
        Args:
            doc (Dict[str, Any]): 要处理的文档
            
        Return:
            Dict[str, Any]: 包含ID和类型的处理结果
        """
        name = doc.get("name") or doc.get("title") or doc.get("path")
        node_id = compute_mdhash_id(str(name or doc), prefix=PREFIX_MAP[self.type])
        doc["id"] = node_id
        doc["type"] = self.type
        self.nodedata.info_doc_list.append(doc)
        return {"id": node_id, "type": self.type}
    async def call_llm(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用LLM进行文档处理（占位符实现）。
        
        Args:
            doc (Dict[str, Any]): 要处理的文档
            
        Return:
            Dict[str, Any]: 包含LLM输出的文档
        """
        res = ""
        if not doc.get("llm_out"):  # llm_out为空
            doc["llm_out"] = []
        doc["llm_out"].append(res)
        doc["type"] = self.type
        doc = self.check_llm_out(doc)
        return doc
    
    def post_process(self, doc: Dict[str, Any], llm_out: str = "", merge: bool = True) -> Dict[str, Any]:
        """
        通过添加描述和属性来后处理文档。
        
        Args:
            doc (Dict[str, Any]): 要后处理的文档
            llm_out (str): LLM输出（在当前实现中未使用）
            
        Return:
            Dict[str, Any]: 后处理后的文档
        """
        doc = doc.copy()
        doc["desc"] = ""
        doc["attr"] = {}
        return self._process(doc)
    
    def check_llm_out(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查文档的LLM输出（占位符实现）。
        
        Args:
            node (Dict[str, Any]): 可能包含LLM输出的节点
            
        Return:
            Dict[str, Any]: 包含错误状态的节点
        """
        # 尚未实现
        node["llm_out_error"] = None
        return node

class ChunkProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        """
        使用特定节点类型初始化ChunkProcessor。
        """
        super().__init__(*args, **kwargs) 
        self.type = NodeType.Chunk
    
    def add_mdhash_id(self, chunk: Dict[str, Any]) -> str:
        """
        根据块的内容添加唯一ID。
        
        Args:
            chunk (Dict[str, Any]): 块字典
            
        Return:
            str: 添加了ID的块
        """
        content = chunk["content"]
        node_id = compute_mdhash_id(content, prefix=PREFIX_MAP[self.type])
        chunk["id"] = node_id
        return chunk

    def _process(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        通过添加ID、类型并存储到节点数据中来处理块。
        
        Args:
            chunk (Dict[str, Any]): 要处理的块
            
        Return:
            List[Dict[str, Any]]: 包含ID和类型的处理结果
        """
        content = chunk["content"]
        node_id = compute_mdhash_id(content, prefix=PREFIX_MAP[self.type]) 
        
        chunk["id"] = node_id
        chunk["type"] = self.type
        self.nodedata.info_chunk_list.append(chunk)
        return {"id": node_id, "type": self.type}
    
    async def call_llm(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        调用LLM进行块处理（占位符实现）。
        
        Args:
            chunk (Dict[str, Any]): 要处理的块
            
        Return:
            List[Dict[str, Any]]: 包含LLM输出的块
        """
        res = ""
        if not chunk.get("llm_out"):  # llm_out为空
            chunk["llm_out"] = []
        chunk["llm_out"].append(res)
        chunk["type"] = self.type
        chunk = self.check_llm_out(chunk)
        return chunk
    def post_process(self, chunk: Dict[str, Any], llm_out: str = "", merge: bool = True) -> List[Dict[str, Any]]:
        """
        通过添加名称、描述和属性来后处理块。
        
        Args:
            chunk (Dict[str, Any]): 要后处理的块
            llm_out (str): LLM输出（在当前实现中未使用）
            
        Return:
            List[Dict[str, Any]]: 后处理后的块
        """
        chunk = chunk.copy()
        # llm_out = chunk.get("llm_out",[""])[-1]
        chunk["name"] = chunk["content"]
        chunk["desc"] = ""
        chunk["attr"] = {}
        return self._process(chunk)
    
    def check_llm_out(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查块的LLM输出（占位符实现）。
        
        Args:
            node (Dict[str, Any]): 可能包含LLM输出的节点
            
        Return:
            Dict[str, Any]: 包含错误状态的节点
        """
        # 尚未实现
        node["llm_out_error"] = None
        return node

class EntityProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        """
        使用特定节点类型初始化EntityProcessor。
        """
        super().__init__(*args, **kwargs) 
        self.type = NodeType.Entity

    def add_mdhash_id(self, chunk: Dict[str, Any]) -> str:
        """
        根据实体的内容添加唯一ID。
        
        Args:
            chunk (Dict[str, Any]): 实体字典
            
        Return:
            str: 添加了ID的实体
        """
        content = chunk["content"]
        node_id = compute_mdhash_id(content, prefix=PREFIX_MAP[self.type])
        chunk["id"] = node_id
        return chunk

    async def call_llm(self, chunk: Dict[str, Any], entity_types: Optional[List[str]] = None, language: str = "the language of the input content(after Text)") -> List[Dict[str, Any]]:
        """
        调用LLM从块内容中提取实体。
        
        Args:
            chunk (Dict[str, Any]): 包含要提取实体内容的块
            entity_types (Optional[List[str]]): 要提取的实体类型列表
            language (str): 输入内容的语言
            
        Return:
            List[Dict[str, Any]]: 包含提取实体的LLM输出的块
        """
        tuple_delim = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        compl_delim = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        examples = "".join(PROMPTS["entity_extraction_examples"]).format(tuple_delimiter=tuple_delim, completion_delimiter=compl_delim)
        entity_types = chunk.get("Ontology", ["Person","Organization","Location","Event","Product","Other"])
        system= PROMPTS["entity_extraction_system_prompt"].format(
            entity_types=", ".join(entity_types), tuple_delimiter=tuple_delim, language=language, 
            completion_delimiter=compl_delim, examples=examples, input_text=chunk["content"], supplementary_information=chunk.get("context", "无")
        )
        user = PROMPTS["entity_extraction_user_prompt"].format(
            completion_delimiter=compl_delim, language=language
        )
        res = await self.llm_model_func(prompt=user, system_prompt=system)
        if not chunk.get("llm_out"):  # llm_out为空
            chunk["llm_out"] = []
        chunk["llm_out"].append(res)
        chunk = self.check_llm_out(chunk)
        return chunk
    
    def compute_entity_id_by_name(self, name: str) -> str:
        """
        根据实体名称计算实体ID。
        
        Args:
            name (str): 实体名称
            
        Return:
            str: 计算出的实体ID
        """
        return compute_mdhash_id(name, prefix=PREFIX_MAP[self.type])
    def post_process(self, chunk: Dict[str, Any], llm_out: str = "", merge: bool = True) -> List[Dict[str, Any]]:
        """
        后处理块，从LLM输出中提取和格式化实体。
        
        Args:
            chunk (Dict[str, Any]): 包含LLM输出的块
            llm_out (str): LLM输出字符串（未使用，从块中获取）
            
        Return:
            List[Dict[str, Any]]: 提取的实体列表
        """
        chunk = chunk.copy()
        tuple_delim = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        compl_delim = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        llm_out = chunk.get("llm_out",[""])[-1]
        lines = [l.strip() for l in llm_out.splitlines() if l.strip() and l.strip() != compl_delim]
        entities: List[Dict[str, Any]] = []
        from_modal = chunk.get("from_modal", False)

        for ln in lines:
            parts = ln.split(tuple_delim)
            if len(parts) == 4 and parts[0] == "entity":
                name = parts[1]
                node_id = compute_mdhash_id(name, prefix=PREFIX_MAP[self.type]) 
                entities.append({"id": node_id, "src_id": chunk["id"], "type": self.type, "name": name, "desc": parts[3], "attr": {"entity_type": parts[2], "from_modal": from_modal}})
        if merge:
            self.nodedata.info_entity_list.extend(entities)
        return entities

    def check_llm_out(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查实体提取的LLM输出（占位符实现）。
        
        Args:
            node (Dict[str, Any]): 可能包含LLM输出的节点
            
        Return:
            Dict[str, Any]: 包含错误状态的节点
        """
        # 尚未实现
        node["llm_out_error"] = None
        return node


    def _group_assertions_by_chunk(
        self, 
        assertion_list: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        按块ID对断言进行分组。
        
        Args:
            assertion_list (List[Dict[str, Any]]): 要分组的断言列表
            
        Return:
            Dict[str, List[Dict[str, Any]]]: 按块ID分组的断言
        """
        chunk_assertions = {}
        for assertion in assertion_list:
            chunk_id = assertion.get("src_id")
            if chunk_id:
                if chunk_id not in chunk_assertions:
                    chunk_assertions[chunk_id] = []
                chunk_assertions[chunk_id].append(assertion)
        return chunk_assertions

    async def recall_entity(self, entity_types: Optional[List[str]] = None, language: str = "the language of the input content(after Text)") -> List[Dict[str, Any]]:
        """
        根据块中的断言召回实体。
        
        Args:
            entity_types (Optional[List[str]]): 要召回的实体类型列表
            language (str): 输入内容的语言
            
        Return:
            List[Dict[str, Any]]: 召回的实体列表
        """
        chunk2assertions = self._group_assertions_by_chunk(self.nodedata.info_assertion_list)
        id2chunk = self.nodedata.list2id_dict(self.nodedata.flattened_node_list())
        task_list = []
        
        for chunk_id, assertions in chunk2assertions.items():
            chunk = id2chunk[chunk_id].copy()
            task = asyncio.create_task(
                self._recall_entity_single_chunk(chunk=chunk, assertions=assertions, entity_types=entity_types, language=language)
            )
            task_list.append(task)
        
        nodelist = await tqdm_asyncio.gather(*task_list, desc="entity recall")
        recall_entity_cnt = 0
        for node in nodelist:
            recall_entity_cnt+=len(self.post_process(node))
        
        logger.info(f"entity recall: {recall_entity_cnt}")


    async def _recall_entity_single_chunk(self, chunk: Dict[str, Any], assertions: List[Dict[str, Any]], entity_types: Optional[List[str]] = None, language: str = "the language of the input content(after Text)"):
        """
        根据断言召回单个块的实体。
        
        Args:
            chunk (Dict[str, Any]): 要处理的块
            assertions (List[Dict[str, Any]]): 与此块相关的断言
            entity_types (Optional[List[str]]): 要召回的实体类型列表
            language (str): 输入内容的语言
            
        Return:
            Dict[str, Any]: 包含LLM输出的块，其中包含召回的实体
        """
        tuple_delim = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        name_list = [e["name"] for e in self.nodedata.info_entity_list]
        tasks = ""
        for assertion in assertions:
            entity_list = []
            if assertion["head"] not in name_list:
                entity_list.append(assertion["head"])
            if assertion["tail"] not in name_list:
                entity_list.append(assertion["tail"])
            if len(entity_list) == 0: 
                continue
            tasks += PROMPTS["entity_recall_task_prompt"].format(
                tuple_delimiter=tuple_delim,
                assertion=assertion["desc"],
                entity_list=",".join(entity_list),
            )

        input_text = chunk["content"]
        entity_types = entity_types or ["Person","Organization","Location","Event","Product","Other"]

        # Prompt
        tuple_delim = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        compl_delim = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        entity_types = entity_types or ["Person","Organization","Location","Event","Product","Other"]
        system = PROMPTS["entity_recall_sys_prompt"].format(
            entity_types=", ".join(entity_types), tuple_delimiter=tuple_delim, language=language, 
            completion_delimiter=compl_delim, input_text=chunk["content"],tasklist=tasks
        )
        
        user = PROMPTS["entity_extraction_user_prompt"].format(
            completion_delimiter=compl_delim, language=language
        )
        res = await self.llm_model_func(prompt=user, system_prompt=system)
        if not chunk.get("llm_out"):
            chunk["llm_out"] = []
        chunk["llm_out"].append(res)
        chunk = self.check_llm_out(chunk)
        return chunk

class AssertionProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        """
        使用特定节点类型初始化AssertionProcessor。
        """
        super().__init__(*args, **kwargs)
        self.type = NodeType.Assertion

    def add_mdhash_id(self, chunk: Dict[str, Any]) -> str:
        """
        根据断言的内容添加唯一ID。
        
        Args:
            chunk (Dict[str, Any]): 断言字典
            
        Return:
            str: 添加了ID的断言
        """
        content = chunk["content"]
        node_id = compute_mdhash_id(content, prefix=PREFIX_MAP[self.type])
        chunk["id"] = node_id
        return chunk

    """抽取 <实体-关系-实体> 的断言所在 chunk"""
    async def call_llm(self, chunk: Dict[str, Any], entity_types: Optional[List[str]] = None, language: str = "the language of the input content(after Text)") -> List[Dict[str, Any]]:
        """
        调用LLM从块内容中提取断言（实体-关系-实体三元组）。
        
        Args:
            chunk (Dict[str, Any]): 包含要提取断言内容的块
            entity_types (Optional[List[str]]): 要考虑的实体类型列表
            language (str): 输入内容的语言
            
        Return:
            List[Dict[str, Any]]: 包含提取断言的LLM输出的块
        """
        tuple_delim = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        compl_delim = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        examples = "".join(PROMPTS["entity_extraction_examples"]).format(tuple_delimiter=tuple_delim, completion_delimiter=compl_delim)
        entity_types = entity_types or ["Person","Organization","Location","Event","Product","Other"]
        system= PROMPTS["entity_extraction_system_prompt"].format(
            entity_types=", ".join(entity_types), tuple_delimiter=tuple_delim, language=language, 
            completion_delimiter=compl_delim, examples=examples, input_text=chunk["content"], supplementary_information=chunk.get("context", "无")
        )
        user = PROMPTS["entity_extraction_user_prompt"].format(
            completion_delimiter=compl_delim, language=language
        )
        res = await self.llm_model_func(prompt=user, system_prompt=system)
        if not chunk.get("llm_out"):
            chunk["llm_out"] = []
        chunk["llm_out"].append(res)
        chunk = self.check_llm_out(chunk)
        # chunk["type"] = self.type
        return chunk

    def post_process(self, chunk: Dict[str, Any], llm_out: str = "", merge: bool = True) -> Dict[str, Any]:
        """
        后处理块，从LLM输出中提取和格式化断言。
        
        Args:
            chunk (Dict[str, Any]): 包含LLM输出的块
            llm_out (str): LLM输出字符串（未使用，从块中获取）
            
        Return:
            Dict[str, Any]: 提取的断言列表
        """
        chunk = chunk.copy()
        tuple_delim = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        compl_delim = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        llm_out = chunk.get("llm_out",[""])[-1]
        lines = [l.strip() for l in llm_out.splitlines() if l.strip() and l.strip() != compl_delim]
        relations: List[Dict[str, Any]] = []
        from_modal = chunk.get("from_modal", False)
        for ln in lines:
            parts = ln.split(tuple_delim)
            if len(parts) == 5 and parts[0] == "relation":
                node_id = compute_mdhash_id(parts[1] + "|" + parts[2] + "|" + parts[3], prefix=PREFIX_MAP[self.type]) 
                relations.append({"id": node_id, "src_id": chunk["id"], "type": self.type, "head": parts[1], "tail": parts[2], "relation": parts[3], "name": parts[4],"desc": parts[4], "attr": {}, "from_modal": from_modal})
        if merge:
            self.nodedata.info_assertion_list.extend(relations)
        return relations
    def check_llm_out(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查断言提取的LLM输出（占位符实现）。
        
        Args:
            node (Dict[str, Any]): 可能包含LLM输出的节点
            
        Return:
            Dict[str, Any]: 包含错误状态的节点
        """
        # 尚未实现
        node["llm_out_error"] = None
        return node

    def _group_entitys_by_chunk(
        self, 
        entity_list: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        按块ID对实体进行分组。
        
        Args:
            entity_list (List[Dict[str, Any]]): 要分组的实体列表
            
        Return:
            Dict[str, List[Dict[str, Any]]]: 按块ID分组的实体
        """
        """按chunk_id对断言进行分组"""
        chunk_entities = {}
        for entity in entity_list:
            chunk_id = entity.get("src_id")
            if chunk_id:
                if chunk_id not in chunk_entities:
                    chunk_entities[chunk_id] = []
                chunk_entities[chunk_id].append(entity)
        return chunk_entities

    def _group_assertions_by_chunk(
        self, 
        assertion_list: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        按块ID对断言进行分组。
        
        Args:
            assertion_list (List[Dict[str, Any]]): 要分组的断言列表
            
        Return:
            Dict[str, List[Dict[str, Any]]]: 按块ID分组的断言
        """
        chunk_assertions = {}
        for assertion in assertion_list:
            chunk_id = assertion.get("src_id")
            if chunk_id:
                if chunk_id not in chunk_assertions:
                    chunk_assertions[chunk_id] = []
                chunk_assertions[chunk_id].append(assertion)
        return chunk_assertions

    async def recall_assertion(self, entity_types: Optional[List[str]] = None, language: str = "the language of the input content(after Text)") -> List[Dict[str, Any]]:
        """
        根据块中的实体召回断言。
        
        Args:
            entity_types (Optional[List[str]]): 要考虑的实体类型列表
            language (str): 输入内容的语言
            
        Return:
            List[Dict[str, Any]]: 召回的断言列表
        """
        # TODO: 添加剪枝条件
        
        chunk2entities = self._group_entitys_by_chunk(self.nodedata.info_entity_list)
        chunk2assertions = self._group_assertions_by_chunk(self.nodedata.info_assertion_list)
        id2chunk = self.nodedata.list2id_dict(self.nodedata.flattened_node_list())
        task_list = []
        
        for chunk_id, entities in chunk2entities.items():
            chunk = id2chunk[chunk_id].copy()
            ex_relations = chunk2assertions.get(chunk_id, [])
            task = asyncio.create_task(
                self._recall_assertion_single_chunk(chunk=chunk, entities=entities, ex_relations=ex_relations, entity_types=entity_types, language=language)
            )
            task_list.append(task)
        
        nodelist = await tqdm_asyncio.gather(*task_list, desc="assertion recall")
        recall_rels_cnt = 0
        for node in nodelist:
            recall_rels_cnt+=len(self.post_process(node))
        
        logger.info(f"recall relations: {recall_rels_cnt}")


    async def _recall_assertion_single_chunk(self, chunk: Dict[str, Any], entities: List[Dict[str, Any]],ex_relations: List[Dict[str, Any]], entity_types: Optional[List[str]] = None, language: str = "the language of the input content(after Text)"):
        """
        根据实体召回单个块的断言。
        
        Args:
            chunk (Dict[str, Any]): 要处理的块
            entities (List[Dict[str, Any]]): 此块中的实体
            entity_types (Optional[List[str]]): 要考虑的实体类型列表
            language (str): 输入内容的语言
            
        Return:
            Dict[str, Any]: 包含LLM输出的块，其中包含召回的断言
        """
        tuple_delim = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        task_template = PROMPTS["assertion_recall_task_prompt"]
        name_list = [task_template.format(tuple_delimiter=tuple_delim,entity=e["name"],entity_desc=e['desc']) for e in entities]
        tasklist = "\n".join(name_list)

        # 排除的关系
        ex_rel_template = PROMPTS["assertion_exclude_prompt"]
        ex_rel_list = [ex_rel_template.format(tuple_delimiter=tuple_delim, source_entity=r["head"], target_entity=r['tail'], relation=r["relation"], relation_desc=r["desc"]) for r in ex_relations]
        ex_rel_list = "\n".join(ex_rel_list)
        
        input_text = chunk["content"]
        entity_types = entity_types or ["Person","Organization","Location","Event","Product","Other"]

        # Prompts
        tuple_delim = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        compl_delim = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        entity_types = entity_types or ["Person","Organization","Location","Event","Product","Other"]
        system = PROMPTS["assertion_recall_sys_prompt"].format(
            entity_types=", ".join(entity_types), tuple_delimiter=tuple_delim, language=language, 
            completion_delimiter=compl_delim, input_text=chunk["content"],tasklist=tasklist,ex_rel_list=ex_rel_list
        )
        
        user = PROMPTS["assertion_extraction_user_prompt"].format(
            completion_delimiter=compl_delim, language=language
        )
        res = await self.llm_model_func(prompt=user, system_prompt=system)
        if not chunk.get("llm_out"):
            chunk["llm_out"] = []
        chunk["llm_out"].append(res)
        chunk = self.check_llm_out(chunk)
        return chunk

class TableProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        """
        使用特定节点类型和必需字段初始化TableProcessor。
        """
        super().__init__(*args, **kwargs)
        self.type = NodeType.Table
        self.need_llm_out_field = ["name", "caption", "desc"]
    
    def add_mdhash_id(self, table: Dict[str, Any]) -> str:
        """
        根据表格的内容添加唯一ID。
        
        Args:
            table (Dict[str, Any]): 表格字典
            
        Return:
            str: 添加了ID的表格
        """
        content = table["content"]
        node_id = compute_mdhash_id(content, prefix=PREFIX_MAP[self.type])
        table["id"] = node_id
        return table

    """为每个表格生成 name，并规整结构（含 html 与可选 caption）"""
    async def call_llm(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用LLM分析表格并生成元数据（名称、标题、描述）。
        
        Args:
            table (Dict[str, Any]): 要分析的表格
            
        Return:
            Dict[str, Any]: 包含分析结果的LLM输出的表格
        """
        node_id = compute_mdhash_id(table["content"], prefix=PREFIX_MAP[self.type])
        system = PROMPTS["TABLE_ANALYSIS_SYSTEM"]

        # 补充的context
        context = "### 该表格的相关信息补充\n"
        context = context + table.get("context", "无")
        table_info = "## 该表格的数据\n"
        table_info = table_info + table["content"]
        add_info = "\n".join([context, table_info])
        user = PROMPTS["TABLE_INFO_GENE"].format(json_tag_start=self.json_tag_start, json_tag_end=self.json_tag_end) + add_info
        res = await self.llm_model_func(prompt=user, system_prompt=system)
        if not table.get("llm_out"):
            table["llm_out"] = []
        table["llm_out"].append(res)
        table["type"] = self.type
        table = self.check_llm_out(table)
        return table

    def post_process(self, table: Dict[str, Any], llm_out: str = "", merge: bool = True) -> Dict[str, Any]:
        """
        后处理表格，提取并应用LLM生成的元数据。
        
        Args:
            table (Dict[str, Any]): 包含LLM输出的表格
            llm_out (str): LLM输出字符串（未使用，从表格中获取）
            
        Return:
            Dict[str, Any]: 包含ID和类型的处理结果
        """
        table = table.copy()
        llm_out = table.get("llm_out",[""])[-1]
        if isinstance(llm_out, str):
            data = json.loads(llm_out) #{name, caption, desc}
        else:
            data = llm_out
        if table.get("caption") is None or len(table.get("caption")) == 0:
            table["caption"] = data["caption"]

        # if table.get("name") is None:
        if isinstance(table.get("caption"), str):
            table["name"] = table["caption"]
        elif isinstance(table.get("caption"), list):
            table["name"] = table["caption"][0]
        table["desc"] = data["desc"]
        if merge:
            self.nodedata.info_table_list.append(table)
        return {"id": table["id"], "type": self.type}
    
    # 检查 llm_out中字段是否存在
    def check_llm_out(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查LLM输出是否包含表格所需的所有字段。
        
        Args:
            node (Dict[str, Any]): 可能包含LLM输出的节点
            
        Return:
            Dict[str, Any]: 包含验证后的LLM输出或错误状态的节点
        """
        llm_out = node.get("llm_out",[""])[-1]
        node["llm_out_error"] = llm_out
        llm_out = self._extract_json_between_tags(llm_out)
        if llm_out is None or len(llm_out) == 0: 
            node["llm_out"] = None
            return node
        llm_dict = self._parse_json(llm_out)
        for field in self.need_llm_out_field:
            if field not in llm_dict or len(llm_dict[field]) == 0:
                node["llm_out"] = None
                return node
        if not node.get("llm_out"):
            node["llm_out"] = []
        node["llm_out"].append(llm_dict)
        node["llm_out_error"] = None
        return node

class FormulaProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        """
        使用特定节点类型和必需字段初始化FormulaProcessor。
        """
        super().__init__(*args, **kwargs)
        self.type = NodeType.Formula
        self.need_llm_out_field = ["name", "caption", "desc"]

    def add_mdhash_id(self, formula: Dict[str, Any]) -> str:
        """
        根据公式的内容添加唯一ID。
        
        Args:
            formula (Dict[str, Any]): 公式字典
            
        Return:
            str: 添加了ID的公式
        """
        content = formula["content"]
        node_id = compute_mdhash_id(content, prefix=PREFIX_MAP[self.type])
        formula["id"] = node_id
        return formula
    """生成公式的 caption 与 name（说明公式讲了什么、输入输出与参数）"""
    async def call_llm(self, formula: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用LLM分析公式并生成元数据（名称、标题、描述）。
        
        Args:
            formula (Dict[str, Any]): 要分析的公式
            
        Return:
            Dict[str, Any]: 包含分析结果的LLM输出的公式
        """
        node_id = compute_mdhash_id(formula["content"], prefix=PREFIX_MAP[self.type])
        system = PROMPTS["FORMULA_ANALYSIS_SYSTEM"]
        context = formula.get("context", "")
        user = PROMPTS["FORMULA_ANALYSIS_USER"].format(formula=formula["content"], json_tag_start=self.json_tag_start, json_tag_end=self.json_tag_end, context=context)
        res = await self.llm_model_func(prompt=user, system_prompt=system)
        if not formula.get("llm_out"):
            formula["llm_out"] = []
        formula["llm_out"].append(res)
        formula["type"] = self.type
        formula = self.check_llm_out(formula)
        return formula
    def post_process(self, formula: Dict[str, Any], llm_out: str = "", merge: bool = True) -> Dict[str, Any]:
        """
        后处理公式，提取并应用LLM生成的元数据。
        
        Args:
            formula (Dict[str, Any]): 包含LLM输出的公式
            llm_out (str): LLM输出字符串（未使用，从公式中获取）
            
        Return:
            Dict[str, Any]: 包含更新后公式数据的处理结果
        """
        formula = formula.copy()
        llm_out = formula.get("llm_out",[""])[-1]
        if isinstance(llm_out, str):
            data = json.loads(llm_out) #{name, caption, desc}
        else:
            data = llm_out
        
        formula.update(data)
        # formula["id"] = node_id
        if merge:
            self.nodedata.info_formula_list.append(formula)
        return data
    
    # 检查 llm_out中字段是否存在
    def check_llm_out(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查LLM输出是否包含公式所需的所有字段。
        
        Args:
            node (Dict[str, Any]): 可能包含LLM输出的节点
            
        Return:
            Dict[str, Any]: 包含验证后的LLM输出或错误状态的节点
        """
        llm_out = node.get("llm_out",[""])[-1]
        node["llm_out_error"] = llm_out
        llm_out = self._extract_json_between_tags(llm_out)
        if llm_out is None or len(llm_out) == 0: 
            node["llm_out"] = None
            return node
        llm_dict = self._parse_json(llm_out)
        for field in self.need_llm_out_field:
            if field not in llm_dict or len(llm_dict[field]) == 0:
                node["llm_out"] = None
                return node
        if not node.get("llm_out"):
            node["llm_out"] = []
        node["llm_out"].append(llm_dict)
        node["llm_out_error"] = None
        return node

class ImageProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        """
        使用特定节点类型和必需字段初始化ImageProcessor。
        """
        super().__init__(*args, **kwargs)
        self.type = NodeType.Image
        self.need_llm_out_field = ["name", "class", "caption", "content", "desc"] # {name, class, caption, content, desc}

    def add_mdhash_id(self, image: Dict[str, Any]) -> str:
        """
        根据图像的路径添加唯一ID。
        
        Args:
            image (Dict[str, Any]): 图像字典
            
        Return:
            str: 添加了ID的图像
        """
        content = image["img_path"]
        node_id = compute_mdhash_id(content, prefix=PREFIX_MAP[self.type])
        image["id"] = node_id
        return image

    """图像处理：命名配对、分类（流程图/数值图/思维导图/其他），并生成对应表示"""
    async def call_llm(self, image: Dict[str, Any], surrounding_text: Optional[str] = None, model: str = "qwen-vl3-2b") -> Dict[str, Any]:
        """
        调用视觉LLM分析图像并生成元数据。
        
        Args:
            image (Dict[str, Any]): 要分析的图像
            surrounding_text (Optional[str]): 图像周围的文本
            model (str): 要使用的视觉模型
            
        Return:
            Dict[str, Any]: 包含分析结果的LLM输出的图像
        """
        node_id = compute_mdhash_id(image["img_path"], prefix=PREFIX_MAP[self.type])
        system = PROMPTS["IMAGE_ANALYSIS_SYSTEM"]
        # 补充的context
        context = "\n## 该图片在原文中的相关信息段落\n"
        context = context + image.get("context", "无")

        user = PROMPTS["IMAGE_ANALYSIS_USER"].format(json_tag_start=self.json_tag_start, json_tag_end=self.json_tag_end) + context
        res = await self.vision_model_func(image_source=image["img_path"], prompt=user, system_prompt=system)
        if not image.get("llm_out"):  # 为空
            image["llm_out"] = []
        image["llm_out"].append(res)
        image["type"] = self.type
        image = self.check_llm_out(image)
        return image


    def post_process(self, image: Dict[str, Any], llm_out: str = "", merge: bool = True) -> Dict[str, Any]:
        """
        后处理图像，提取并应用LLM生成的元数据。
        
        Args:
            image (Dict[str, Any]): 包含LLM输出的图像
            llm_out (str): LLM输出字符串（未使用，从图像中获取）
            
        Return:
            Dict[str, Any]: 包含更新后图像数据的处理结果
        """
        image = image.copy()
        # print("image:", str(image))
        llm_out = image.get("llm_out",[""])[-1]
        data = {}
        try:
            if isinstance(llm_out, str):
                data = json.loads(llm_out)  #{name, class, caption, content, desc}
            else:
                data = llm_out
        except Exception as e:
            logger.error(e)
            logger.warning(f"LLM响应中未找到类JSON结构.{llm_out}")
        if image.get("caption") is not None and len(image.get("caption")) > 0:
            data.pop("caption")
        image.update(data)
        
        # image["id"] = node_id
        if merge:
            self.nodedata.info_image_list.append(image)
        return data

    # 检查 llm_out中字段是否存在
    def check_llm_out(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查LLM输出是否包含图像所需的所有字段。
        
        Args:
            node (Dict[str, Any]): 可能包含LLM输出的节点
            
        Return:
            Dict[str, Any]: 包含验证后的LLM输出或错误状态的节点
        """
        llm_out = node.get("llm_out",[""])[-1]
        node["llm_out_error"] = llm_out
        llm_out = self._extract_json_between_tags(llm_out)
        if llm_out is None or len(llm_out) == 0: 
            node["llm_out"] = None
            return node
        llm_dict = self._parse_json(llm_out)
        for field in self.need_llm_out_field:
            if field not in llm_dict or len(llm_dict[field]) == 0:
                node["llm_out"] = None
                return node
        if not node.get("llm_out"):
            node["llm_out"] = []
        node["llm_out"].append(llm_dict)
        node["llm_out_error"] = None
        return node