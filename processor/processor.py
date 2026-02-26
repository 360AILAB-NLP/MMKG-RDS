"""
处理器模块，处理文档调用大语言模型处理提取各类节点数据和关系数据
"""

import os
from typing import Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from omegaconf import DictConfig
import json
from tqdm import tqdm
import logging
import asyncio
from tqdm.asyncio import tqdm_asyncio
import networkx as nx
import util.json2graph as jg


from .node import NodeData, node_type_list
from .edge import EdgeData
from .modal import *

logger = logging.getLogger(__name__)

class process_stage:
    # first stage 调用大模型抽取实体关系和相关字段信息
    stage_1 = 1
    # second stage 调用大模型从图表公式等以及其相关上下文中提取实体和关系
    stage_2 = 2
    # third stage 大模型根据关系召回实体和根据实体召回关系
    stage_3 = 3
    # fourth stage 模型根据关系和实体进行实体关系合并以及实体关系标准化
    stage_4 = 4

def get_stage(file_path: str):
    """
    从文件路径中提取最后匹配到的 'stage_{}.json' 模式中的 '{}' 字段。
    
    参数:
        file_path (str): 完整的文件路径或单纯的文件名
    
    返回:
        str 或 None: 成功则返回最后匹配到的字段字符串，否则返回 None
    """
    # 从路径中提取纯文件名
    filename = os.path.basename(file_path)
    
    pattern = r'stage_(.*?)\.json'
    matches = re.findall(pattern, filename)
    
    # 返回最后一个匹配到的字段（如果存在）
    return int(matches[-1]) if matches else -1



class Processor:
    """
    处理器主类，负责协调各种类型的处理器处理不同节点类型的数据
    """
    def __init__(self, cfg: DictConfig, llm_func: Callable, vlm_func: Callable):
        """
        初始化处理器
        
        Args:
            cfg (DictConfig): 配置信息
            llm_func (Callable): 大语言模型调用函数
            vlm_func (Callable): 视觉语言模型调用函数
        """
        self.cfg = cfg
        self.nodedata = NodeData(cfg)
        self.edgedata = EdgeData()
        self.processor = {
            "Document": DocumentProcessor(llm_func, vlm_func, self.nodedata),
            "Chunk": ChunkProcessor(llm_func, vlm_func, self.nodedata),
            "Assertion": AssertionProcessor(llm_func, vlm_func, self.nodedata),
            "Entity": EntityProcessor(llm_func, vlm_func, self.nodedata),
            "Table": TableProcessor(llm_func, vlm_func, self.nodedata),
            "Image": ImageProcessor(llm_func, vlm_func, self.nodedata),
            "Formula": FormulaProcessor(llm_func, vlm_func, self.nodedata)
        }

        if not os.path.exists(self.cfg.data.output_dir):
            os.makedirs(self.cfg.data.output_dir)
        self.output_dir = self.cfg.data.output_dir
        self.enable_entity_assertion_share_llm_call = True
        self.schema = json.load(open(self.cfg.dataprocessing.schema, "r", encoding='utf-8'))
        self.Ontology = self.schema["Ontology"].keys()
    

    def _merge_chunk_content(self, chunkA, chunkB):
        """
        合并两个chunk的内容
        
        Args:
            chunkA (dict): 第一个chunk信息
            chunkB (dict): 第二个chunk信息
            
        Returns:
            dict: 合并后的chunk信息
        """
        return {
        "type": chunkA["type"],
        "text": chunkA["text"] + chunkB["text"],
        "page_idx": (chunkA["page_idx"]+chunkB["page_idx"])/2.0,        
        }
    
    # 处理文档解析的一些问题
    # 跨页，合并跨页chunk
    # 表格图片公式，编号
    def pre_process(self):
        """
        预处理文档解析结果，处理跨页问题和合并跨页chunk等
        """
        # output_dir = self.cfg.data.parse_dir
        # dir_list = os.listdir(output_dir)
        # content_list = []
        # for doc_name in dir_list:
        #     doc_path = os.path.join(parsedfile_dir, doc_name, "vlm")
        #     doc_content_list = []
        #     with open(os.path.join(doc_path, f"{doc_name}_content_list.json"), "r", encoding='utf-8') as f:
        #         content_list = json.load(f)
        #     last_chunk_idx = 0
        #     for old_idx, content in enumerate(content_list):
        #         if content["type"] == "text":
        #             last_chunk_idx
        #             doc_content_list.append(content)
        #             if last_incomplete_chunk_idx != -1:
        #                 pass

        #         # elif content["type"] == "table":
        #         #     doc_content_list.append(content)
        #         # elif content["type"] == "image":
        #         #     doc_content_list.append(content)
        #         elif content["type"] == "formula":
        #             doc_content_list.append(content)
        #         else:
        #             # todo
        #             pass

    async def post_process(self, node_llm_out_list_or_path: str | Dict[str, Any], enable_modal2entity: bool = True):
        """
        对大模型输出结果进行后处理
        
        Args:
            node_llm_out_list_or_path (str | Dict[str, Any]): 节点大模型输出结果列表或文件路径
            enable_modal2entity (bool): 是否启用模态到实体的处理，默认为True
        """
        if isinstance(node_llm_out_list_or_path, str):
            with open(node_llm_out_list_or_path, "r", encoding='utf-8') as f:
                nodelist = json.load(f)
        else:
            nodelist = node_llm_out_list_or_path
        
        # 大模型调用异常，删除该元素
        new_node_list = []
        err_node_list = []
        for node in nodelist:
            if "llm_out" in node and node["llm_out"] is not None:
                new_node_list.append(node)
            else:
                err_node_list.append(node)
        logger.info(f"{len(err_node_list)}/{len(nodelist)} nodes are deleted due to llm call error")
        with open("llm_err_node_list.json", "w", encoding='utf-8') as f:
            json.dump(err_node_list, f, ensure_ascii=False)
        nodelist = new_node_list

        for node in tqdm(nodelist, desc="Processing"):
            self.processor[node["type"]].post_process(node)
            if (node["type"] == NodeType.Assertion and 
                self.enable_entity_assertion_share_llm_call):
                # 共享llm_out
                self.processor[NodeType.Entity].post_process(node)
        
        if enable_modal2entity:
            modal_list = []
            for nodetype in [NodeType.Table, NodeType.Image, NodeType.Formula]:
                modal_list.extend(self.nodedata.get_nodelist(nodetype))
            await self.modal2entity_process(modal_list)

    async def llm_call(self, parsedfile_dir): #, llm_out_save_path: str = "./node_llm_out.json"):
        """
        调用大语言模型处理各种类型的节点
        
        Args:
            parsedfile_dir (str): 解析后的文件目录路径
            llm_out_save_path (str): 大模型输出保存路径，默认为"./node_llm_out.json"
            
        Returns:
            list: 包含所有节点处理结果的列表
        """
        # self.relation_list = []
        # 读取output_dir下的所有目录
        dir_list = os.listdir(parsedfile_dir)
        node_list = []
        # 提取需要的字段与添加id
        for doc_name in dir_list:
            doc_path = os.path.join(parsedfile_dir, doc_name, "vlm")
            with open(os.path.join(doc_path, f"{doc_name}_content_list.json"), "r", encoding='utf-8') as f:
                content_list = json.load(f)
            
            # For Document
            doc_info = {
                "name": doc_name,
                "content": "".join([content["text"]+"\n\n" for content in content_list if content["type"] == "text"]),
                "type": NodeType.Document
            }
            doc_info = self.processor[NodeType.Document].add_mdhash_id(doc_info)
            node_list.append(doc_info)
            doc_id = doc_info["id"]
            
            doc_path = os.path.join(parsedfile_dir, doc_name, "vlm")
            
            chunk_idx = 0
            chunk_list = [content["text"] for content in content_list if content["type"] == "text"]
            chunk_len = len(chunk_list)
            pre_context_len = 1
            suf_context_len = 1
            
            for idx, content in enumerate(content_list):

                # For Chunk
                if content["type"] == "text":
                    chunk_info = {
                        "doc_id": doc_id,
                        "content": content["text"],
                        "type": NodeType.Chunk,
                    }
                    chunk_info = self.processor[NodeType.Chunk].add_mdhash_id(chunk_info)
                    node_list.append(chunk_info)
                    chunk_id = chunk_info["id"]
                    # chunk
                    chunk_idx += 1
                
                    
                # For Assertion
                    assertion_info = {
                        "doc_id": doc_id,
                        "id": chunk_id,
                        "content": content["text"],
                        "type": NodeType.Assertion,
                    }
                    node_list.append(assertion_info)
                    
                # For Entity 
                # 实体和断言共用call_llm
                    if not self.enable_entity_assertion_share_llm_call:
                        entity_info = {
                            "doc_id": doc_id,
                            "id": chunk_id,
                            "content": content["text"],
                            "type": NodeType.Entity,
                            "Ontology": self.Ontology,
                        }
                        node_list.append(entity_info)
                
                # For Table
                if content["type"] == "table":
                    
                    table_info = {
                        "doc_id": doc_id,
                        "content": content.get("table_body", ""),
                        "caption":content.get("table_caption", ""),
                        "img_path":os.path.join(doc_path, content["img_path"]),
                        "type": NodeType.Table,
                    }
                    table_info = self.processor[NodeType.Table].add_mdhash_id(table_info)
                    node_list.append(table_info)
                    # _table_info = await self.processor[NodeType.Table].call_llm(table_info)
                
                # For Image
                if content["type"] == "image":
                    image_info = {
                        "doc_id": doc_id,
                        "img_path":os.path.join(doc_path, content["img_path"]),
                        "caption":content["image_caption"],
                        "type": NodeType.Image,
                    }
                    image_info = self.processor[NodeType.Image].add_mdhash_id(image_info)
                    node_list.append(image_info)
                
                # For Formula
                if content["type"] == "equation":
                    # 上下文
                    start_idx = max(0, chunk_idx - pre_context_len)
                    end_idx = min(chunk_len, chunk_idx + suf_context_len)
                    context = "".join(chunk_list[start_idx:end_idx])
                    formula_info = {
                        "doc_id": doc_id,
                        "content": content["text"],
                        "type": NodeType.Formula,
                        "context": context,
                    }
                    formula_info = self.processor[NodeType.Formula].add_mdhash_id(formula_info)
                    node_list.append(formula_info)

        # 补充信息
        id2context = self.get_modal_context(node_list)
        tasks = []
        # 调用模型
        for node in node_list:
            # 补充context
            if node["id"] in id2context:
                context = "\n".join([chunk["content"] for chunk in id2context[node["id"]]])
                node["context"] = node.get("context", "")+context
            task = asyncio.create_task(
                self.processor[node['type']].call_llm(node)
            )
            tasks.append(task)      
                        
        nodelist = await tqdm_asyncio.gather(*tasks, desc="llm calling")
        return nodelist

    # 处理结构化数据
    def process_sd(self, tuple_list_or_path: List[Tuple[str, str, str]] | str, delimiter=","):
        """
        处理结构化数据
        
        Args:
            tuple_list_or_path (List[Tuple[str, str, str]] | str): 三元组列表或文件路径
            delimiter (str): 分隔符，默认为","
        """
        self.nodedata.load_from_tuple(tuple_list_or_path, delimiter)
        self.gene_edge()
    
    def save_stage_i(self, stage_i: int = 1, info = None):
        """
        保存第i个阶段的结果
        
        Args:
            stage_i (int): 阶段索引，默认为1
            info (Any): 要保存的信息
        """
        llm_call_file = os.path.join(self.cfg.data.output_dir, f"llm_call_stage_{stage_i}.json")
        with open(llm_call_file, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

    def load_stage_i(self, stage_i: int = 1):
        """
        加载第i个阶段的结果
        
        Args:
            stage_i (int): 阶段索引，默认为1
        Returns:
            Any: 加载的信息
        """
        llm_call_file = os.path.join(self.cfg.data.output_dir, f"llm_call_stage_{stage_i}.json")
        with open(llm_call_file, "r", encoding="utf-8") as f:
            return json.load(f)
    # 处理非结构化数据
    async def aprocess_ud(self, node_llm_out_path=None, stage_i: int = -1,enable_modal2entity: bool = True):
        """
        异步处理非结构化数据
        
        Args:
            node_llm_out_path (str, optional): 节点大模型输出文件路径
            stage_i (str, optional): 继续处理的阶段，-1代表根据传入文件名称自动判断
            # enable_modal2entity (bool): 是否启用模态到实体的处理，默认为True
        """
        output_dir = self.cfg.data.parse_dir
        # 根据文件自动判断阶段
        if stage_i == -1 and isinstance(node_llm_out_path, str):
            stage_i = get_stage(node_llm_out_path)
        elif node_llm_out_path is None:
            stage_i = 0
        
        # 初始化相关变量
        node_list = None
        if stage_i == 1:
            node_list = self.load_stage_i(node_llm_out_path)
        if stage_i > 1:
            self.nodedata.load(node_llm_out_path)

        if stage_i < 1: # 阶段1：LLM调用生成节点信息
            node_list = await self.llm_call(output_dir)
            self.save_stage_i(stage_i=1, info=node_list)

        if stage_i < 2: # 阶段2：从非文本元素提取实体关系
            await self.post_process(node_list, enable_modal2entity)
            self.save_stage_i(stage_i=2, info=self.nodedata.info_dict)
            logger.info(f"Entity:{len(self.nodedata.info_entity_list)}, the assertion:{len(self.nodedata.info_assertion_list)}")

        if stage_i < 3: # 阶段3：实体关系召回
            if self.cfg.dataprocessing.enable_assertion_recall:
                await self.processor[NodeType.Assertion].recall_assertion()
            if self.cfg.dataprocessing.enable_entity_recall:
                await self.processor[NodeType.Entity].recall_entity()
            self.save_stage_i(stage_i=3, info=self.nodedata.info_dict)


        if stage_i < 4: # 阶段4：实体关系对齐，以及标准化
            # 去重合并, 实体关系对齐，以及标准化
            await self.nodedata.merge()
            self.save_stage_i(stage_i=4, info=self.nodedata.info_dict)

        # 生成边
        self.gene_edge()
        # # 对大模型llm_out字段进行处理
        # if node_llm_out_path is None:
        #     node_list = await self.llm_call(output_dir)
        #     self.save_stage_i(stage_i=1, info=node_list)
        #     await self.post_process(node_list, enable_modal2entity)
        # else:
        #     await self.post_process(node_llm_out_path, enable_modal2entity)
        
        
        # logger.info(f"Entity:{len(self.nodedata.info_entity_list)}, the assertion:{len(self.nodedata.info_assertion_list)}")
        # # 关系以及实体召回
        # # 抽取抽取实体描述中存在但断言中没有的断言，断言中存在但实体中不存在的实体
        # if self.cfg.dataprocessing.enable_assertion_recall:
        #     await self.processor[NodeType.Assertion].recall_assertion()
        # if self.cfg.dataprocessing.enable_entity_recall:
        #     await self.processor[NodeType.Entity].recall_entity()



        
        

        # 将边的id进行修正
        self.edgedata.correct_merged_id(self.nodedata.merge_id_map)
        self.edgedata.correct_merged_relation(self.nodedata.relation2std_rel)
    
    def get_modal_context(self, node_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        获取模态（图标公式）的文本上下文
        
        Args:
            node_list (List[Dict[str, Any]], optional): 内容列表
        """
        modal_dict = {
            NodeType.Table: [],
            NodeType.Image: [],
            NodeType.Formula: [],
            NodeType.Entity: [],
            NodeType.Assertion: [],
            NodeType.Document: [],
            NodeType.Chunk: [],
        }
        for node in node_list:
            if node["type"] in modal_dict:
                modal_dict[node["type"]].append(node)
        
        # for table
        table_list = modal_dict[NodeType.Table]
        table_label2id_dict = self.get_label2id_dict(table_list, ["caption"], self.edgedata.table_label_match_pattern)

        # for image
        image_list = modal_dict[NodeType.Image]
        image_label2id_dict = self.get_label2id_dict(image_list, ["caption"], self.edgedata.image_label_match_pattern)

        # for formula
        formula_list = modal_dict[NodeType.Formula]
        formula_label2id_dict = self.get_label2id_dict(formula_list, ["content"], self.edgedata.formula_label_match_pattern)

        from collections import defaultdict
        # id2context
        id2context = defaultdict(list)
        chunk_list = modal_dict[NodeType.Chunk]
        for chunk in chunk_list:
            # for table
            content = chunk["content"]
            doc_id = chunk["doc_id"]
            tab_labels = self.edgedata.regex_match(content, self.edgedata.table_label_match_pattern)
            if len(tab_labels) != 0:
                for tab_label in tab_labels:
                    tab_label = doc_id + "_" + str(tab_label)
                    if tab_label in table_label2id_dict:
                        table_id = table_label2id_dict.get(tab_label)
                        id2context[table_id].append(chunk)
            # for image
            content = chunk["content"]
            img_labels = self.edgedata.regex_match(content, self.edgedata.image_label_match_pattern)
            if len(img_labels) != 0:
                for img_label in img_labels:
                    img_label = doc_id + "_" + str(img_label)
                    if img_label in image_label2id_dict:
                        image_id = image_label2id_dict.get(img_label)
                        id2context[image_id].append(chunk)
            # for formula
            content = chunk["content"]
            formula_labels = self.edgedata.regex_match(content, self.edgedata.formula_label_match_pattern)
            if len(formula_labels) != 0:
                for formula_label in formula_labels:
                    formula_label = doc_id + "_" + str(formula_label)
                    if formula_label in formula_label2id_dict:
                        formula_id = formula_label2id_dict.get(formula_label)
                        id2context[formula_id].append(chunk)
        return id2context

    def get_label2id_dict(self, node_list: List[Dict[str, Any]] = None, check_field: List = [], regex_pattern: str = "") -> Dict[str, Any]:
        """
        获取模态（图标公式）的标签2id字典
        
        Args:
            node_list (List[Dict[str, Any]], optional): 某类型的节点列表
            check_field (List, optional): 检查的字段，默认为[]
            regex_pattern (str, optional): 正则匹配模式，默认为""
        Returns:
            Dict[str, Any]: 模态（图标公式）的标签2id字典，key为标签，value为模态（图标公式）的id
        """
        node_label2id_dict = {}
        
        for node_info in node_list:
            text = ""
            for cf in check_field:
                text += str(node_info.get(cf, ""))
            node_label = self.edgedata.regex_match(text, regex_pattern)
            
            if len(node_label) != 0:
                label = node_info["doc_id"] + "_" + str(node_label[0])
                node_label2id_dict[label] = node_info["id"]
        
        return node_label2id_dict
    

    def gene_edge(self, node_list: List[Dict[str, Any]] = None):
        """
        生成图谱中的边关系
        
        Args:
            node_list (List[Dict[str, Any]], optional): 节点列表，默认为None时使用self.nodedata
        """
        if node_list is None:
            node_list = self.nodedata
        doc_list = node_list[NodeType.Document]
        chunk_list = node_list[NodeType.Chunk]
        table_list = node_list[NodeType.Table]
        image_list = node_list[NodeType.Image]
        formula_list = node_list[NodeType.Formula]
        entity_list = node_list[NodeType.Entity]
        assertion_list = node_list[NodeType.Assertion]

        # 表格，label为doc_id_+表格标号
        table_label2id_dict = self.get_label2id_dict(table_list, ["caption"], self.edgedata.table_label_match_pattern)
        image_label2id_dict = self.get_label2id_dict(image_list, ["caption"], self.edgedata.table_label_match_pattern)
        formula_label2id_dict = self.get_label2id_dict(formula_list, ["content"], self.edgedata.formula_label_match_pattern)
    

        # For Document, 暂无Document指出的边
        doc_list = node_list[NodeType.Document]
        
        # For Chunk
        chunk_list = node_list[NodeType.Chunk]
        for chunk_info in chunk_list:
            chunk_id = chunk_info["id"]
            doc_id = chunk_info["doc_id"]

            self.edgedata.add_edge(source_id=chunk_id, target_id=chunk_info["doc_id"], relation="belongs_to")
            content = chunk_info["content"]
            tab_labels = self.edgedata.regex_match(content, self.edgedata.table_label_match_pattern)
            if len(tab_labels) != 0:
                for tab_label in tab_labels:
                    tab_label = doc_id + "_" + str(tab_label)
                    if tab_label in table_label2id_dict:
                        table_id = table_label2id_dict.get(tab_label)
                        self.edgedata.add_edge(source_id=chunk_id, target_id=table_id, relation="refer_to")
            img_labels = self.edgedata.regex_match(content, self.edgedata.image_label_match_pattern)
            if len(img_labels) != 0:
                for img_label in img_labels:
                    img_label = doc_id + "_" + str(img_label)
                    if img_label in image_label2id_dict:
                        image_id = image_label2id_dict.get(img_label)
                        self.edgedata.add_edge(source_id=chunk_id, target_id=image_id, relation="refer_to")
            
            eq_labels = self.edgedata.regex_match(content, self.edgedata.formula_label_match_pattern)
            if len(eq_labels) != 0:
                for eq_label in eq_labels:
                    eq_label = doc_id + "_" + str(eq_label)
                    if eq_label in formula_label2id_dict:
                        eq_id = formula_label2id_dict.get(eq_label)
                        self.edgedata.add_edge(source_id=chunk_id, target_id=eq_id, relation="refer_to")

        chunk_id2doc_id = {}
        for chunk_info in chunk_list:
            chunk_id2doc_id[chunk_info["id"]] = chunk_info["doc_id"]
        # For Assertion
        assertion_list = node_list[NodeType.Assertion]
        for assertion_info in assertion_list:
            assertion_id = assertion_info["id"]
            chunk_id_list = []
            if "src_id" in assertion_info:
                self.edgedata.add_edge(source_id=assertion_id, target_id=assertion_info["src_id"], relation="belongs_to")
                chunk_id_list.append(assertion_info["src_id"])
            elif "src_id_list" in assertion_info:
                for src_id in assertion_info["src_id_list"]:
                    self.edgedata.add_edge(source_id=assertion_id, target_id=src_id, relation="belongs_to")
                chunk_id_list.extend(assertion_info["src_id_list"])
            
            

            head = self.processor[NodeType.Entity].compute_entity_id_by_name(assertion_info["head"])
            tail = self.processor[NodeType.Entity].compute_entity_id_by_name(assertion_info["tail"])

            # assertion_chunk_id = assertion_info["src_id"]
            # assertion_doc_id = chunk_id2doc_id.get(assertion_chunk_id, assertion_chunk_id)

            # # 检查head， tail是否为表格
            # head_tab_label = self.edgedata.regex_match(assertion_info["head"], self.edgedata.table_label_match_pattern)
            # tail_tab_label = self.edgedata.regex_match(assertion_info["tail"], self.edgedata.table_label_match_pattern)
            # if len(head_tab_label) != 0:
            #     head_tab_label = assertion_doc_id + "_" + str(head_tab_label[0])
            #     head = table_label2id_dict.get(head_tab_label, head)
            # if len(tail_tab_label) != 0:
            #     tail_tab_label = assertion_doc_id + "_" + str(tail_tab_label[0])
            #     tail = table_label2id_dict.get(tail_tab_label, tail)
            
            # # 检查head， tail是否为公式
            # head_eq_label = self.edgedata.regex_match(assertion_info["head"], self.edgedata.formula_label_match_pattern)
            # tail_eq_label = self.edgedata.regex_match(assertion_info["tail"], self.edgedata.formula_label_match_pattern)
            # if len(head_eq_label) != 0:
            #     head_eq_label = assertion_doc_id + "_" + str(head_eq_label[0])
            #     head = formula_label2id_dict.get(head_eq_label, head)
            # if len(tail_eq_label) != 0:
            #     tail_eq_label = assertion_doc_id + "_" + str(tail_eq_label[0])
            #     tail = formula_label2id_dict.get(tail_eq_label, tail)

            # # 检查head， tail是否为图
            # head_img_label = self.edgedata.regex_match(assertion_info["head"], self.edgedata.image_label_match_pattern)
            # tail_img_label = self.edgedata.regex_match(assertion_info["tail"], self.edgedata.image_label_match_pattern)
            # if len(head_img_label) != 0:
            #     head_img_label = assertion_doc_id + "_" + str(head_img_label[0])
            #     head = image_label2id_dict.get(head_img_label, head)
            # if len(tail_img_label) != 0:
            #     tail_img_label = assertion_doc_id + "_" + str(tail_img_label[0])
            #     tail = image_label2id_dict.get(tail_img_label, tail)

            relation = assertion_info["relation"]
            desc = assertion_info.get("desc", "")
            attr = assertion_info.get("attr", {})
            self.edgedata.add_edge(source_id=head, target_id=tail, relation=relation, desc=desc, attr=attr)
        
        # For Entity,其中src_id可能代表图表公式chunk
        id2node = self.nodedata.list2id_dict(self.nodedata.flattened_node_list())
        entity_list = node_list[NodeType.Entity]
        for entity_info in entity_list:
            entity_id = entity_info["id"]
            # print(entity_info) node合并后chunk_id字段去掉了，不能node合并后再生成边
            if "src_id" in entity_info:
                target_id = entity_info["src_id"]
                node = id2node.get(target_id)
                if node.get("type") in [NodeType.Table, NodeType.Image, NodeType.Formula]:
                    relation = "relate_to"
                else:
                    relation = "locate"
                self.edgedata.add_edge(source_id=entity_id, target_id=target_id, relation=relation)
            elif "src_id_list" in entity_info:
                for src_id in entity_info["src_id_list"]:
                    node = id2node.get(src_id)
                    if node.get("type") in [NodeType.Table, NodeType.Image, NodeType.Formula]:
                        relation = "relate_to"
                    else:
                        relation = "locate"
                    self.edgedata.add_edge(source_id=entity_id, target_id=src_id, relation=relation)
        
        # For Table
        table_list = node_list[NodeType.Table]
        for table_info in table_list:
            table_id = table_info["id"]
            self.edgedata.add_edge(source_id=table_id, target_id=table_info["doc_id"], relation="locate")
        # For Image
        for image_info in image_list:
            image_id = image_info["id"]
            self.edgedata.add_edge(source_id=image_id, target_id=image_info["doc_id"], relation="locate")
        # For Formula
        for formula_info in formula_list:
            formula_id = formula_info["id"]
            self.edgedata.add_edge(source_id=formula_id, target_id=formula_info["doc_id"], relation="locate")
        
        # 边去重
        self.edgedata.deduplication()
        
        
    async def modal2entity_process(self, node_list: List[Dict[str, Any]]):
        """
        处理模态数据到实体的转换过程
        
        Args:
            node_list (List[Dict[str, Any]]): 节点列表
            
        Returns:
            list: 处理后的节点列表
        """
        tasks = []
        for node in node_list:
            task = asyncio.create_task(
                self.processor[NodeType.Entity].call_llm(node)
            )
            tasks.append(task)
        node_list = await tqdm_asyncio.gather(*tasks, desc="modal2entity calling")
        for node in tqdm(node_list, desc="modal2entity post processing"):
            node["from_modal"] = True
            self.processor[NodeType.Entity].post_process(node)
            # self.processor[NodeType.Assertion].post_process(node)
        return node_list

    def save_node(self):
        """
        保存节点数据到文件
        """
        path = os.path.join(self.cfg.data.output_dir, "node_list.json")
        self.nodedata.save(path)
    
    def save_edge(self):
        """
        保存边数据到文件
        """
        path = os.path.join(self.cfg.data.output_dir, "edge_list.json")
        self.edgedata.save(path)
    
    def json2graph(self, node_list_path: str = "./output_dir/node_list.json", 
        edge_list_path: str = "./output_dir/edge_list.json"):
        """
        将JSON格式的数据转换为图结构
        
        Args:
            node_list_path (str): 节点列表文件路径，默认为"./output_dir/node_list.json"
            edge_list_path (str): 边列表文件路径，默认为"./output_dir/edge_list.json"
        """
        node = self.nodedata
        node.load(node_list_path)
        nodelist = node.flattened_node_list()
        node_list_flat_path = os.path.join(self.cfg.data.output_dir, "node_list_flat.json")
        with open(node_list_flat_path, "w", encoding="utf-8") as f:
            json.dump(nodelist, f, ensure_ascii=False)

        edge = EdgeData()
        edge.load(edge_list_path)
        edgelist = edge.relation_list

        node_id_list = [node["id"] for node in nodelist]
        print("node_id_list:", len(set(node_id_list)))
        print("node:", len(nodelist), "edge:", len(edgelist))
        G = jg.build_kg(nodelist, edge.relation_list)
        # G = jg.serialize_lists(G)
        graph_path = os.path.join(self.cfg.data.output_dir, "graph.graphml")
        jg.save(G, graph_path)
        
    def visualize_kg(self, entities: str| List[Dict[str, Any]], 
            relations: str | List[Dict[str, Any]], 
            file_name: str = "./output_dir/knowledge_graph.html",
            vis_node_types: List[str] = node_type_list):
        """
        可视化知识图谱
        
        Args:
            entities (str | List[Dict[str, Any]]): 实体数据或文件路径
            relations (str | List[Dict[str, Any]]): 关系数据或文件路径
            file_name (str): 可视化结果保存文件名，默认为"./output_dir/knowledge_graph.html"
            vis_node_types (List[str]): 需要可视化的节点类型列表，默认为所有节点类型
        """
        jg.visualize_kg_with_legend(entities=entities, 
        relations=relations, 
        file_name=file_name,
        vis_node_types=vis_node_types)