""""
节点数据管理类
# 定义七类节点
# 文档、chunk、断言、实体、表格、图片、公式
"""

import os
from typing import List, Dict, Tuple, Any
from enum import Enum
import json
from collections import defaultdict
from omegaconf import DictConfig
from llms.emb import EmbeddingClient
from llms.client import AsyncLLMClient
from util.tool import compute_mdhash_id
from prompts.dataprocess_prompt import PROMPTS
from util.jsonparser import parse_json
from prompts.dataprocess_prompt import PROMPTS


class NodeType:
    Document = "Document"
    Chunk = "Chunk"
    Assertion = "Assertion"
    Entity = "Entity"
    Table = "Table"
    Image = "Image"
    Formula = "Formula"
    
node_type_list = [
    NodeType.Document,
    NodeType.Chunk,
    NodeType.Assertion,
    NodeType.Entity,
    NodeType.Table,
    NodeType.Image,
    NodeType.Formula
]
    
PREFIX_MAP = {
    "Document": "doc-",
    "Chunk": "chk-",
    "Assertion": "ass-",
    "Entity": "ent-",
    "Table": "tbl-",
    "Image": "img-",
    "Formula": "fml-"      
}

class NodeData:
    """
    节点数据管理类，用于处理和管理各种类型的节点数据
    """
    def __init__(self, cfg: DictConfig):
        """
        初始化NodeData实例
        
        Args:
            cfg (DictConfig): 配置对象，包含embedding模型的相关配置
        """
        self.info_doc_list = []
        self.info_chunk_list = []
        self.info_assertion_list = []
        self.info_entity_list = []
        self.info_table_list = []
        self.info_image_list = []
        self.info_formula_list = []
        self.cfg = cfg
        self.emb = EmbeddingClient(base_url=self.cfg.embedding_model.base_url,
            api_key=self.cfg.embedding_model.api_key,
            model=self.cfg.embedding_model.model,
        )

        self.info_dict = {
                NodeType.Document: self.info_doc_list,
                NodeType.Chunk: self.info_chunk_list,
                NodeType.Assertion: self.info_assertion_list,
                NodeType.Entity: self.info_entity_list,
                NodeType.Table: self.info_table_list,
                NodeType.Image: self.info_image_list,
                NodeType.Formula: self.info_formula_list
        }
        # 合并前的原id->新的id, 主要针对entity和assertion, 目的是后期构造边时进行id转换
        self.merge_id_map = {}
        # 别称->标准化名字
        self.alias2std_entity_info = {}

        # 关系名->标准化名字
        self.relation2std_rel = {}
        self.client = AsyncLLMClient(api_key=cfg.dataprocessing.llm.api_key, base_url=cfg.dataprocessing.llm.base_url, model=cfg.dataprocessing.llm.model,
                                    max_concurrent_requests=cfg.dataprocessing.llm.max_concurrent_requests)
        
    
    def make_message(self, system_prompt, user_prompt):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def save(self, path):
        """
        将节点数据保存到指定路径的JSON文件中
        
        Args:
            path (str): 保存文件的路径
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                NodeType.Document: self.info_doc_list,
                NodeType.Chunk: self.info_chunk_list,
                NodeType.Assertion: self.info_assertion_list,
                NodeType.Entity: self.info_entity_list,
                NodeType.Table: self.info_table_list,
                NodeType.Image: self.info_image_list,
                NodeType.Formula: self.info_formula_list
            }, f, ensure_ascii=False, indent=2)

    def list2id_dict(self, node_list: List[Dict[str, Any]]):
        """
        将节点列表转换为以节点ID为键的字典
        
        Args:
            node_list (List[Dict[str, Any]]): 节点列表
            
        Returns:
            dict: 以节点ID为键，节点对象为值的字典
        """
        return {node["id"]: node for node in node_list}
    
    def load(self, node_list_or_path: str | List[Dict[str, Any]]):
        """
        从文件路径或节点列表加载节点数据
        
        Args:
            node_list_or_path (str | List[Dict[str, Any]]): 文件路径或节点列表
        """
        if isinstance(node_list_or_path, str):
            path = node_list_or_path
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = node_list_or_path

        self.info_dict = data
        self.info_doc_list = data[NodeType.Document]
        self.info_chunk_list = data[NodeType.Chunk]
        self.info_assertion_list = data[NodeType.Assertion]
        self.info_entity_list = data[NodeType.Entity]
        self.info_table_list = data[NodeType.Table]
        self.info_image_list = data[NodeType.Image]
        self.info_formula_list = data[NodeType.Formula]

    def load_from_tuple(self, tuple_list_or_path: List[Tuple[str, str, str]] | str, delimiter=","):
        """
        从三元组列表或文件路径加载数据，生成断言和实体节点
        
        Args:
            tuple_list_or_path (List[Tuple[str, str, str]] | str): 三元组列表或文件路径
            delimiter (str): 文件中字段的分隔符，默认为逗号
        """
        if isinstance(tuple_list_or_path, str):
            with open(tuple_list_or_path, "r", encoding="utf-8") as f:
                tuple_list = [tuple(line.strip().split(delimiter)) for line in f]
        else:
            tuple_list = tuple_list_or_path
        
        # for assertion and entity
        tuple_list = list(set(tuple_list))
        entity_list = []
        for triple in tuple_list:
            if len(triple) != 3: continue
            head, relation, tail = triple
            assertion_id=compute_mdhash_id(head+"|"+relation+"|"+ tail, PREFIX_MAP[NodeType.Assertion])
            self.info_assertion_list.append({
                "id": assertion_id,
                "name": " ".join([head, relation, tail]),
                "head": head,
                "relation": relation,
                "tail": tail,
                "type": NodeType.Assertion,
            })
            entity_list.append(head)
            entity_list.append(tail)
        
        entity_list = list(set(entity_list))
        for ent in entity_list:
            ent_id = compute_mdhash_id(ent, PREFIX_MAP[NodeType.Entity])

            self.info_entity_list.append({
                "id": ent_id,
                "name": ent,
                "type": NodeType.Entity,
            })

    def get_nodelist(self, key):
        """
        根据节点类型获取对应的节点列表
        
        Args:
            key (str): 节点类型键
            
        Returns:
            list: 对应类型的节点列表
        """
        return self.info_dict[key]
    
    def __getitem__(self, key):
        """
        通过索引访问节点数据
        
        Args:
            key (str): 节点类型键
            
        Returns:
            list or attribute: 对应类型的节点列表或其他属性
            
        Raises:
            KeyError: 当指定的键不存在时抛出异常
        """
        if key in self.info_dict:
            return self.info_dict[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"属性 '{key}' 不存在")

    def flattened_node_list(self):
        """
        将所有类型的节点合并成一个扁平化的列表
        
        Returns:
            list: 包含所有节点的列表
        """
        node_list = []
        for key in self.info_dict.keys():
            if not isinstance(self.info_dict[key], list):
                print(key, self.info_dict[key])
            node_list.extend(self.info_dict[key])
        return node_list

    def _merge_single_group(self, entity_group: List[Dict[str, Any]], mode: str = "id") -> Dict[str, Any]:
        """
        合并具有相同ID的实体组
        
        Args:
            entity_group (List[Dict[str, Any]]): 具有相同ID的实体列表
            
        Returns:
            Dict[str, Any]: 合并后的实体字典
        """
        first_entity = None
        for entity in entity_group:
            if entity.get("from_modal"):
                # 找到组中第一个来自纯文本的断言
                continue
            first_entity = entity.copy()
            break
        if not first_entity:  # 如果没有来自纯文本的断言，下一组
            return None


        if mode == "id":
            # id映射
            for entity in entity_group:
                self.merge_id_map[entity["id"]] = first_entity["id"]
            name = first_entity["name"]
            desc = first_entity["desc"]
            # 来源处理
            src_id_list = []
            for entity in entity_group:
                src_id = entity.get("src_id", None)
                if src_id: src_id_list.append(src_id)
                else:
                    src_id_list.extend(entity.get("src_id_list", []))
            # 别名处理
            alias_list = []
            for entity in entity_group:
                alias = entity.get("alias", None)
                if alias: alias_list.extend(alias)
                else:
                    alias_list.append(entity.get("name"))
            # 基础信息（取第一个）
            merged = {
                "id": first_entity["id"],
                # 别名
                "alias": list(set(alias_list)), # json.dumps(list(set(alias_list)), ensure_ascii=False),
                "type": first_entity["type"],
                "name": name, # first_entity["name"],
                "src_id_list": list(set(src_id_list)), # json.dumps(list(set(src_id_list)), ensure_ascii=False),
                # 合并desc（用\n分隔）
                "desc": desc, # "\n".join([e["desc"] for e in entity_group if e.get("desc")]),
                # 合并attr
                "attr": self._merge_attributes([e["attr"] for e in entity_group])
            }
            return merged
        elif mode == "sim":
            # 标准化的实体信息
            # std_ent_info = self.alias2std_entity_info.get(first_entity["name"], None)
            # name = None
            # desc = None
            # if std_ent_info:
            #     name = std_ent_info.get("name", None)
            #     desc = std_ent_info.get("desc", None)
            #     if not name or not desc:
            #         return None
            # else:
            #     return None
            from collections import defaultdict
            # 分组
            std_name2entity_list = defaultdict(list)
            for entity in entity_group:
                std_ent_info = self.alias2std_entity_info.get(entity["name"], None)
                if std_ent_info:
                    std_name = std_ent_info.get("name", None)
                    if std_name:
                        std_name2entity_list[std_name].append(entity)
            
            # 合并
            merged_list = []
            for std_name, group in std_name2entity_list.items():
                first_entity = group[0]
                std_ent_info = self.alias2std_entity_info.get(first_entity["name"], None)
                std_name = std_ent_info.get("name", '')
                src_id_list = []
                for entity in group:
                    # 来源处理
                    src_id = entity.get("src_id", None)
                    if src_id: src_id_list.append(src_id)
                    else:
                        src_id_list.extend(entity.get("src_id_list", []))
                    
                    # id映射
                    self.merge_id_map[entity["id"]] = std_ent_info["id"]
                    
                # 别名
                alias_list = std_ent_info.get("alias", [])
                desc = std_ent_info.get("desc", '')
                # 基础信息（取第一个）
                merged = {
                    "id": std_ent_info["id"],
                    # 别名
                    "alias": list(set(alias_list)), # json.dumps(list(set(alias_list)), ensure_ascii=False),
                    "type": first_entity["type"],
                    "name": std_name, # first_entity["name"],
                    "src_id_list": list(set(src_id_list)), # json.dumps(list(set(src_id_list)), ensure_ascii=False),
                    # 合并desc（用\n分隔）
                    "desc": desc, # "\n".join([e["desc"] for e in entity_group if e.get("desc")]),
                    # 合并attr
                    "attr": self._merge_attributes([e["attr"] for e in group])
                }
                merged_list.append(merged)
            return merged_list
        return None

    def _merge_attributes(self, attr_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并属性字典列表
        
        Args:
            attr_list (List[Dict[str, Any]]): 属性字典列表
            
        Returns:
            Dict[str, Any]: 合并后的属性字典
        """
        merged_attr = {}
        
        for attr in attr_list:
            for key, value in attr.items():
                if key not in merged_attr:
                    if isinstance(value, list):
                        merged_attr[key] = value
                    else:
                        merged_attr[key] = [value]
                elif isinstance(merged_attr[key], list) and isinstance(value, list):
                    merged_attr[key].extend(value)
                elif merged_attr[key] != value:
                    # 值不同，转换为列表存储
                    if not isinstance(merged_attr[key], list):
                        merged_attr[key] = [merged_attr[key]]
                    merged_attr[key].append(value)
        merged_attr = {k: list(set(v)) for k, v in merged_attr.items() if v}
        return merged_attr

    def merge_document_by_id(self):
        """
        根据ID去重文档节点，相同ID的文档只保留一个
        """
        merged_dict = defaultdict(list)
        for doc in self.info_doc_list:
            merged_dict[doc["id"]] = doc
        
        self.info_doc_list = list(merged_dict.values())
        self.info_dict[NodeType.Document] = self.info_doc_list
    
    def merge_chunk_by_id(self):
        """
        根据ID合并Chunk节点，相同ID的Chunk只保留一个
        """
        merged_dict = defaultdict(list)
        
        # 按ID分组
        for chunk in self.info_chunk_list:
            merged_dict[chunk["id"]] = chunk
        
        # 获取第一个Chunk的ID
        self.info_chunk_list = list(merged_dict.values())
        self.info_dict[NodeType.Chunk] = self.info_chunk_list

    def merge_entity_by_id(self):
        """
        根据ID合并实体节点
        
        Returns:
            list: 合并后的实体节点列表
        """
        merged_dict = defaultdict(list)
        
        # 按ID分组
        for entity in self.info_entity_list:
            merged_dict[entity["id"]].append(entity)
        
        # 合并每个组
        result = []
        for entity_id, entity_list in merged_dict.items():
            # if len(entity_list) == 1:
            #     # 单个实体，直接添加（移除chunk_id）
            #     entity = entity_list[0].copy()
            #     entity.pop("src_id", None)
            #     result.append(entity)
            # else:
                # 合并多个实体
            merged_entity = self._merge_single_group(entity_group=entity_list, mode="id")
            if not merged_entity:
                continue
            result.append(merged_entity)
        self.info_entity_list = result
        self.info_dict[NodeType.Entity] = result
        return result
    
    async def merge_entity_by_sim(self, threshold: float = 0.95):
        """
        根据嵌入相似度合并实体节点
        
        Args:
            threshold (float): 相似度阈值，默认为0.95
            
        Returns:
            list: 合并后的实体节点列表
        """
        print("使用嵌入相似度进行实体合并")
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(embedding_model)
        # embedding_func = model.encode
        from sklearn.metrics.pairwise import cosine_similarity
        embedding_func = self.emb.get_embedding
        name_list = [ent["name"] for ent in self.info_entity_list]
        embeddings = embedding_func(name_list)
        id_to_vec = {ent["id"]: embedding for ent, embedding in zip(self.info_entity_list, embeddings)}
        
        # 2. 聚类：相似度 >= threshold 的放在同一连通分量
        visited = set()
        clusters = []          # 每个元素是一个 list[entity]

        for ent in self.info_entity_list:
            if ent["id"] in visited:
                continue
            cluster = [ent]
            visited.add(ent["id"])
            # BFS 找所有连通实体
            queue = [ent]
            while queue:
                cur = queue.pop()
                cur_vec = id_to_vec[cur["id"]]
                for other in self.info_entity_list:
                    if other["id"] in visited:
                        continue
                    sim = cosine_similarity(cur_vec.reshape(1, -1), id_to_vec[other["id"]].reshape(1, -1))
                    if sim >= threshold:
                        cluster.append(other)
                        visited.add(other["id"])
                        queue.append(other)
            clusters.append(cluster)

        # 标准化实体映射
        await self.get_std_entity_map(clusters)
        # 3. 合并每个簇
        merged_entities = []
        for group in clusters:
            if len(group) == 1:
                single = group[0].copy()
                if single.get("from_modal"):
                    continue
                src_id = single.get("src_id", None)
                if src_id: # 如果其src_id没有被合并过be_merged的，则加入src_id_list
                    single["src_id_list"] = [single["src_id"]]
                    single.pop("src_id", None)
                self.merge_id_map[single["id"]] = single["id"]
                merged_entities.append(single)
            else:
                merged_entity = self._merge_single_group(entity_group=group, mode="sim")
                if not merged_entity:
                    continue
                if isinstance(merged_entity, list):
                    merged_entities.extend(merged_entity)
                else:
                    merged_entities.append(merged_entity)

        # 4. 更改所有实体名相关的节点，
        new_assertion_list = []
        for assertion in self.info_assertion_list:
            head = assertion['head']
            tail = assertion['tail']
            nhead = self.alias2std_entity_info.get(head)
            if nhead:
                nhead = nhead['name']
            else:
                nhead = head
            ntail = self.alias2std_entity_info.get(tail)
            if ntail:
                ntail = ntail['name']
            else:
                ntail = tail
            assertion['head'] = nhead
            assertion['tail'] = ntail
            new_assertion_list.append(assertion)
        self.info_assertion_list = new_assertion_list
        self.info_dict[NodeType.Assertion] = self.info_assertion_list

        # 5. 写回
        self.info_entity_list = merged_entities
        self.info_dict[NodeType.Entity] = self.info_entity_list
        return merged_entities
    
    def merge_assertion_by_id(self):
        """
        根据ID合并断言节点，相同ID的断言只保留一个
        """
        merged_dict = defaultdict(list)
        
        # 相同ID的断言只保留一个
        for assertion in self.info_assertion_list:
            merged_dict[assertion["id"]] = assertion
        
        # 更新存储
        self.info_assertion_list = list(merged_dict.values())
        self.info_dict[NodeType.Assertion] = self.info_assertion_list

    def single_pass(self, threshold: float = 0.95):
        """
        根据相似度合并断言节点
        
        Args:
            threshold (float): 相似度阈值，默认为0.95
        """
        from sklearn.metrics.pairwise import cosine_similarity
        embedding_func = self.emb.get_embedding
        
        # 创建断言的文本表示用于嵌入计算
        assertion_texts = []
        for assertion in self.info_assertion_list:
            # 使用 head relation tail 的组合来表示断言 # 可能得换，head和tail可能及其不一样 alias2name
            # text = f"{assertion['head']} {assertion['relation']} {assertion['tail']}"
            text = assertion['relation']
            assertion_texts.append(text)
        
        if not assertion_texts:
            return
            
        embeddings = embedding_func(assertion_texts)
        id_to_vec = {assertion["id"]: embedding for assertion, embedding in zip(self.info_assertion_list, embeddings)}

        # 聚类：相似度 >= threshold 的放在同一连通分量
        visited = set()
        clusters = []          # 每个元素是一个 list[assertion]

        for assertion in self.info_assertion_list:
            if assertion["id"] in visited:
                continue
            cluster = [assertion]
            visited.add(assertion["id"])
            # BFS 找所有连通断言
            queue = [assertion]
            while queue:
                cur = queue.pop()
                cur_vec = id_to_vec[cur["id"]]
                for other in self.info_assertion_list:
                    if other["id"] in visited:
                        continue
                    sim = cosine_similarity(cur_vec.reshape(1, -1), id_to_vec[other["id"]].reshape(1, -1))
                    if sim >= threshold:
                        cluster.append(other)
                        visited.add(other["id"])
                        queue.append(other)
            clusters.append(cluster)
        return clusters
        
    async def merge_assertion_by_sim(self, threshold: float = 0.95):
        """
        根据相似度合并断言节点
        
        Args:
            threshold (float): 相似度阈值，默认为0.95
        """
        clusters = self.single_pass(threshold=threshold)

        # 获取标准化关系映射
        await self.get_std_relation_map(clusters)

        # 关系名标准化,
        rel_list = []
        from collections import defaultdict
        clusters = defaultdict(list)
        for rel in self.info_assertion_list:
            rel_name = rel.get('relation')
            std_rel_name = self.relation2std_rel.get(rel_name, rel_name)
            rel['relation'] = std_rel_name
            rel_k = f"{rel['head']}-{std_rel_name}-{rel['tail']}"
            clusters[rel_k].append(rel)
        
        # 合并每个簇
        merged_assertions = []
        for group in clusters.values():
            if len(group) == 1:
                # 单个断言，直接添加
                # 如果来自多模态数据
                single = group[0].copy()
                if single.get("from_modal"):  # 如果来自多模态数据, 纯文本中无该实体且和纯文本中实体相似度不达标，则抛弃
                    continue
                src_id = single.get("src_id", None)
                if src_id: # 如果其src_id没有被合并过be_merged的，则加入src_id_list
                    single["src_id_list"] = [single["src_id"]]
                    single.pop("src_id", None)
                self.merge_id_map[single["id"]] = single["id"]
                merged_assertions.append(single)
            else:
                # 合并多个断言，使用第一个断言的信息作为基础
                first_assertion = None
                for assertion in group:
                    if assertion.get("from_modal"):
                        # 找到组中第一个来自纯文本的断言
                        continue
                    first_assertion = assertion.copy()
                    break
                if not first_assertion:  # 如果没有来自纯文本的断言，下一组
                    continue
                for assertion in group:
                    self.merge_id_map[assertion["id"]] = first_assertion["id"]
                # 标准化的关系
                # std_rel = self.relation2std_rel.get(first_assertion["relation"], None)
                # relation = None
                # if std_rel:
                #     relation = std_rel.get("name", None)
                #     if not relation: continue
                # else:
                #     continue
                

                # 来源列表
                src_id_list = []
                for assertion in group:
                    src_id = assertion.get("src_id", None)
                    if src_id: src_id_list.append(src_id)
                    else:
                        src_id_list.extend(assertion.get("src_id_list", []))
                # 关系别名
                alias_list = []
                for assertion in group:
                    alias = assertion.get("alias", None)
                    if alias: alias_list.extend(alias)
                    else:
                        alias_list.append(assertion.get("relation"))
                    
                merged = {
                    "id": first_assertion["id"],
                    "name": first_assertion["name"],
                    "src_id_list": list(set([r["src_id"] for r in group if r.get("src_id")])), # json.dumps(list(set([r["src_id"] for r in group if r.get("src_id")])), ensure_ascii=False),
                    "head": first_assertion["head"],
                    "relation": first_assertion["relation"],
                    "tail": first_assertion["tail"],
                    "alias": list(set(alias_list)), # list(set([r["relation"] for r in group if r.get("relation")])), # json.dumps(list(set([r["relation"] for r in group if r.get("relation")])), ensure_ascii=False),
                    "desc": "\n".join([r["desc"] for r in group if r.get("desc")]),
                    "type": first_assertion["type"],
                }
                merged_assertions.append(merged)
        
        # 更新存储
        self.info_assertion_list = merged_assertions
        self.info_dict[NodeType.Assertion] = self.info_assertion_list

    def merge_table_by_id(self):
        """
        根据ID合并表格节点，相同ID的表格只保留一个
        """
        merged_dict = defaultdict(list)
        
        # 按ID分组
        for table in self.info_table_list:
            merged_dict[table["id"]] = table
        
        # 获取第一个Assertion的ID
        self.info_table_list = list(merged_dict.values())
        self.info_dict[NodeType.Table] = self.info_table_list

    def merge_formula_by_id(self):
        """
        根据ID合并公式节点，相同ID的公式只保留一个
        """
        merged_dict = defaultdict(list)
        
        # 按ID分组
        for formula in self.info_formula_list:
            merged_dict[formula["id"]] = formula
        
        # 获取第一个Assertion的ID
        self.info_formula_list = list(merged_dict.values())
        self.info_dict[NodeType.Formula] = self.info_formula_list

    def merge_image_by_id(self):
        """
        根据ID合并图片节点，相同ID的图片只保留一个
        """
        merged_dict = defaultdict(list)
        
        # 按ID分组
        for image in self.info_image_list:
            merged_dict[image["id"]] = image
        
        # 获取第一个Assertion的ID
        self.info_image_list = list(merged_dict.values())
        self.info_dict[NodeType.Image] = self.info_image_list

    # 标准化实体
    async def get_std_entity_map(self, group_list: List[Dict[str, Any]]):
        message_list = []
        merge_group = []
        for group in group_list:
            sys = PROMPTS["entity_standard_sys"]
            group = [{"name": e["name"], "desc": e["desc"]} for e in group]
            if len(group) == 1: continue
            merge_group.append(group)
            user = PROMPTS["entity_standard_user"].format(entity_info=str(group))
            message = self.make_message(system_prompt=sys, user_prompt=user)
            message_list.append(message)
        
        responses = await self.client.agenerate_batch( 
            messages_list=message_list,
            temperature=0.1,
            max_tokens=4096,
            task_desc="entity standard calling"
        )
        # 处理响应
        self.entity_std_log = []
        for i, res in enumerate(responses):
            self.entity_std_log.append({
                "group": merge_group[i],
                "res": res['choices'][0]['message']['content'].strip()
            })
            if isinstance(res, Exception):
                logger.error(f" 请求 {i+1} 错误: {str(res)}")
                continue
            else:
                content = res['choices'][0]['message']['content'].strip()
                # result.append(content)
                group = merge_group[i]
                ent_dict = parse_json(content)
            
                if isinstance(ent_dict, list) and len(ent_dict) > 0:
                    std_ent = ent_dict
                    for ent in std_ent:
                        name = ent.get('name', None)
                        std_id = compute_mdhash_id(name, prefix=PREFIX_MAP[NodeType.Entity])
                        ent["id"] = std_id
                        alias_list = ent.get('alias', None)
                        # try:
                        #     alias_list = json.loads(alias_list)
                        # except:
                        #     continue
                        for al in alias_list:
                            self.alias2std_entity_info[al] = ent
        
            entity_std_file = os.path.join(self.cfg.data.output_dir, "entity_std.json")
            with open(entity_std_file, "w", encoding="utf-8") as f:
                json.dump(self.alias2std_entity_info, f, ensure_ascii=False, indent=4)
                # elif isinstance(ent_dict, dict):
                #     std_ent = ent_dict
                # else:
                #     std_ent = {}
                # name = std_ent.get('name', None)
                # if not name:
                #     for ent in group:
                #         self.alias2std_entity_info[ent['name']] = std_ent
                #     continue
                # for ent in group:
                #     self.alias2std_entity_info[ent['name']] = std_ent
        return self.alias2std_entity_info

    # 关系标准化        
    async def get_std_relation_map(self, group_list: List[Dict[str, Any]]):
        message_list = []
        merge_group = []
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        compl_delim = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        for group in group_list:
            sys = PROMPTS["relation_standard_sys"]
            group = [r["relation"]for r in group]
            group = list(set(group))
            if len(group) == 1:
                continue
            merge_group.append(group)
            user = PROMPTS["relation_standard_user"].format(relation_info=str(group), tuple_delimiter=tuple_delimiter, completion_delimiter=compl_delim)
            message = self.make_message(system_prompt=sys, user_prompt=user)
            message_list.append(message)
        
        responses = await self.client.agenerate_batch( 
            messages_list=message_list,
            temperature=0.1,
            max_tokens=4096,
            task_desc="relation standard calling"
        )
        # 处理响应
        # result = []
        self.relation_std_log = []
        for i, res in enumerate(responses):
            self.relation_std_log.append({
                "group": merge_group[i],
                "res": res['choices'][0]['message']['content'].strip()
            })
            if isinstance(res, Exception):
                logger.error(f" 请求 {i+1} 错误: {str(res)}")
                continue
            else:
                content = res['choices'][0]['message']['content'].strip()
                # result.append(content)
                group = group_list[i]
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                for ln in lines:
                    parts = ln.split(tuple_delimiter)
                    if len(parts) == 3 and parts[0] == "relation_std_map":
                        rel_std = parts[1]
                        alias_list = parts[2].split(',')
                        for r_name in alias_list:
                            r_name = r_name.strip()
                            self.relation2std_rel[r_name] = rel_std

        relation_std_file = os.path.join(self.cfg.data.output_dir, "relation_std.json")
        with open(relation_std_file, "w", encoding="utf-8") as f:
            json.dump(self.relation2std_rel, f, ensure_ascii=False, indent=4)
                # elif isinstance(rel_dict, dict):
                #     relation = rel_dict.get("relation", None)
                # else:
                #     relation = {}
                # relation = rel_dict.get("name", None)
                # if not relation:
                #     for rel in group:
                #         self.relation2std_rel[rel["relation"]] = rel_dict
                #     continue
                # for rel in group:
                #     self.relation2std_rel[rel["relation"]] = rel_dict
        return self.relation2std_rel

    async def merge(self):
        """
        执行所有类型的节点合并操作
        """
        # 节点去重或合并
        self.merge_document_by_id()
        self.merge_chunk_by_id()
        self.merge_entity_by_id()
        self.merge_assertion_by_id()
        self.merge_table_by_id()
        self.merge_formula_by_id()
        self.merge_image_by_id()

        if self.cfg.enable_merge_entity_by_sim:
            await self.merge_entity_by_sim(self.cfg.merge_entity_by_sim.threshold)
        if self.cfg.enable_merge_assertion_by_sim:
            await self.merge_assertion_by_sim(self.cfg.merge_assertion_by_sim.threshold)

        # 保存实体标准化关系标准化的log
        entity_std_log_file = os.path.join(self.cfg.data.output_dir, "entity_std_log.json")
        with open(entity_std_log_file, "w", encoding="utf-8") as f:
            json.dump(self.entity_std_log, f, ensure_ascii=False, indent=4)
        relation_std_log_file = os.path.join(self.cfg.data.output_dir, "relation_std_log.json")
        with open(relation_std_log_file, "w", encoding="utf-8") as f:
            json.dump(self.relation_std_log, f, ensure_ascii=False, indent=4)