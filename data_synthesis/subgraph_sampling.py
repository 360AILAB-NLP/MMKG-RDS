"""
知识图谱子图采样器
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import random
import networkx as nx
from typing import *
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from data_synthesis.net_utils import *


logger = logging.getLogger(__name__)


class SubGraphSampler(ABC):
    """子图采样器"""
    def __init__(self, G: nx.Graph, order: int, **kwargs):
        """
        初始化采样器
        Args:
            G: 知识图谱NX有向图
            order: 子图阶数(节点数)
            kwargs: 其他参数
        """
        self.G = G
        self.node_ids = list(self.G.nodes()) 
        self.relations = get_all_relations(self.G, format="json")
        self.order = order  # 子图阶数(节点数)
        self.config = kwargs

    @abstractmethod
    def sample_subgraph(self):
        raise NotImplementedError

    def _get_sampled_relations(self, sampled_nodes: List[str], relations: List[Dict]) -> List[Dict]:
        """获取采样节点之间的关系"""
        sampled_relations = []
        for relation in relations:
            head = relation.get('head', '')
            tail = relation.get('tail', '')
            if head in sampled_nodes and tail in sampled_nodes:
                sampled_relations.append(relation)
        return sampled_relations

    def subgraph_statistics(self, ):
        """子图统计信息"""
        pass
        

@dataclass
class SubgraphSamplingOutput:
    """
    子图采样算法的返回结果容器。

    Attributes
    ----------
    node_ids : list[str]
        被采样到的实体节点 ID 列表（去重且有序）。
    nodes : dict[str, Any]
        节点 ID → 节点属性字典。
    relations : list[dict[str, Any]]
        子图中存在的关系列表，每条关系至少包含 ``head``、``tail``等字段。
    subgraph_sample_algorithm : str
        子图采样策略名称，例如 ``"random_walk"``。
    start_node : str
        起始节点 ID。
    subgraph_order : int
        子图阶数，即节点数目。
    """
    node_ids: List[str]  
    nodes: Dict  
    relations: List[dict]  
    subgraph_sample_algorithm: str  
    start_node: str 
    subgraph_order: int  

    def __post_init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({items})"

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


class DefaultSampler(SubGraphSampler):
    """不进行子图采样，选择整个知识图谱"""
    def __init__(self, G: nx.Graph, order: int, **kwargs):
        super().__init__(G, order, **kwargs)

    def sample_subgraph(self):
        sampled_node_ids = self.node_ids
        output = SubgraphSamplingOutput(
                    node_ids=sampled_node_ids,
                    nodes={node: node_attr(self.G, node) for node in sampled_node_ids},  
                    relations=self.relations,
                    subgraph_sample_algorithm='no_subgraph_sampling',
                    start_node=None,  # 未选择首节点
                    subgraph_order=len(sampled_node_ids)
                )
        logger.info(f"采样了 {len(sampled_node_ids)} 个节点和 {len(self.relations)} 个关系")
        return output


class RandomSampler(SubGraphSampler):
    """随机选择节点"""
    def __init__(self, G: nx.Graph, order: int, **kwargs):
        super().__init__(G, order, **kwargs)

    def sample_subgraph(self):
        max_tries = 10
        sampled_relations = []
        for _ in range(max_tries):
            sampled_node_ids = random.sample(self.node_ids, min(len(self.node_ids), self.order))  
            sampled_relations = self._get_sampled_relations(sampled_node_ids, relations=self.relations)
            if sampled_relations: break
            sampled_relations = list(sampled_relations)

        output = SubgraphSamplingOutput(
                    node_ids=sampled_node_ids,
                    nodes={node: node_attr(self.G, node) for node in sampled_node_ids},
                    relations=sampled_relations,
                    subgraph_sample_algorithm='random',
                    start_node=sampled_node_ids[0],
                    subgraph_order=len(sampled_node_ids)
                )
        
        logger.info(f"采样了 {len(sampled_node_ids)} 个节点和 {len(sampled_relations)} 个关系")
        return output


class AugmentedChainSampler(SubGraphSampler):
    """主干增强采样 (Augmented Chain Sampling)
    
    先找到核心逻辑链，然后用相关节点来丰满它
    """
    def __init__(self, G: nx.Graph, order: int, **kwargs):
        super().__init__(G, order, **kwargs)

    def sample_subgraph(self):
        try:
            nodes = list(self.G.nodes())
            attempts = 0
            start_node, end_node = None, None
            
            while attempts < 10:
                start_node = random.choice(nodes)
                potential_ends = [n for n in nodes if n != start_node and not self.G.has_edge(start_node, n)]
                
                if potential_ends:
                    end_node = random.choice(potential_ends)
                    break
                attempts += 1
            
            if not end_node:
                start_node, end_node = random.sample(nodes, 2)
            
            try:
                backbone_path = nx.shortest_path(self.G, start_node, end_node)
            except nx.NetworkXNoPath:
                connected_nodes = list(nx.node_connected_component(self.G, start_node))
                if len(connected_nodes) >= 2:
                    end_node = random.choice([n for n in connected_nodes if n != start_node])
                    backbone_path = nx.shortest_path(self.G, start_node, end_node)
                else:
                    backbone_path = [start_node]
                    if len(connected_nodes) > 1:
                        backbone_path.append(random.choice([n for n in connected_nodes if n != start_node]))
            
            sampled_nodes = set(backbone_path)
            
            for backbone_node in backbone_path:
                neighbors = list(self.G.neighbors(backbone_node))
                available_neighbors = [n for n in neighbors if n not in sampled_nodes]
                
                num_to_add = min(
                    random.randint(1, 2),
                    len(available_neighbors),
                    self.order - len(sampled_nodes)
                )
                
                if num_to_add > 0:
                    selected_neighbors = random.sample(available_neighbors, num_to_add)
                    sampled_nodes.update(selected_neighbors)
                
                if len(sampled_nodes) >= self.order:
                    break
            
            sampled_node_ids = list(sampled_nodes)
            sampled_relations = self._get_sampled_relations(sampled_node_ids, relations=self.relations)
            
            output = SubgraphSamplingOutput(
                        node_ids=sampled_node_ids,
                        nodes={node: node_attr(self.G, node) for node in sampled_node_ids},
                        relations=sampled_relations,
                        subgraph_sample_algorithm='augmented_chain',
                        start_node=start_node,
                        subgraph_order=len(sampled_node_ids)
                    )
            
            logger.info(f"主干增强采样: 采样了 {len(sampled_node_ids)} 个节点和 {len(sampled_relations)} 个关系，主干路径长度: {len(backbone_path)}")
            return output
            
        except Exception as e:
            logger.error(f"主干增强采样失败: {e}, 回退到随机采样")
            return self._fallback_sampling()
    
    def _fallback_sampling(self) -> SubgraphSamplingOutput:
        """回退到随机采样"""
        return RandomSampler(self.G, self.order, **self.config).sample_subgraph()


class BFSSampler(SubGraphSampler):
    """广度优先搜索采样 (Breadth-First Sampling)
    
    从起始节点开始进行BFS探索，构建包含丰富邻居信息的子图
    """
    def __init__(self, G: nx.Graph, order: int, **kwargs):
        super().__init__(G, order, **kwargs)

    def sample_subgraph(self):
        try:
            nodes = list(self.G.nodes())
            start_node = random.choice(nodes)
            
            sampled_nodes = set()
            queue = [(start_node, 0)]
            visited = {start_node}
            
            while queue and len(sampled_nodes) < self.order:
                current_node, depth = queue.pop(0)
                sampled_nodes.add(current_node)
                
                if len(sampled_nodes) >= self.order:
                    break
                
                neighbors = list(self.G.neighbors(current_node))
                random.shuffle(neighbors)
                
                max_neighbors = min(3, len(neighbors))
                for neighbor in neighbors[:max_neighbors]:
                    if neighbor not in visited and len(sampled_nodes) < self.order:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
            
            if len(sampled_nodes) < self.order:
                remaining = self.order - len(sampled_nodes)
                available_nodes = [n for n in nodes if n not in sampled_nodes]
                if available_nodes and remaining > 0:
                    additional = random.sample(available_nodes, min(remaining, len(available_nodes)))
                    sampled_nodes.update(additional)
            
            sampled_node_ids = list(sampled_nodes)
            sampled_relations = self._get_sampled_relations(sampled_node_ids, relations=self.relations)
            
            output = SubgraphSamplingOutput(
                        node_ids=sampled_node_ids,
                        nodes={node: node_attr(self.G, node) for node in sampled_node_ids},
                        relations=sampled_relations,
                        subgraph_sample_algorithm='bfs',
                        start_node=start_node,
                        subgraph_order=len(sampled_node_ids)
                    )
            
            logger.info(f"BFS采样: 采样了 {len(sampled_node_ids)} 个节点和 {len(sampled_relations)} 个关系")
            return output
            
        except Exception as e:
            logger.error(f"BFS采样失败: {e}, 回退到随机采样")
            return self._fallback_sampling()
    
    def _fallback_sampling(self) -> SubgraphSamplingOutput:
        """回退到随机采样"""
        return RandomSampler(self.G, self.order, **self.config).sample_subgraph()



def main():
    graph_path = "./graph.graphml"
    graph_path = "/home/xxxx/xxxx/df_v1/output_dir/graph.graphml"
    G = load_nx_graphml(file_path=graph_path)
    all_relations = get_all_relations(G, format="json")
    print(len(all_relations))
    count = 0
    for rel in all_relations:
        if rel['tail'] == rel['head']:
            count += 1
    print(count)
    sampler = RandomSampler(G, 100)
    subgraph = sampler.sample_subgraph()
    print(subgraph["node_ids"][: 5])
    print([v["name"] for k, v in (subgraph["nodes"]).items()][: 5])
    print(subgraph["nodes"][subgraph.node_ids[0]])
    print(subgraph["relations"])



if __name__ == "__main__":
    main()



