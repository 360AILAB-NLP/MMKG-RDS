"""
路径选择
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import networkx as nx
from typing import *
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC, abstractmethod
from data_synthesis.subgraph_sampling import *
from collections import Counter
from prompts.task_prompt import *
from processor.node import node_type_list


class TraceSelector(ABC):
    """路径选择器"""
    def __init__(self, G: nx.DiGraph, sampling_output: SubgraphSamplingOutput, node_types: List[str]=None, **kwargs):
        """
        Args:
            G: 知识图谱NX有向图
            sampling_output: 子图采样输出容器
            node_types: 节点类型列表
        """
        self.G = G
        self.sampling_output = sampling_output
        # assert set(node_types).issubset(set(get_node_types(self.G))), "出现异常节点类型!!!"
        self.node_types = node_types   # if node_types else get_node_types(self.G)
        self.config = kwargs
        assert self.sampling_output.node_ids,  "子图采样结果的节点ID列表为空!!!"

    @abstractmethod
    def select_trace(self):
        raise NotImplementedError

    def _get_sampled_relations(self, sampled_nodes: List[str], relations: List[Dict]) -> List[Dict]:
        """获取路径上节点之间的关系"""
        
        sampled_relations = []
        for relation in relations:
            head = relation.get('head', '')
            tail = relation.get('tail', '')
            if head in sampled_nodes and tail in sampled_nodes:
                sampled_relations.append(relation)
        return sampled_relations

    def _find_next_candidate(self, neighbors: List[str], visited: Set[str]) -> Optional[str]:
        filtered = [n for n in neighbors if n not in visited]
        return random.choice(filtered) if filtered else None

    def _get_subgraph_neighbors(self, node_id: str, mode: str = "out", extra_requirements: Dict[str, any] = None) -> int:
        """
        计算子图中某节点的邻节点和度（出度、入度或总度）, 同时是基于所要求的节点类型进行过滤。
        Args:
            node_id: 节点ID
            mode: 选择节点类型,如出边节点、入边节点、邻节点,'out' | 'in' | 'all'
        """
        relations = self.sampling_output.relations
        if self.sampling_output.subgraph_sample_algorithm=='no_subgraph_sampling':
            if mode == "out":
                neighbors = list(self.G.successors(node_id))
            elif mode == "in":
                neighbors = list(self.G.predecessors(node_id))
            elif mode == "all":
                neighbors = list(set(self.G.successors(node_id)) | set(self.G.predecessors(node_id)))
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            if mode == "out":
                neighbors = [r["tail"] for r in relations if r["head"] == node_id]
            elif mode == "in":
                neighbors = [r["head"] for r in relations if r["tail"] == node_id]
            elif mode == "all":
                out_nodes = [r["tail"] for r in relations if r["head"] == node_id]
                in_nodes  = [r["head"] for r in relations if r["tail"] == node_id]
                neighbors = out_nodes + in_nodes
            else:
                raise ValueError(f"Invalid mode: {mode}")
        # 只保留所要求类型的节点
        if self.node_types and  isinstance(self.node_types, list):   # and self.node_types!=get_node_types(self.G):  
            neighbors = [n for n in neighbors if self.sampling_output.nodes[n].get("type", "unknown") in self.node_types]
        
        # 过滤掉不符合的节点
        if extra_requirements:  # 筛选出满足额外要求的节点
            for ntype, require in extra_requirements.items():
                for n in neighbors: # n为id
                    node_info = self.sampling_output.nodes[n]
                    if node_info.get("type", "unknown") == ntype:  # 类型定位
                        if require:
                            require_k = require.get("type", None) 
                            require_vlist = require.get("require", None) # 要求字段值在其中
                            if require_k:
                                if node_info.get(require_k, None) in require_vlist:  # 要求字段值在其中
                                    continue
                                else:
                                    neighbors.remove(n)
        # neighbors = list(set(neighbors))
        return neighbors
        
    def _select_start_nodes(
        self,
        mode: Literal["in", "out", "all"] = "all",
        alg: str = "high",
        start_node_requirement: Dict[str, int] = None,
        extra_requirements: Dict[str, any] = None,
    ) -> List[str]:
        connect_type_cnt = {
            id: Counter([self.sampling_output.nodes[n].get('type') for n in self._get_subgraph_neighbors(id, mode="all", extra_requirements=extra_requirements)])
            for id in self.sampling_output.node_ids
        }
        start_node_requirement = Counter(start_node_requirement)

        if start_node_requirement:  # 筛选出满足要求的起始节点
            candidates = [
                id
                for id in self.sampling_output.node_ids
                if connect_type_cnt[id] >= start_node_requirement and self.sampling_output.nodes[id].get("type", "") in self.ans_types
            ]
            
            return random.sample(candidates, min(10, len(candidates)))

        node_degrees = [
            (n, len(self._get_subgraph_neighbors(n, mode=mode)))
            for n in self.sampling_output.node_ids
        ]
        node_degrees.sort(key=lambda x: x[1], reverse=True)

        if alg == "high":
            candidates = node_degrees[: len(node_degrees) // 4]
        elif alg == "medium":
            mid_start = len(node_degrees) // 4
            mid_end = 3 * len(node_degrees) // 4
            candidates = node_degrees[mid_start:mid_end] or node_degrees
        else:
            candidates = node_degrees

        ids = [nid for nid, _ in candidates]
        if not ids:
            ids = self.sampling_output.node_ids

        if self.node_types and isinstance(self.node_types, list):
            ids = [
                nid
                for nid in ids
                if self.sampling_output.nodes[nid].get("type", "") in self.node_types
            ]
        return random.sample(ids, min(10, len(ids)))

@dataclass
class Trace:
    node_ids: List[str]          # 这条路径的节点序列
    nodes: Dict                  # 采样的实体节点属性列表，key是node id，value是node attr
    gene_ans_node: str
    task_type: str
    relations: List[dict]        # 这条路径上出现的关系（局部视图）


    def __post_init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)
        
        # self.set_neighbors(neighbor_nodes=neighbor_nodes)

    # def set_neighbors(self, neighbor_nodes):
    #     for neighbor in neighbor_nodes:
    #         self[neighbor.get('type')].append(neighbor)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({items})"

    def __getitem__(self, key):
        """允许通过 obj[key] 访问属性"""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"属性 '{key}' 不存在")
    
    # def choice_neighbor(self, start_node_requirement):
    #     pass
        # return random.choice(neighbors)
        


@dataclass
class TraceSelectOutput:
    """
    路径选择策略的返回结果容器。
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
    paths : list[Trace]
        路径列表
    max_steps : int
        最大跳数
    """
    node_ids: List[str]  
    nodes: Dict  
    relations: List[dict]  
    subgraph_sample_algorithm: str 
    start_node: str  
    subgraph_order: int  
    paths: List[Trace]         
    max_steps: int            

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


class DFSSelector(TraceSelector):
    """基于 DFS 的路径选择器（带类型控制、完整回溯与放宽机制）"""
    def __init__(self, G: nx.DiGraph, sampling_output: SubgraphSamplingOutput, node_types: List[str]=None, **kwargs):
        super().__init__(G, sampling_output, node_types, **kwargs)

    def select_trace(
        self,
        max_steps: int = 5,
        num_traces: int = 1,
        min_deg: int = 4,
        max_deg: int = 8,
        mode: Literal["in", "out", "all"] = "out",
        task_info: Dict[str, Any] = {},
        # start_node_requirement: Dict[str, int] = None,
    ) -> TraceSelectOutput:
        """
        执行 DFS 路径选择
        Args:
            max_steps : 单条路径最大跳数（节点数 = max_steps + 1）
            num_traces : 需要采样的路径条数
            min_deg / max_deg : 对“下一跳”节点出度范围的过滤（仍用 out 度）
            mode : 与 _get_subgraph_neighbors 对齐，决定候选邻居集合
                'out'  → 仅用出边邻居
                'in'   → 仅用入边邻居
                'all'  → 出入边邻居并集
        Returns:
            TraceSelectOutput类型结果容器
        """
        paths: List[Trace] = []
        max_steps = task_info.get("max_steps", max_steps)
        self.ans_types = task_info.get("ans_types", ["Entity"])

        # ---------------- 局部函数 ----------------
        def _get_candidates(node: str, visited: Set[str]) -> List[str]:
            """
            根据 mode 拿到候选邻居，再做度过滤/排序/扰动
            """
            # 1. 按 mode 取邻居
            neighbors = self._get_subgraph_neighbors(node, mode=mode)
            # 2. 去掉已访问
            neighbors = [n for n in neighbors if n not in visited]
            # 3. 用出度做范围过滤（保持与你原逻辑一致）
            neighbors = [
                n
                for n in neighbors
                if min_deg <= len(self._get_subgraph_neighbors(n, mode=mode)) <= max_deg  # len(self._get_subgraph_neighbors(n, mode="out"))
            ]

            # 4. 高出度优先 + 随机打乱
            neighbors.sort(
                key=lambda x: len(self._get_subgraph_neighbors(x, mode=mode)),
                reverse=True,
            )
            top_candidates = neighbors[:20]
            random.shuffle(top_candidates)
            return top_candidates
        def _dfs(start: str) -> List[str]:
            """带回溯的 DFS"""
            visited = {start}
            best_path = [start]
            stack = [(start, [start], _get_candidates(start, visited))]
            while stack:
                node, path, candidates = stack[-1]

                if len(path) >= max_steps + 1:         
                    return path[::-1] if mode == "in" else path
                if not candidates:  # 回溯：当前节点没有候选邻居，则回溯到上一个节点                    
                    stack.pop()
                    continue

                nxt = candidates.pop(0)
                if nxt in visited:
                    continue
                visited.add(nxt)
                new_path = path + [nxt]
                if len(new_path) > len(best_path):
                    best_path = new_path
                next_candidates = _get_candidates(nxt, visited)
                random.shuffle(next_candidates)
                stack.append((nxt, new_path, next_candidates))
            return best_path[::-1] if mode == "in" else best_path

        start_node_requirement = task_info.get("start_node_requirement", None)
        extra_requirements = task_info.get("extra_requirements", None)
        path_ids_set = set()
        start_node_candidates = self._select_start_nodes(mode=mode, start_node_requirement=start_node_requirement, extra_requirements=extra_requirements)
        for _ in tqdm(range(num_traces), desc="选择路径中: "):
            try:
                start = random.choice(
                    start_node_candidates
                ) # or self.sampling_output.start_node
                neighbors = self._get_subgraph_neighbors(start, mode="all", extra_requirements=extra_requirements)
                # 抽取start邻居中随机一个作为起始点
                neighbor_nodes = [self.sampling_output.nodes[n] for n in neighbors]
            except Exception:
                continue
            # 随机从neighbors中抽取每个类型数量要求为start_node_requirement的节点
            neighbor_type2ids = {
                tp: [n.get("id") for n in neighbor_nodes if n.get('type') == tp] for tp in node_type_list
            }
            start_connected_ids = []
            for k, v in start_node_requirement.items():
                if v >= 0:
                    start_connected_ids.extend(random.sample(neighbor_type2ids[k], v))
                else: # 随机选取,
                    v = abs(v) - 1
                    neighbor = neighbor_type2ids[k]
                    sample_num = random.randint(v, len(neighbor))
                    start_connected_ids.extend(random.sample(neighbor, sample_num))
            
            path_ids = _dfs(start)
            if '->'.join(path_ids) in path_ids_set: continue  # 路径去重
            path_ids = start_connected_ids + path_ids
            trace = Trace(
                node_ids=path_ids,
                nodes={i: self.sampling_output.nodes[i] for i in path_ids},
                relations=self._get_sampled_relations(
                    path_ids, self.sampling_output.relations
                ),
                # neighbor_nodes=neighbor_nodes,
                gene_ans_node=self.sampling_output.nodes[start],
                task_type=task_info.get("task_type", "multi_hop")
            )
            
            paths.append(trace)
            path_ids_set.add('->'.join(path_ids))
        
        return TraceSelectOutput(
            node_ids=self.sampling_output.node_ids,
            nodes=self.sampling_output.nodes,
            relations=self.sampling_output.relations,
            subgraph_sample_algorithm=self.sampling_output.subgraph_sample_algorithm,
            start_node=self.sampling_output.start_node,
            subgraph_order=self.sampling_output.subgraph_order,
            paths=paths,
            max_steps=max_steps,
        )

    # def _select_start_nodes(
    #     self,
    #     mode: Literal["in", "out", "all"] = "all",
    #     alg: str = "high",
    #     start_node_requirement: Dict[str, str] = None,
    # ) -> List[str]:
    #     targets = {'Table', 'Image', 'Formula'} & set(self.node_types)
    #     if targets:
    #         cand = [n for n in self.sampling_output.node_ids if self.sampling_output.nodes[n].get('type') in targets]
    #         # print(cand)
    #         return random.sample(cand, min(10, len(cand)))
    #     node_degrees = [
    #         (n, len(self._get_subgraph_neighbors(n, mode=mode)))
    #         for n in self.sampling_output.node_ids
    #     ]
    #     node_degrees.sort(key=lambda x: x[1], reverse=True)

    #     if alg == "high":
    #         candidates = node_degrees[: len(node_degrees) // 4]
    #     elif alg == "medium":
    #         mid_start = len(node_degrees) // 4
    #         mid_end = 3 * len(node_degrees) // 4
    #         candidates = node_degrees[mid_start:mid_end] or node_degrees
    #     else:
    #         candidates = node_degrees

    #     ids = [nid for nid, _ in candidates]
    #     if not ids:
    #         ids = self.sampling_output.node_ids

    #     if self.node_types and isinstance(self.node_types, list):
    #         ids = [
    #             nid
    #             for nid in ids
    #             if self.sampling_output.nodes[nid].get("type", "") in self.node_types
    #         ]
    #     return random.sample(ids, min(10, len(ids)))


### 这个是多表/图/公式的类，还没写完
class TIFDFSSelector(TraceSelector):
    """
    基于 Table-Image-Formula 社群节点的多步推理路径选择器。
    逻辑：
      1. 先选两个高入度社群节点（Image/Table/Formula）
      2. 找到两者的共同连接实体（bridge）
      3. 通过 DFS 从两侧延展形成多跳路径
    """

    def __init__(self, G: nx.DiGraph, sampling_output: SubgraphSamplingOutput, node_types: List[str] = None, **kwargs):
        super().__init__(G, sampling_output, node_types, **kwargs)

    def select_trace(
        self,
        max_steps: int = 6,
        num_traces: int = 3,
        mode: Literal["in", "out", "all"] = "all",
        community_types: List[str] = ["Image", "Table", "Formula"],
    ) -> TraceSelectOutput:

        paths = []
        path_ids_set = set()

        # Step 1️⃣：找出所有社群节点（类型为 Image/Table/Formula）
        community_nodes = [
            n for n in self.sampling_output.node_ids
            if self.sampling_output.nodes[n].get("type") in community_types
        ]
        if not community_nodes:
            print("❗未找到任何社群节点（Image/Table/Formula）")
            return TraceSelectOutput(paths=[], max_steps=max_steps, **vars(self.sampling_output))

        # Step 2️⃣：根据入度排序（入度大的往往是社群中心节点）
        indeg = sorted([(n, self.G.in_degree(n)) for n in community_nodes],
                       key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, _ in indeg[:min(10, len(indeg))]]

        # Step 3️⃣：多次采样不同组合
        for _ in tqdm(range(num_traces), desc="选择TIF路径中: "):
            if len(top_nodes) < 2:
                break

            n1, n2 = random.sample(top_nodes, 2)

            # Step 4️⃣：找出两者共同连接的节点（桥接节点）
            nbrs1 = set(self._get_subgraph_neighbors(n1, mode=mode))
            nbrs2 = set(self._get_subgraph_neighbors(n2, mode=mode))
            common = list(nbrs1 & nbrs2)

            if not common:
                continue

            bridge = random.choice(common)

            # Step 5️⃣：分别从两侧做部分DFS，再拼接路径
            path1 = self._dfs_partial(n1, bridge, max_steps=max_steps // 2, mode=mode)
            path2 = self._dfs_partial(bridge, n2, max_steps=max_steps // 2, mode=mode)

            # 拼接成完整路径（去重bridge）
            full_path = path1 + path2[1:]
            if len(full_path) < 3:
                continue

            pid = "->".join(full_path)
            if pid in path_ids_set:
                continue

            trace = Trace(
                node_ids=full_path,
                nodes={i: self.sampling_output.nodes[i] for i in full_path},
                relations=self._get_sampled_relations(full_path, self.sampling_output.relations),
            )
            paths.append(trace)
            path_ids_set.add(pid)

        return TraceSelectOutput(
            node_ids=self.sampling_output.node_ids,
            nodes=self.sampling_output.nodes,
            relations=self.sampling_output.relations,
            subgraph_sample_algorithm=self.sampling_output.subgraph_sample_algorithm,
            start_node=None,
            subgraph_order=self.sampling_output.subgraph_order,
            paths=paths,
            max_steps=max_steps,
        )

    def _dfs_partial(self, start: str, target: str, max_steps: int, mode="out") -> List[str]:
        visited = {start}
        stack = [(start, [start], self._get_subgraph_neighbors(start, mode))]
        best_path = [start]

        while stack:
            node, path, cand = stack[-1]

            # 到达目标节点
            if node == target:
                return path

            if len(path) >= max_steps:
                stack.pop()
                continue

            if not cand:
                stack.pop()
                continue

            nxt = cand.pop()
            if nxt in visited:
                continue
            visited.add(nxt)
            stack.append((nxt, path + [nxt], self._get_subgraph_neighbors(nxt, mode)))

            if len(path) > len(best_path):
                best_path = path

        return best_path



def main():
    import time
    graph_path = "./graph.graphml"
    G = load_nx_graphml(file_path=graph_path)
    sampler = DefaultSampler(G, 100)
    subgraph = sampler.sample_subgraph()
    print("Sampling Algorithm: ", subgraph.subgraph_sample_algorithm)
    selector = DFSSelector(G, sampling_output=subgraph)
    st = time.time()
    trace_output = selector.select_trace(max_steps=5,
                                  num_traces=2,
                                  min_deg=1,
                                  max_deg=200,
                                  mode="out")  # 找出边
    end = time.time()
    print("用时: ", end-st)                       
    path = trace_output.paths
    print(len(path))
    if path:
        node_ids = path[0].node_ids
        print(node_ids)
        node_ids1 = path[1].node_ids
        print(node_ids1)
        names = [path[0].nodes[id_]['name'] for id_ in path[0].node_ids]
        print(names)
        relations = path[0].relations
        print("关系数: ", len(relations))
        for i in range(len(node_ids)-1):
            for rel in relations:
                if node_ids[i] == rel['head'] and node_ids[i+1] == rel['tail']:
                    print(path[0].nodes[node_ids[i]]['name'], rel["relation"], path[0].nodes[node_ids[i+1]]['name'])
        # print(path[0].nodes)


        
if __name__ == "__main__":
    main()