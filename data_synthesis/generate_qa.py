"""
数据合成
"""
import os
import sys
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import logging
import networkx as nx
from typing import *
from dataclasses import dataclass
from abc import ABC, abstractmethod
from llms.client import *
from data_synthesis.subgraph_sampling import *
from data_synthesis.trace_generate import *
from data_synthesis.net_utils import *
from data_synthesis.information_blur import InformationBlur
from prompts.datasynthesis_prompt import MULTI_HOP_PROMPT, MULTI_HOP_TIF_PROMPT, MCQ_PROMPT
from prompts.task_prompt import task_prompt
from processor.node import NodeType

logger = logging.getLogger(__name__)

class QAGenerator:
    """QA数据生成"""
    def __init__(self, client: AsyncLLMClient, task_info: Dict, **kwargs):
        self.client = client
        self.config = kwargs
        self.question_need_info = task_info.get("question_need_info", {})
        self.task_info = task_info
        # for k, v in task_info.get("start_node_requirement", {}).items():
        #     if v > 0:
        #         self.question_need_info.append(k)

    def generate_qa(self, trace_output: TraceSelectOutput, task_type: str, is_blurry: bool = False):
        logger.info(f"[合成数据阶段] 开始合成数据，任务类型: {task_type}")
        results = []
        # 实体模糊化
        blur = InformationBlur()
        if is_blurry:
            trace_output = blur.blur(trace_output)
        # 构建上下文
        contexts = self._construct_path_context(trace_output)
        for context, trace in zip(contexts, trace_output.paths):
            if task_type == "multi_hop":
                prompt = MULTI_HOP_PROMPT.replace("{{{graph_context}}}", context)
            elif task_type == "multi_hop_tif":
                prompt = MULTI_HOP_TIF_PROMPT.replace("{{{graph_context}}}", context)
            elif task_type == "mcq":
                prompt = MCQ_PROMPT.replace("{{{graph_context}}}", context)
            else:
                pass
            messages = [  # 第一个请求
            # {"role": "system", "content": "你是合成数据专家"},
            {"role": "user", "content": prompt}
            ]
            # llm_response = get_local_llm(prompt)
            llm_response = self.client.generate(messages=messages, temperature=0.1, max_tokens=4096)
            content = llm_response['choices'][0]['message']['content'].strip()
            res = self._parse_qa_response(content, trace, task_type)
            results.append(res)
        return results
    
    async def generate_qa_batch(self, trace_output: TraceSelectOutput, task_type: str, is_blurry: bool = False):
        logger.info(f"[ 合成数据阶段] 开始合成数据，任务类型: {task_type}")
        results = []
         # 实体模糊化
        blur = InformationBlur()
        if is_blurry:
            trace_output = blur.blur(trace_output)
        contexts, queston_need_infos = self._construct_path_context(trace_output)
        # 准备批量请求 
        msgs_list = []
        traces = trace_output.paths
        for context, trace in zip(contexts, traces): 
            prompt_template = self.task_info.get("prompt", """请根据以下内容合成qa：{{{graph_context}}}
### 输出规范 
- 严格返回JSON格式，无任何额外文本（包括'```json'等标记）
- 字段要求：
  "question": 复杂的自然语言问题
  "answer": 简洁且具体唯一的答案
  "reasoning_path": 问题设计的推理过程，是一个CoT思考或推理过程，要求自然连贯
  """) # task_prompt.get(task_type, self.task_info.get("prompt", ))
            prompt = prompt_template.replace("{{{graph_context}}}", context)
            
            sys_prompt = "你是合成数据专家, 你需要根据提供的知识路径生成符合要求的问答数据。\n"
            # 生成问题的要求，过滤掉简单问题
            req = [
                "如果问题答案过于简单，如答案是无意义的词语或数字或者指代词，可以拒绝回答，返回一个句子并解释原因，如：'该问题的答案xx是无意义的词语或数字，无法回答。'",
            ]
            req = "\n".join(req)
            sys_prompt += req

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
            msgs_list.append(messages) 
        responses = await self.client.agenerate_batch( 
            messages_list=msgs_list,
            temperature=0.1,
            max_tokens=4096
        )
        # 处理响应
        for i, (response, trace) in enumerate(zip(responses, traces)):
            if isinstance(response, Exception):
                logger.error(f" 请求 {i+1} 错误: {str(response)}")
                continue
            else:
                content = response['choices'][0]['message']['content'].strip()
                res = self._parse_qa_response(content, queston_need_infos[i], trace, task_type)
                results.append(res) 
        
        return results 

    def _get_path_relation(self, trace: Trace):
        rels = []
        node_ids = trace.node_ids
        for i in range(len(node_ids) - 1):
            for rel in trace.relations:
                if node_ids[i] == rel['head'] and node_ids[i + 1] == rel['tail']:
                    rels.append({
                        'head': trace.nodes[node_ids[i]].get('name', ''),
                        'relation': rel['relation'],
                        'tail': trace.nodes[node_ids[i + 1]].get('name', ''),
                        'desc': rel['desc'],
                    })
        return rels

    def _parse_qa_response(self, response: str, queston_need_info: str, trace: Trace, task_type: str) -> Dict[str, str]:
        """解析问答响应"""
        try:
            cleaned_response = response.strip().replace('```json', '').replace('```', '')
            parsed = json.loads(cleaned_response)
            kgpath = []
            kgpath.append("实体序列: " + " -> ".join([node.get('name', '') for node in trace.nodes.values()]))
            kgpath.append("关系序列: " + '\n'.join(
                f'Relation({t["head"]}, {t["relation"]}, {t["tail"]}), Desc: {t["desc"]}' for t in self._get_path_relation(trace)))
            question = parsed.get('question', '')
            

            if len(queston_need_info) > 0:
                question = "请根据下列内容回答下列问题:\n" + queston_need_info + "\n" + question
            cot = parsed.get('reasoning_path', '')
            if isinstance(cot, list):
                cot = "\n".join(cot)
            return {
                'q': question,
                'a': parsed.get('answer', ''),
                'cot': cot,
                # 'kgpath': "\n".join(kgpath),
                'task_type': task_type,
                'node_types': list(set([node.get('type', 'unknown') for node in trace.nodes.values()])),
                'node_ids': trace.node_ids,
                'node_name': [node.get('name', '') for node in trace.nodes.values()],
                # 'relations': self._get_path_relation(trace),
                'evidence':{
                    'nodes': [{
                        'name': node.get('name', ''),
                        'type': node.get('type', 'unknown'),
                        'img_path': node.get('img_path', ''),
                        'content': node.get('content', ''),
                        'desc': node.get('desc', ''),
                        } for node in trace.nodes.values()],
                    'relations': self._get_path_relation(trace),
                }
            }
        except json.JSONDecodeError:
            logger.warning(f"无法解析JSON响应，尝试简单解析: {response[:200]}...")
            return {
                'question': '解析失败',
                'answer': '解析失败',
                'reasoning_path': '解析失败',
                'task_type': task_type,
                'node_ids': [],
                'relations': [],
                'response': response,
            }

    def _construct_path_context(self, trace_output: TraceSelectOutput) -> List[str]:
        """构建详细的知识图谱上下文"""
        contexts = []
        queston_need_infos = []
        traces = trace_output.paths
        for trace in traces:
            context_parts = ["### 路径信息"]
            context_parts.append("实体序列: " + " -> ".join([node.get('name', '') for node in trace.nodes.values() if node.get('type', 'unknown') == 'Entity']))
            context_parts.append("关系序列: " + '\n'.join(
                f'(Relation({t["head"]}, {t["relation"]}, {t["tail"]}), Desc: {t["desc"]})' for t in self._get_path_relation(trace)))

            queston_need_info_parts = []
            
            doc_list = []
            chunk_list = []
            assertion_list = []
            entity_list = []
            image_list = []
            table_list = []
            formula_list = []
            doc_i = 0
            chunk_i = 0
            assertion_i = 0
            image_i = 0
            table_i = 0
            formula_i = 0
            entity_i = 0
            for i, node in enumerate(trace.nodes.values(), 1):
                match node.get('type', 'unknown'):
                    case 'Document':
                        if doc_i == 0:
                            doc_list.append(f"### 文档信息")
                        doc_i += 1
                        name = node.get('name', f'文档{doc_i}')
                        content = node.get('content', '无内容')
                        doc_list.append(f"文档名：{name}")
                        doc_list.append(f"文档内容：{content}")
                        doc_list.append("\n")
                    case 'Chunk':
                        if chunk_i == 0:
                            chunk_list.append(f"### 文档块信息")
                        chunk_i += 1
                        name = node.get('name', f'块{chunk_i}')
                        content = node.get('content', '无内容')
                        chunk_list.append(f"块名：{name}")
                        chunk_list.append(f"块内容：{content}")
                        chunk_list.append("\n")
                    case 'Assertion':
                        if assertion_i == 0:
                            assertion_list.append(f"### 断言信息")
                        assertion_i += 1
                        name = node.get('name', f'断言{assertion_i}')
                        assertion_list.append(f"断言：{name}")
                        assertion_list.append(f"断言三元组：{node.get('head', '')}->{node.get('relation', '')}->{node.get('tail', '')}")
                        assertion_list.append("\n")
                    case 'Entity':
                        if entity_i == 0:
                            entity_list.append(f"### 实体信息")
                        entity_i += 1
                        name = node.get('name', f'实体{entity_i}')
                        node_type = node.get('type', 'unknown')
                        description = node.get('desc') or '无描述'
                        is_blurry = node.get('is_blurry', False)
                        if is_blurry:
                            blurry_name = node.get('blurry_name', '')
                            entity_list.append(f"{entity_i}. {blurry_name} (原名字: {name} 类型: {node_type}, 已模糊化)")
                        else:
                            entity_list.append(f"{entity_i}. {name} (类型: {node_type}, 未模糊化)")
                        if description != '无描述':
                            entity_list.append(f"描述: {description}")
                        attrs = {k: v for k, v in node.items() if k not in {'id', 'name', 'type', 'desc'}}
                        attrs = '\n'.join(f'{k}:{v}' for k, v in attrs.items())
                        entity_list.append(attrs)
                        entity_list.append(f"\n")
                    case 'Image':
                        if image_i == 0:
                            image_list.append(f"### 图片信息")
                        image_i += 1
                        name = node.get('name', f'图片{image_i}')
                        image_class = node.get('class', 'unknown')

                        image_list.append(f"图片名：{name}")
                        image_list.append(f"图片类别：{image_class}")
                        image_list.append(f"图片描述：{node.get('desc', '无描述')}")
                        image_list.append(f"图片内容：{node.get('content', '')}")
                        image_list.append("\n")
                    case 'Table':
                        if table_i == 0:
                            table_list.append(f"### 表格信息")
                        table_i += 1
                        name = node.get('name', f'表格{table_i}')
                        content = node.get('content', '无内容')
                        table_list.append(f"表格名：{name}")
                        table_list.append(f"表格内容：{content}")
                        table_list.append(f"表格描述：{node.get('desc', '无描述')}")
                        table_list.append("\n")
                    case 'Formula':
                        if formula_i == 0:
                            formula_list.append(f"### 公式信息")
                        formula_i += 1
                        name = node.get('name', f'公式{formula_i}')
                        formula_list.append(f"公式名：{name}")
                        formula_list.append(f"公式内容：{node.get('content', '')}")
                        formula_list.append(f"公式描述：{node.get('desc', '无描述')}")
                        formula_list.append(f"公式上下文：{node.get('context', '无上下文')}")
                        formula_list.append("\n")
                    case _:
                        print(f"未知节点类型: {node.get('type', 'unknown')}")
                    

            need_info_dict = {
                NodeType.Document: doc_list,
                NodeType.Chunk: chunk_list,
                NodeType.Assertion: assertion_list,
                NodeType.Entity: entity_list,
                NodeType.Image: image_list,
                NodeType.Table: table_list,
                NodeType.Formula: formula_list
            }
            for tp in self.question_need_info:
                queston_need_info_parts.extend(need_info_dict.get(tp, []))

            queston_need_infos.append("\n".join(queston_need_info_parts))

            context_parts.extend(doc_list)
            context_parts.extend(chunk_list)
            context_parts.extend(assertion_list)
            context_parts.extend(entity_list)
            context_parts.extend(image_list)
            context_parts.extend(table_list)
            context_parts.extend(formula_list)


            # 添加answer
            context_parts.append(f"### 答案信息")
            answer = f"请以{trace.gene_ans_node.get('name', 'unknown')}为答案进行问题生成。"
            context_parts.append(answer)

            # # 额外要求
            # context_parts.append(f"### 其它要求")
            # cot_require = "生成的思维链路推理中，请不要出现"

            # 拼接context
            context = "\n".join(context_parts)
            contexts.append(context)
        return contexts, queston_need_infos



async def main():
    # print(MULTI_HOP_PROMPT)
    # exit(0)
    import time
    random.seed(202511)
    graph_path = "/home/xxxx/xxxx/data_syn/v1/medical.graphml"
    G = load_nx_graphml(file_path=graph_path)
    sampler = DefaultSampler(G, 100)
    subgraph = sampler.sample_subgraph()
    print("Sampling Algorithm: ", subgraph.subgraph_sample_algorithm)
    selector = DFSSelector(G, sampling_output=subgraph)
    st = time.time()
    trace_output = selector.select_trace(max_steps=5,
                                  num_traces=4,
                                  min_deg=1,
                                  max_deg=200,
                                  is_neighbor=False)  # 找出边
    end = time.time()
    print("用时: ", end-st)                       
    path = trace_output.paths
    print(len(path))
    client = AsyncLLMClient("http://10.178.133.1:8000/v1", api_key="EMPTY", model="Qwen3-235B-A22B-Instruct-2507", max_concurrent_requests=5)
    generator = QAGenerator(client=client)
    # res = generator.generate_qa(trace_output=trace_output, task_type="multi_hop",)
    task_type = "multi_hop"
    res = await generator.generate_qa_batch(trace_output=trace_output, task_type=task_type,)
    for idx, r in enumerate(res):
        logger.info(f"生成第{idx+1}条路径: ")
        # path1 = path[idx]
        print(json.dumps(r, indent=4, ensure_ascii=False))



def test2():
    # print(MULTI_HOP_PROMPT)
    # exit(0)
    import time
    # random.seed(2025)
    graph_path = "/home/xxxx/xxxx/data_syn/v1/medical.graphml"
    G = load_nx_graphml(file_path=graph_path)
    # ===子图采样===
    print("===子图采样===")
    sampler = DefaultSampler(G, 100)  # 相当于选用整个知识图谱
    subgraph = sampler.sample_subgraph()
    print()
    print("Sampling Algorithm: ", subgraph.subgraph_sample_algorithm)
    selector = DFSSelector(G, sampling_output=subgraph)
    st = time.time()
    # ===路径选择===
    print("\n===路径选择===")
    trace_output = selector.select_trace(max_steps=5,
                                  num_traces=200,
                                  min_deg=1,
                                  max_deg=200,
                                  mode="out")  # 找出边
    end = time.time()
    print("用时: ", end-st)                       
    path = trace_output.paths
    print("一共搜索到路径数目: ", len(path))
    client = AsyncLLMClient("http://10.178.133.1:8000/v1", api_key="EMPTY", model="Qwen3-235B-A22B-Instruct-2507", max_concurrent_requests=5)
    print("加载Agent!")
    generator = QAGenerator(client=client)
    res = generator.generate_qa(trace_output=trace_output, task_type="multi_hop",)
    trace_output.paths = trace_output.paths[: 100]
    # for idx, r in enumerate(res):
    #     logger.info(f"生成第{idx+1}条路径: ")
    #     # path1 = path[idx]
    #     print(json.dumps(r, indent=4, ensure_ascii=False))
    new_data = []
    for idx, item in enumerate(res):
        new_item = {'id': str("exp_med_qwen3_235B_index_"+str(idx))}
        new_item.update(item)   # 把原字段追加到后面
        new_data.append(new_item)
    file_path = "/home/xxxx/xxxx/df_v2/qa_dir/exp5_财报/exp_med_qwen3_235B_1.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data,  f, ensure_ascii=False, indent=4)



async def test3():
    # print(MULTI_HOP_PROMPT)
    # exit(0)
    import time
    # 参数配置!!!
    mode = 'out'
    max_steps=3
    num_traces=20
    task_type = "multi_hop"  # "multi_hop", "mcq"
    random.seed(20251112)  # 随机数种子，便于复现
    graph_path = "/home/xxxx/xxxx/df_v2/output_dir/exp1_med/medical1.graphml"
    graph_path = '/home/xxxx/xxxx/df_v2/output_dir/exp5_财报/graph.graphml'
    # graph_path = '/home/xxxx/xxxx/df_v2/output_dir/exp6_结构化数据/graph.graphml'
    G = load_nx_graphml(file_path=graph_path)
    entity_types = get_node_types(G)  # list(set([G.nodes[n]['type'] for n in list(G.nodes())]))
    print("实体类型有: ", entity_types)
    # entity_types = ['Disease', 'Drug']  # ['Disease', 'Drug']  # 控制/选定节点类型
    entity_types = ['Image', 'Entity', 'Table', 'Formula']
    entity_types = ['Entity']
    # entity_types = ['Entity']
    # entity_types = ['Entity']
    sampler = DefaultSampler(G, 100)
    subgraph = sampler.sample_subgraph()
    print("Sampling Algorithm: ", subgraph.subgraph_sample_algorithm)
    selector = DFSSelector(G, sampling_output=subgraph, node_types=entity_types)
    st = time.time()
    trace_output = selector.select_trace(max_steps=max_steps,
                                  num_traces=num_traces,
                                  min_deg=0,  # 1
                                  max_deg=200,
                                  mode=mode)  # 找出边
    end = time.time()
    print("用时: ", end-st)                       
    path = trace_output.paths
    # trace_output.paths = trace_output.paths[: 2]
    # print(trace_output.paths)
    for p in path:
        print([p.nodes[id_]['name'] for id_ in p.node_ids])
    # exit(0)
    client = AsyncLLMClient("http://10.178.133.1:8000/v1", api_key="EMPTY", model="Qwen3-235B-A22B-Instruct-2507", max_concurrent_requests=5)
    generator = QAGenerator(client=client)
    # res = generator.generate_qa(trace_output=trace_output, task_type="multi_hop",)
    
    res = await generator.generate_qa_batch(trace_output=trace_output, task_type=task_type,)
    # for idx, r in enumerate(res):
    #     logger.info(f"生成第{idx+1}条路径: ")
    #     # path1 = path[idx]
    #     print("="*10+"QA"+"="*10+"\n")
    #     print(json.dumps(r, indent=4, ensure_ascii=False))
    with open("/home/xxxx/xxxx/df_v2/outputs/exp5_财报/exp5_qwen3_235B_1.json", 'w', encoding='utf-8') as f:
        json.dump(res,  f, ensure_ascii=False, indent=4)
    


if __name__ == "__main__":
    # test2()
    # asyncio.run(main())
    asyncio.run(test3())