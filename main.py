import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# from util.pdf2md import pdf2md
# from util.any2pdf import any2pdf

from processor.processor import Processor
from llms.client import AsyncLLMClient
from llms.vision_client import AsyncVisionClient

from data_synthesis.net_utils import *
from data_synthesis.subgraph_sampling import *
from data_synthesis.trace_generate import *
from data_synthesis.constants import SAMPLING_AlGORITHM_MAPPING, TRACE_GENERATION_MAPPING
from data_synthesis.generate_qa import *
import asyncio
import os
import json
from util.errors import *
from util.tool import stage_context
from processor.node import NodeType, node_type_list
from tqdm.asyncio import tqdm_asyncio
from data_synthesis.constants import *
from util.monitor import monitor_function
from util.export2std_data import convert_to_std_format



def run_processor(cfg: DictConfig) -> None:
    parse_dir = cfg.data.parse_dir
    input_dir = cfg.data.input_dir
    # 结构化数据
    if cfg.data.structured_data:
        processor = Processor(
            cfg=cfg,
            llm_func=None,
            vlm_func=None
        )
        processor.process_sd(cfg.data.input_dir, ",")
    else:
        # 非结构化数据
        # 0. any2pdf

        pdf_dir = any2pdf(input_dir=input_dir)
        # pdf_dir = input_dir
        # 1. 先提取pdf成基本chunk等
        if not os.path.exists(parse_dir):
            pdf2md(input_dir=pdf_dir, output_dir=parse_dir, server_url=cfg.dataprocessing.mineru.server_url)

        llm_client = AsyncLLMClient(api_key=cfg.dataprocessing.llm.api_key, base_url=cfg.dataprocessing.llm.base_url, model=cfg.dataprocessing.llm.model,
                                    max_concurrent_requests=cfg.dataprocessing.llm.max_concurrent_requests)
        vlm_client = AsyncVisionClient(api_key=cfg.dataprocessing.vlm.api_key, base_url=cfg.dataprocessing.vlm.base_url, model=cfg.dataprocessing.vlm.model,
                                    max_concurrent_requests=cfg.dataprocessing.vlm.max_concurrent_requests,
                                    max_tokens=cfg.dataprocessing.vlm.max_tokens)
        # @monitor_function("llm_call_log.json") # 监控llm调用,
        async def llm_func(prompt, system_prompt="", require_json=False, **kwargs):
            try:
                res = await llm_client.agenerate(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=cfg.dataprocessing.llm.model,
                    require_json=require_json,
                    **kwargs,
                )
                # print("input_len"+str(len(system_prompt+prompt))+", output_len"+str(len(res['choices'][0]['message']['content'])))
                return res['choices'][0]['message']['content'].strip()
            except LLMRetry_Error as e:
                return ""

        async def vlm_func(image_source, system_prompt="", prompt="", require_json=False, **kwargs):
            try:
                res = await vlm_client.agenerate(
                    system_prompt=system_prompt,
                    prompts=prompt,
                    image_source=image_source,
                    model=cfg.dataprocessing.vlm.model,
                    require_json=require_json,
                    **kwargs,
                )
                return res['choices'][0]['message']['content'].strip()
            except LLMRetry_Error as e:
                return ""

        processor = Processor(
            cfg=cfg,
            llm_func=llm_func,
            vlm_func=vlm_func
        )

        # 2. 处理数据,生成节点,生成边
        asyncio.run(processor.aprocess_ud())  # "./node_llm_out.json"

    processor.save_node()
    processor.save_edge()
    output_dir = cfg.data.output_dir
    node_list_path = os.path.join(output_dir, "node_list.json")
    edge_list_path = os.path.join(output_dir, "edge_list.json")
    processor.json2graph(node_list_path=node_list_path, edge_list_path=edge_list_path)
    if cfg.data.enable_visual:
        node_list_flat_path = os.path.join(output_dir, "node_list_flat.json")
        processor.visualize_kg(entities=node_list_flat_path, relations=edge_list_path, file_name=os.path.join(output_dir, "knowledge_entity_graph.html"), vis_node_types=[NodeType.Entity])
        processor.visualize_kg(entities=node_list_flat_path, relations=edge_list_path, file_name=os.path.join(output_dir, "knowledge_node_graph.html"), vis_node_types=node_type_list)

    

async def data_generate(cfg: DictConfig) -> None:
    print(cfg.data_synthesis.base_url)
    print(cfg.data_synthesis.api_key)
    print(cfg.data_synthesis.model)
    client = AsyncLLMClient(api_key=cfg.data_synthesis.api_key, base_url=cfg.data_synthesis.base_url,
                            model=cfg.data_synthesis.model,
                            max_concurrent_requests=cfg.data_synthesis.max_concurrent_requests)
    path = os.path.join(cfg.data.output_dir, "graph.graphml")
    subgraph_num = cfg.subgraph_sampling.subgraph_num
    # G = load_nx_graphml(path)
    G = load_nx_graphml(path)  # 作为测试
    # 1. 子图采样
    sampler_chosen = SAMPLING_AlGORITHM_MAPPING[cfg.subgraph_sampling.sampling_algorithm]
    sampler = sampler_chosen(G=G, order=cfg.subgraph_sampling.order, **cfg.subgraph_sampling.kwargs)  # kwargs会放在config中
    subgraphs = []
    for _ in range(subgraph_num):
        subgraphs.append(sampler.sample_subgraph())

    # 2. 路径生成
    task_list = []
    selector_chosen = TRACE_GENERATION_MAPPING[cfg.trace_generation.selection_method]
    # node_types = None or cfg.trace_generation.node_types
    if "all" in cfg.data_synthesis.task_type_list:
        task_type_list = TASK_INFO.keys()
    else:
        task_type_list = cfg.data_synthesis.task_type_list
    for i, subgraph in enumerate(subgraphs):
        # node_types=None or cfg.trace_generation.node_types
        for task_type in task_type_list:
            selector = selector_chosen(G=G, sampling_output=subgraph, node_types=cfg.trace_generation.node_types, **cfg.trace_generation.kwargs)
            traces = selector.select_trace(max_steps=cfg.trace_generation.max_steps,
                                        num_traces=cfg.trace_generation.num_traces,
                                        min_deg=cfg.trace_generation.min_deg,
                                        max_deg=cfg.trace_generation.max_deg,
                                        mode=cfg.trace_generation.mode,
                                        task_info=TASK_INFO[task_type])                  
            if len(traces.paths)==0: 
                logger.info(f"task_type: {task_type}, subgraph: {i}, no paths")
                continue
            else:
                logger.info(f"task_type: {task_type}, subgraph: {i}, paths_len: {len(traces.paths)}")
            # 3. 数据生成
            qa_generator = QAGenerator(client=client, task_info=TASK_INFO[task_type])
            task = asyncio.create_task(
                qa_generator.generate_qa_batch(trace_output=traces, task_type=task_type)
            )
            task_list.append(task)
    res = await tqdm_asyncio.gather(*task_list,desc="data generating")
    qa_list = []
    for r in res:
        qa_list.extend(r)
    
    data_gene_path = os.path.join(cfg.data.output_dir, "data_gene.json")
    with open(data_gene_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, indent=4, ensure_ascii=False)


from qafilter.enhanced_refactored_pipeline import RefactoredEnhancedEvaluationPipeline as qa_eval_pipeline

async def datafilter(cfg: DictConfig):
    pipeline = qa_eval_pipeline(cfg)
    output_dir = cfg.data.output_dir
    input_file = os.path.join(output_dir, "data_gene.json")
    output_file = os.path.join(output_dir, "data_filter.json")
    await pipeline.run_pipeline(
        input_file=input_file,
        output_file=output_file,
        batch_size=256
    )

from eval.eval_up_vl import vlm_test
from eval.eval_up import llm_test

llm_eval_model_config_list =[
    {
        "name": "Qwen3-0.6B",
        "api_key": "empty",
        "base_url": "http://10.178.141.71:8000/v1",
        "model": "qwen3-0.6B",
    },
    {
        "name": "Qwen3-1.7B",
        "api_key": "empty",
        "base_url": "http://10.178.141.71:8001/v1",
        "model": "qwen3-1.7B",
    },
    {
        "name": "Qwen3-4B",
        "api_key": "empty",
        "base_url": "http://10.178.141.71:8002/v1",
        "model": "qwen3-4B",
    },
    {
        "name": "Qwen3-8B",
        "api_key": "empty",
        "base_url": "http://10.178.129.210:8000/v1",
        "model": "qwen3-8B",
    },
    {
        "name": "Qwen3-14B",
        "api_key": "empty",
        "base_url": "http://10.178.131.34:8000/v1",
        "model": "qwen3-14B",
    },
    {
        "name": "Qwen3-32B",
        "api_key": "empty",
        "base_url": "http://10.178.141.71:8003/v1",
        "model": "qwen3-32B",
    },
    {
        "name": "Qwen3-30A3B",
        "api_key": "empty",
        "base_url": "http://10.178.131.44:8000/v1",
        "model": "qwen3-30A3B",
    },
    {
        "name": "Qwen3-235B-A22B-Instruct-2507",
        "api_key": "empty",
        "base_url": "http://10.178.133.1:8000/v1",
        "model": "Qwen3-235B-A22B-Instruct-2507",
    },
    
]


vlm_eval_model_config_list =[
    {
        "name": "Qwen3-VL-2B",
        "api_key": "empty",
        "base_url": "http://10.178.142.6:9000/v1",
        "model": "qwen3-vl-2B",
    },
    {
        "name": "Qwen3-VL-4B",
        "api_key": "empty",
        "base_url": "http://10.178.142.6:9001/v1",
        "model": "qwen3-vl-4B",
    },
    {
        "name": "Qwen3-VL-8B",
        "api_key": "empty",
        "base_url": "http://10.178.142.6:9002/v1",
        "model": "qwen3-vl-8B",
    },
    {
        "name": "Qwen3-VL-32B",
        "api_key": "empty",
        "base_url": "http://10.178.142.6:9003/v1",
        "model": "qwen3-vl-32B",
    },
    {
        "name": "Qwen3-VL-30BA3B",
        "api_key": "empty",
        "base_url": "http://10.178.131.44:9000/v1",
        "model": "qwen3-vl-30BA3B",
    },
    {
        "name": "Qwen3-VL-235A22B",
        "api_key": "empty",
        "base_url": "http://10.178.141.71:9000/v1",
        "model": "qwen3-vl-235A22B",
    },
    
]

@hydra.main(config_path="config", config_name="dev", version_base=None)
def main(cfg: DictConfig) -> None:
    # # 1. 数据处理构建图谱
    # with stage_context("数据处理构建图谱", 1):
    #     run_processor(cfg)
    # 2. 数据生成
    with stage_context("数据生成", 2):
        asyncio.run(data_generate(cfg))
    # 3. 数据过滤
    with stage_context("数据过滤", 3):
        asyncio.run(datafilter(cfg))
    # 4. 数据转换为标准格式
    with stage_context("数据转换为标准格式", 4):
        output_dir = cfg.data.output_dir
        input_file = os.path.join(output_dir, "data_filter.json")
        output_file = os.path.join(output_dir, "data_filter_std.json")
        convert_to_std_format(input_file, output_file, mode='sft', format="messages")

    with stage_context("评估", 5):
        # 5. 评估
        llm_args = {
            'input': os.path.join(output_dir, "data_filter_std.json"),
            'output': os.path.join(output_dir, "./results/llm/evaluation_results"),
            'model_configs': llm_eval_model_config_list,
        }
        llm_test(llm_args)

        vlm_args = {
            'input': os.path.join(output_dir, "data_filter_std.json"),
            'output': os.path.join(output_dir, "./results/vlm/evaluation_results"),
            'model_configs': vlm_eval_model_config_list,
        }
        vlm_test(vlm_args)


if __name__ == "__main__":
    main()
    
