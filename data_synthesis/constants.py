"""
定义基本数据生成配置相关映射
"""

from data_synthesis.subgraph_sampling import *
from data_synthesis.trace_generate import *
from prompts.task_prompt import *
from prompts.datasynthesis_prompt import *


SAMPLING_AlGORITHM_MAPPING = {
    "no_subgraph_sampling": DefaultSampler,  # 不进行子图采样，选择整个图谱
    "random": RandomSampler,  # 随机选择节点
    'acs': AugmentedChainSampler,  # 基于ACS的子图采样
    'bfs': BFSSampler,  # 基于BFS的子图采样
}

TRACE_GENERATION_MAPPING= {
    "dfs": DFSSelector,  # 深度遍历优先
    # "tif": TIFDFSSelector,  # 多表推理，暂时还没写完
}


# TASK_INFO = {
#     "multi_hop_1": {
#         "task_type": "multi_hop_1",
#         # 任务prompt
#         "prompt": task_prompt["QA"],
#         # 最大跳数
#         "max_steps": 1,
#         # 负数表示取相连某类数大于等于（其绝对值-1）的作为起始节点，正数代表取与k个该类相连的的起始节点（并将k个该类节点信息加入qa），
#         # 0代表起始节点不做要求,mode为in第一节点为entity的话之后将全为entity,因为单向且当前只有实体能指向实体
#         "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 0},
#         # 对节点大类中的更细节的要求，key为大类，value为字典（type字段指定要求检查的字段, require字段指定其范围）
#         # "extra_requirements": {
#         #     'Image': {
#         #         "type": "class",
#         #         "require": ['数值图'], # ['流程图', '数值图', '思维导图', '其他图']
#         #     }
#         # },
#     }
# }
TASK_INFO = {
    "Entity": {
        "task_type": "Entity",
        # 任务prompt
        "prompt": task_prompt["QA"],
        # 最大跳数
        "max_steps": 4,
        # 负数表示取相连某类数大于等于（其绝对值-1）的作为起始节点，正数代表取与k个该类相连的的起始节点（并将k个该类节点信息加入qa），
        # 0代表起始节点不做要求,mode为in第一节点为entity的话之后将全为entity,因为单向且当前只有实体能指向实体
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 0},
        # 对节点大类中的更细节的要求，key为大类，value为字典（type字段指定要求检查的字段, require字段指定其范围）
        "extra_requirements": {
            'Image': {
                "type": "class",
                "require": ['数值图'], # ['流程图', '数值图', '思维导图', '其他图']
            }
        },
        # 答案节点的类型，即对start_node节点的类型要求,要求在其中
        "ans_types": ["Entity"],
        # 最终生成的qa中问题需要哪些类别的节点信息
        "question_need_info": [],
    },
    "single_table": {
        "task_type": "single_table",
        "prompt": task_prompt["single_table"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 1, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_table_mutihop": {
        "task_type": "single_table_mutihop",
        "prompt": task_prompt["single_table_mutihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 1, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_table_single_text": {
        "task_type": "single_table_single_text",
        "prompt": task_prompt["single_table_single_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 1, 'Assertion': 0, 'Entity': 0, 'Table': 1, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_table_multi_text_qa": {
        "task_type": "single_table_multi_text_qa",
        "prompt": task_prompt["single_table_multi_text_qa"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 2, 'Assertion': 0, 'Entity': 0, 'Table': 1, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_table": {
        "task_type": "multi_table",
        "prompt": task_prompt["multi_table"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 2, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_table_multihop": {
        "task_type": "multi_table_multihop",
        "prompt": task_prompt["multi_table_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 2, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_table_multi_text": {
        "task_type": "multi_table_multi_text",
        "prompt": task_prompt["multi_table_multi_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 2, 'Assertion': 0, 'Entity': 0, 'Table': 2, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_text": {
        "task_type": "single_text",
        "prompt": task_prompt["single_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 1, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_text": {
        "task_type": "multi_text",
        "prompt": task_prompt["multi_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 2, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_text_multihop": {
        "task_type": "single_text_multihop",
        "prompt": task_prompt["single_text_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 1, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_text_multihop": {
        "task_type": "multi_text_multihop",
        "prompt": task_prompt["multi_text_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 2, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_formula": {
        "task_type": "single_formula",
        "prompt": task_prompt["single_formula"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 1},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_formula_multihop": {
        "task_type": "single_formula_multihop",
        "prompt": task_prompt["single_formula_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 1},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_formula_single_text": {
        "task_type": "single_formula_single_text",
        "prompt": task_prompt["single_formula_single_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 1, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 1},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_formula_multi_text": {
        "task_type": "single_formula_multi_text",
        "prompt": task_prompt["single_formula_multi_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 2, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 1},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_formula": {
        "task_type": "multi_formula",
        "prompt": task_prompt["multi_formula"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 2},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_formula_multihop": {
        "task_type": "multi_formula_multihop",
        "prompt": task_prompt["multi_formula_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 2},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_formula_multi_text": {
        "task_type": "multi_formula_multi_text",
        "prompt": task_prompt["multi_formula_multi_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 2, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 0, 'Formula': 2},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_chart": {
        "task_type": "single_chart",
        "prompt": task_prompt["single_chart"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 1, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_chart_multihop": {
        "task_type": "single_chart_multihop",
        "prompt": task_prompt["single_chart_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 1, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_chart_single_text": {
        "task_type": "single_chart_single_text",
        "prompt": task_prompt["single_chart_single_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 1, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 1, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_chart_multi_text": {
        "task_type": "single_chart_multi_text",
        "prompt": task_prompt["single_chart_multi_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 2, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 1, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_chart": {
        "task_type": "multi_chart",
        "prompt": task_prompt["multi_chart"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 2, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_chart_multihop": {
        "task_type": "multi_chart_multihop",
        "prompt": task_prompt["multi_chart_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 2, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "multi_chart_multi_text": {
        "task_type": "multi_chart_multi_text",
        "prompt": task_prompt["multi_chart_multi_text"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 2, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 2, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_table_single_chart": {
        "task_type": "single_table_single_chart",
        "prompt": task_prompt["single_table_single_chart"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 1, 'Image': 1, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_table_single_chart_multihop": {
        "task_type": "single_table_single_chart_multihop",
        "prompt": task_prompt["single_table_single_chart_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 1, 'Image': 1, 'Formula': 0},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_table_single_formula": {
        "task_type": "single_table_single_formula",
        "prompt": task_prompt["single_table_single_formula"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 1, 'Image': 0, 'Formula': 1},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_table_single_formula_multihop": {
        "task_type": "single_table_single_formula_multihop",
        "prompt": task_prompt["single_table_single_formula_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 1, 'Image': 0, 'Formula': 1},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_chart_single_formula": {
        "task_type": "single_chart_single_formula",
        "prompt": task_prompt["single_chart_single_formula"],
        "max_steps": 1,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 1, 'Formula': 1},
        "ans_types": ["Entity"],
        "question_need_info": [],
    },
    "single_chart_single_formula_multihop": {
        "task_type": "single_chart_single_formula_multihop",
        "prompt": task_prompt["single_chart_single_formula_multihop"],
        "max_steps": 4,
        "start_node_requirement": {'Document': 0, 'Chunk': 0, 'Assertion': 0, 'Entity': 0, 'Table': 0, 'Image': 1, 'Formula': 1}
    }

}