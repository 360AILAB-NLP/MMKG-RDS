"""
Prompt templates for multimodal content processing

Contains all prompt templates used in modal processors for analyzing
different types of content (images, tables, equations, etc.)
"""

# from __future__ import annotations
from typing import Any
# from langdetect import detect

# def get_language(text):
#     """Detect input language and return appropriate language code"""
#     try:
#         lang = detect(text[:500]) 
#         return "zh-CN" if lang == "zh" else "en-US"
#     except:
#         return "en-US" 


PROMPTS: dict[str, Any] = {}

PROMPTS["entity_standard_sys"] = (
"""你是一个实体标准化助手，善于将相同或相似的实体名归纳成一个统一标准的实体，并将归纳的实体的描述进行精准重点的总结,并以json格式输出，输入将会是一组实体，请将实体表意相同的合并为一个并标准化，最后以列表形式json输出"""
)

PROMPTS["entity_standard_user"] = (
"""请将输入的实体进行标准化处理，即对输入列表中表意相同实体以一个名称进行统一，并返回标准名称，以及合并后的描述。
    - 输入实体信息：为一个列表其中每个元素为一个实体，name字段为实体名，desc字段为实体的描述
    - 输出格式要求：
    [
        {{
        "name": "标准化的统一名称1",
        "alias": ["标准化的实体1的别名1", "标准化的实体1的别名2"],
        "desc": "实体描述, 压缩所有实体描述至100字以内"
        }},
        {{
        "name": "标准化的统一名称2",
        "alias": ["标准化的实体2的别名1", "标准化的实体2的别名2"],
        "desc": "实体描述, 压缩所有实体描述至100字以内"
        }},
        ...
    ]
---Example---
<Input>
[
    {{
    "name": "北京",
    "desc": "北京是中国的首都"
    }},
    {{
    "name": "中国首都",
    "desc": "中国的首都是北京"
    }},
    {{
    "name": "南京",
    "desc": "南京市是江苏省的省会"
    }},
    {{
    "name": "金陵",
    "desc": "金陵是南京的古称"
    }}
]
<Output>
[   
    {{
    "name": "北京",
    "alias": ["北京", "中国首都"],
    "desc": "北京是中国的首都"
    }},
    {{
    "name": "南京",
    "alias": ["南京", "金陵"],
    "desc": "南京市是江苏省的省会，古时称为金陵"
    }},
]

---真实数据---
<Input>
{entity_info}
<Output>
"""
)

PROMPTS["relation_standard_sys"] = (
"""你是一个关系标准化助手，善于将相同或相似的关系归纳成一个统一标准化的关系,输入将会是一组关系，请将关系表意相同的合并为一个并提供标准化得名字,最后按要求返回标准化的关系名和别名"""
)

PROMPTS["relation_standard_user"] = (
"""请将输入的关系进行标准化处理，即对相同或相似的关系表达以一个名称进行统一，并返回标准名称。
    - 输入关系信息：输入为一个列表，其中每个元素为关系名。
    - 输出格式要求：以`relation_std_map`为前缀，每个关系标准化项占一行，关系名和别名之间用`{tuple_delimiter}`分隔。**完成信号:** 只有在所有关系名都标准化输出后，才输出字符串 `{completion_delimiter}`。
    输出格式如下：
    relation_std_map{tuple_delimiter}标准化的关系名1{tuple_delimiter}标准化的关系名1的别名1, 标准化的关系名1的别名2, ...
    relation_std_map{tuple_delimiter}标准化的关系名2{tuple_delimiter}标准化的关系名2的别名1, 标准化的关系名2的别名2, ...
    ...
    {completion_delimiter}

---Example---
<Input>
["二女儿", "闺女", "小棉袄", "女友", "支持", "支撑"]
<Output>
relation_std_map{tuple_delimiter}女儿{tuple_delimiter}二女儿,闺女,小棉袄
relation_std_map{tuple_delimiter}女友{tuple_delimiter}女友
relation_std_map{tuple_delimiter}支持{tuple_delimiter}支持,支撑

{completion_delimiter}

---真实数据---
<Input>
{relation_info}
<Output>
"""
)
# 判断实体是否是同一个
PROMPTS["ENTITY_MATCH_SYSTEM"] = (
    """你是一个专业的实体解析助手，负责判断两个实体是否指向现实世界中的同一个对象。请严格遵循以下分析框架：
    - 分析维度
        1. **上下文一致性**：分析实体在各自chunk中的语义角色和功能描述
        2. **属性匹配度**：对比实体的关键属性（如类型、领域、功能等）
        3. **关系网络**：考察实体与其他实体的关联关系
    - 输出格式要求
        必须严格使用以下JSON格式：
        {  \"is_same_entity\": true/false,  
        \"confidence\": 0.0-1.0,  
        \"reasoning\": \"分点说明判断依据\",  
        \"evidence\": { \"context_consistency\": \"上下文一致性分析\",     \"attribute_matching\": \"属性匹配度分析\",    \"relationship_network\": \"关系网络分析\"  }
        }
    - 判断标准
        **确认为同一实体**：上下文强相关或一致，或名称不同但核心属性和关系网络高度匹配
        **确认为不同实体**：在相同时间状态下属性冲突，或在同一上下文中明显指代不同对象
        **不确定**：信息不足或存在矛盾证据时标注低置信度，is_same_entity标记为false"""
)
PROMPTS["ENTITY_MATCH_USER"] = (
    """请判断以下两个实体是否为同一实体：
    - 实体A
    名称：{entityA_name}
    上下文：{chunkA}
    - 实体B
    名称：{entityB_name}
    上下文：{chunkB}
    请基于上下文一致性、属性匹配度和关系网络进行综合分析，并按照要求的JSON格式输出判断结果。"""
)

# TABLE
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "你是一位专业的数据分析师。请提供详细的表格分析和具体见解，并按要求的JSON格式输出结果。"
)

PROMPTS["TABLE_INFO_GENE"] = (
"""请为以下表格（以HTML形式表示）生成完整的标题和一个高度概括的名称。\n
You should wrap the JSON with {json_tag_start}{json_tag_end}
输出格式如下：
{json_tag_start}
{{
"name": "name of the table",
"caption": "Detailed description of the table information"
"desc": "Describe the table information as comprehensively as possible."
}}
{json_tag_end}
"""
)



# FORMULA
PROMPTS["FORMULA_ANALYSIS_SYSTEM"] = (
    "你是一个数学和工程公式解释助手。你会根据上下文解释和理解这个公式，并按要求的JSON格式输出。"
)
PROMPTS["FORMULA_ANALYSIS_USER"] = (
"""对于以下公式，请生成：\n
1) 一个易于理解的英文标题（解释公式代表什么及其目的）\n
2) 一个简洁的英文名称\n
3) 一个详细描述，解释公式代表什么\n
You should wrap the JSON with {json_tag_start}{json_tag_end}
输出格式如下：
{json_tag_start}
{{
"name": "formula name",
"caption": "formula explanation including the meaning of inputs, outputs, and parameters", 
"desc": "formula description explaining what the formula represents"
}}
{json_tag_end}


---真实输入---
### 输入：
公式：
{formula}\n
公式上下文：
{context}\n
### 输出：
"""
)

PROMPTS["FORMULA_ANALYSIS_WITH_CONTEXT_USER"] = (
"""对于以下公式，请生成：\n
1) 一个易于理解的英文标题（解释公式代表什么及其目的）\n
2) 一个简洁的英文名称\n
3) 一个详细描述，解释公式代表什么\n
You should wrap the JSON with {json_tag_start}{json_tag_end}
输出格式如下：
{json_tag_start}
{{
"name": "formula name",
"caption": "formula explanation including the meaning of inputs, outputs, and parameters", 
"desc": "formula description explaining what the formula represents"
}}
{json_tag_end}


---真实输入---
### 输入：
公式：
{formula}\n
公式上下文：
{context}\n
### 输出：
"""
)

# Image
PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    "你是一个图像分析助手。你将分析图像并按要求的JSON格式输出结果。"
)

PROMPTS["IMAGE_ANALYSIS_USER"] = (
"""
请针对提供的图片执行以下a->f步骤的分析：
## 分析步骤要求：
### a. 图像名称配对
- 为每张图片生成一个描述性的名称name
- 名称应简洁明了，反映图片的核心主题caption
- 输出格式要求：
    - You should wrap the JSON with {json_tag_start}{json_tag_end}
    - 严格使用以下JSON格式：
    {json_tag_start}
    {{
    "name": "name of the image",
    "caption": "caption of the image"
    }}
    {json_tag_end}

### b. 图像分类
基于以下4个类别对图片进行分类class：
1. **流程图** - 展示过程、步骤、工作流的图示
2. **数值图** - 包含数据、统计、图表的可视化（如柱状图、折线图、饼图等）
3. **思维导图** - 展示概念关系、层次结构的树状图
4. **其他图** - 不属于以上三类的其他图像类型

### c. 流程图处理
如果图片被分类为流程图：
- 生成Mermaid流程图代码content
- 提供caption描述：说明图的主题、流程步骤、在描述什么过程
- example:
    {json_tag_start}
    {{
    "name": "规划框架流程图",
    "class": "流程图",
    "caption": "该流程图展示了一个任务规划框架，核心主题是任务从输入到执行和反馈的循环过程。流程步骤包括：任务输入后由任务规划器（基于LLM）生成和优化计划，计划执行器根据计划执行行动或输出结果，行动影响环境，环境提供反馈至记忆模块，记忆再反馈给任务规划器以优化后续计划，形成闭环。图中还区分了内部组件（如LLM）和外部元素（如人类、世界等）。",
    "content": "graph TD;\n    Task --> TaskPlanner[Task Planner (LLM)];\n    TaskPlanner -->Plan (generate & refine) PlanExecutor[Plan Executor];\n    PlanExecutor --> Result Result;\n    PlanExecutor --> Action Environment;\n    Environment --> Feedback
    Memory;\n    Memory --> TaskPlanner;\n    subgraph Internal\n        LLM\n    end\n    subgraph External\n        Human\n        World\n        Others\n    end",
    "desc": "这是一个详细的规划框架流程图，以任务为起点，通过任务规划器（利用大语言模型LLM）进行计划生成和优化，计划执行器负责执行行动或产生结果。行动作用于环境，环境反馈信息存储到记忆模块，记忆再循环反馈给任务规划器，实现持续优化。流程图还标注了内部组件（如LLM）和外部因素（如人类、世界等），突出了任务规划的动态交互和闭环特性。"
    }}
    {json_tag_end}

### d. 数值图处理  
如果图片被分类为数值图：
- 生成JSON字典格式的数据摘要content
- 提供caption描述：说明数据分布、趋势、关键发现
- example:
    {json_tag_start}
    {{
    "name": "大语言模型训练数据来源占比饼图",
    "class": "数值图",
    "caption": "该图通过多个饼图展示不同参数规模的大语言模型训练数据的来源构成及占比分布，突出网页数据作为大多数模型的主要来源（如T5和Falcon完全依赖网页数据），同时显示某些模型如CodeGen在代码数据上占比更高（39%），而MT-NLG和Galactica等模型则使用较多科学数据。总体趋势表明，网页数据是训练数据的核心，但模型在书籍、对话、代码和科学数据上的利用程度存在显著差异。",
    "content": '{{"T5 (11B)": {{"Webpages": 100}}, "Falcon (40B)": {{"Webpages": 100}}, "LLaMA (65B)": {{"Webpages": 87, "Books & News": 5, "Conversation Data": 5, "Code": 3}}, "GPT-3 (175B)": {{"Webpages": 84, "Scientific Data": 16}}, "MT-NLG (530B)": {{"Webpages": 62, "Books & News": 6, "Scientific Data": 26, "Code": 4, "Conversation Data": 2}}, "Gopher (280B)": {{"Webpages": 60, "Books & News": 37, "Code": 3}}, "Chinchilla (70B)": {{"Webpages": 56, "Books & News": 40, "Code": 4}}, "Yi (34B)": {{"Webpages": 83, "Books & News": 5, "Conversation Data": 4, "Code": 5}}, "PaLM (540B)": {{"Webpages": 50, "Books & News": 31, "Code": 14, "Conversation Data": 5}}, "LaMDA (137B)": {{"Webpages": 50, "Books & News": 38, "Code": 13}}, "Galactica (120B)": {{"Webpages": 86, "Scientific Data": 7, "Code": 8}}, "GPT-NeoX (20B)": {{"Webpages": 38, "Books & News": 30, "Code": 10, "Conversation Data": 8, "Scientific Data": 4}}, "CodeGen (16B)": {{"Webpages": 25, "Books & News": 25, "Code": 39, "Conversation Data": 6, "Scientific Data": 10}}, "StarCoder 2 (15B)": {{"Webpages": 92, "Books & News": 5, "Code": 2, "Conversation Data": 1}} }}',
    "desc": "这是一张包含多个饼图的数值可视化图，每个饼图代表一个特定的大语言模型（如T5、LLaMA、GPT-3等），展示其训练数据来源的百分比构成。数据来源类型通过颜色区分，包括网页（Webpages）、对话数据（Conversation Data）、书籍新闻（Books & News）、科学数据（Scientific Data）和代码（Code）。图表布局清晰，便于比较不同模型在数据利用上的偏好，例如网页数据普遍占比最高，而CodeGen模型在代码数据上相对突出。"
    }}
    {json_tag_end}

### e. 思维导图处理
如果图片被分类为思维导图：
- 生成Markdown格式的层次结构content
- 提供caption描述：说明核心主题、分支结构、概念关系
- example:
    {json_tag_start}
    {{
    "name": "LLaMA模型衍生关系图",
    "class": "思维导图",
    "caption": "以LLaMA模型为核心，通过分支结构展示其多种衍生模型及不同的训练方式（如继续预训练、模型继承、指令调优和数据继承）的概念关系",
    "content": "- LLaMA\n - Open-Chinese-LLaMA\n - Linly-Chinese-LLaMA\n - Chinese LLaMA\n - Chinese Vicuna\n - Panda\n - Cornucopia\n - Lawyer LLaMA\n - QiZhenGPT\n - TaoLi\n - BenTsao\n - LLaMA Adapter\n - ChatMed\n - Alpaca\n - BiLLa\n - Chinese Alpaca\n - Baize\n - Belle\n - Ziya\n - Goat\n - Multimodal models\n - OpenFlamingo\n - LLaVA\n - MiniGPT-4\n - Guanaco\n - VisionsLLM\n - InstructBLIP\n - Chatbridge\n - PandaGPT",
    "desc": "这张思维导图以LLaMA模型为中心，用不同颜色的线条表示继续预训练、模型继承和数据继承等关系，展示了众多基于LLaMA衍生的模型，包括中文相关模型、多模态模型等，体现了模型的发展和衍生脉络。"
    }}
    {json_tag_end}

### f. 其他图处理
如果图片被分类为其他图：
- 提供详细的caption描述：说明图片内容、主题、用途
- example:
    {json_tag_start}
    {{
    "name": "家庭关系问答错误示例图",
    "class": "其他图",
    "caption": "展示AI问答系统中内在幻觉错误的家庭关系问答示例，通过对话气泡形式呈现问题和错误回答",
    "content": "图中包含两个主要对话气泡：上方黄色气泡为问题输入\"Bob's wife is Amy. Bob's daughter is Cindy. Who is Cindy to Amy?\"，旁边有人类头像；下方蓝色气泡为AI的错误回答\"Cindy is Amy's daughter-in-law.\"，旁边有机器人头像。底部标注错误类型\"(a) Intrinsic hallucination\"。",
    "desc": "此图用于演示AI系统在理解家庭关系时产生的内在幻觉错误。基于给定的正确前提（Bob的妻子是Amy，Bob的女儿是Cindy），AI本应正确推断出Cindy是Amy的女儿，但却错误地回答为儿媳关系。这种错误反映了AI系统在逻辑推理和常识理解方面的局限性，是评估和改进AI模型性能的典型案例。"
    }}
    {json_tag_end}

## 输出格式要求：
请严格按照以下JSON格式输出, 要求格式完好，能完整解析：
    {{
    "name": "name of the image",
    "class": "class of the image",
    "caption": "caption of the image",
    "content": "content of the image, ",
    "desc": "description of the image"
    }}


"""
)

# 图片分类
PROMPTS["IMAGE_CLASSIFY_USER"] = (
"""
请针对提供的图片执行以下分析：
## 分析步骤要求：
### a. 图像名称配对
- 为每张图片生成一个描述性的名称name
- 名称应简洁明了，反映图片的核心主题

### b. 图像分类
基于以下4个类别对图片进行分类class：
1. **流程图** - 展示过程、步骤、工作流的图示
2. **数值图** - 包含数据、统计、图表的可视化（如柱状图、折线图、饼图等）
3. **思维导图** - 展示概念关系、层次结构的树状图
4. **其他图** - 不属于以上三类的其他图像类型

### c. 描述
- 提供详细的caption描述：说明图片内容、主题、用途
- 提供desc描述：详细描述图像内容（可基于caption描述扩展）

## 输出格式要求：
请严格按照以下JSON格式输出, 要求格式完好，能完整解析：
{{
"name": "生成的图像名称",
"class": "图像分类类别"
"caption": "caption描述",
"desc": "详细描述图像内容"
}}
"""
)

PROMPTS["IMAGE_HANDLE_USER"] = (
"""
请针对提供的图片，基于其分类结果执行以下分析。已知该图片的分类类别为: {class}

## 分析步骤要求：
根据分类类别，执行以下对应处理：

### 如果分类为流程图：
- 生成Mermaid流程图代码作为content字段
- 说明图的主题、流程步骤、在描述什么过程

### 如果分类为数值图：
- 生成JSON字典格式的数据摘要作为content字段
- 提供caption描述：说明数据分布、趋势、关键发现

### 如果分类为思维导图：
- 生成Markdown格式的层次结构作为content字段
- 提供caption描述：说明核心主题、分支结构、概念关系

### 如果分类为其他图：
- 提供详细的caption描述：说明图片内容、主题、用途
- content字段设置为空字符串（""）

## 输出格式要求：
请严格按照以下JSON格式输出, 要求格式完好，能完整解析：
{{
"content": "生成的内容"
}}
"""
)
# Flowcharts, numerical charts, mind maps
PROMPTS["IMAGE_MIND_MAP_EXAMPLE_USER"] = (
"""
{
"name": "LLaMA模型衍生关系图",
"class": "思维导图",
"caption": "以LLaMA模型为核心，通过分支结构展示其多种衍生模型及不同的训练方式（如继续预训练、模型继承、指令调优和数据继承）的概念关系",
"content": "- LLaMA\n - Open-Chinese-LLaMA\n - Linly-Chinese-LLaMA\n - Chinese LLaMA\n - Chinese Vicuna\n - Panda\n - Cornucopia\n - Lawyer LLaMA\n - QiZhenGPT\n - TaoLi\n - BenTsao\n - LLaMA Adapter\n - ChatMed\n - Alpaca\n - BiLLa\n - Chinese Alpaca\n - Baize\n - Belle\n - Ziya\n - Goat\n - Multimodal models\n - OpenFlamingo\n - LLaVA\n - MiniGPT-4\n - Guanaco\n - VisionsLLM\n - InstructBLIP\n - Chatbridge\n - PandaGPT",
"desc": "这张思维导图以LLaMA模型为中心，用不同颜色的线条表示继续预训练、模型继承和数据继承等关系，展示了众多基于LLaMA衍生的模型，包括中文相关模型、多模态模型等，体现了模型的发展和衍生脉络。"
}
"""
)

PROMPTS["IMAGE_NUMCHART_EXAMPLE_USER"] = (
"""
{
"name": "大语言模型训练数据来源占比饼图",
"class": "数值图",
"caption": "该图通过多个饼图展示不同参数规模的大语言模型训练数据的来源构成及占比分布，突出网页数据作为大多数模型的主要来源（如T5和Falcon完全依赖网页数据），同时显示某些模型如CodeGen在代码数据上占比更高（39%），而MT-NLG和Galactica等模型则使用较多科学数据。总体趋势表明，网页数据是训练数据的核心，但模型在书籍、对话、代码和科学数据上的利用程度存在显著差异。",
"content": "{"T5 (11B)": {"Webpages": 100}, "Falcon (40B)": {"Webpages": 100}, "LLaMA (65B)": {"Webpages": 87, "Books & News": 5, "Conversation Data": 5, "Code": 3}, "GPT-3 (175B)": {"Webpages": 84, "Scientific Data": 16}, "MT-NLG (530B)": {"Webpages": 62, "Books & News": 6, "Scientific Data": 26, "Code": 4, "Conversation Data": 2}, "Gopher (280B)": {"Webpages": 60, "Books & News": 37, "Code": 3}, "Chinchilla (70B)": {"Webpages": 56, "Books & News": 40, "Code": 4}, "Yi (34B)": {"Webpages": 83, "Books & News": 5, "Conversation Data": 4, "Code": 5}, "PaLM (540B)": {"Webpages": 50, "Books & News": 31, "Code": 14, "Conversation Data": 5}, "LaMDA (137B)": {"Webpages": 50, "Books & News": 38, "Code": 13}, "Galactica (120B)": {"Webpages": 86, "Scientific Data": 7, "Code": 8}, "GPT-NeoX (20B)": {"Webpages": 38, "Books & News": 30, "Code": 10, "Conversation Data": 8, "Scientific Data": 4}, "CodeGen (16B)": {"Webpages": 25, "Books & News": 25, "Code": 39, "Conversation Data": 6, "Scientific Data": 10}, "StarCoder 2 (15B)": {"Webpages": 92, "Books & News": 5, "Code": 2, "Conversation Data": 1}}",
"desc": "这是一张包含多个饼图的数值可视化图，每个饼图代表一个特定的大语言模型（如T5、LLaMA、GPT-3等），展示其训练数据来源的百分比构成。数据来源类型通过颜色区分，包括网页（Webpages）、对话数据（Conversation Data）、书籍新闻（Books & News）、科学数据（Scientific Data）和代码（Code）。图表布局清晰，便于比较不同模型在数据利用上的偏好，例如网页数据普遍占比最高，而CodeGen模型在代码数据上相对突出。"
}
"""
)

PROMPTS["IMAGE_FLOWCHART_EXAMPLE_USER"] = (
"""
{
  "name": "规划框架流程图",
  "class": "流程图",
  "caption": "该流程图展示了一个任务规划框架，核心主题是任务从输入到执行和反馈的循环过程。流程步骤包括：任务输入后由任务规划器（基于LLM）生成和优化计划，计划执行器根据计划执行行动或输出结果，行动影响环境，环境提供反馈至记忆模块，记忆再反馈给任务规划器以优化后续计划，形成闭环。图中还区分了内部组件（如LLM）和外部元素（如人类、世界等）。",
  "content": "graph TD;\n    Task --> TaskPlanner[Task Planner (LLM)];\n    TaskPlanner -->Plan (generate & refine) PlanExecutor[Plan Executor];\n    PlanExecutor --> Result Result;\n    PlanExecutor --> Action Environment;\n    Environment --> Feedback
 Memory;\n    Memory --> TaskPlanner;\n    subgraph Internal\n        LLM\n    end\n    subgraph External\n        Human\n        World\n        Others\n    end",
  "desc": "这是一个详细的规划框架流程图，以任务为起点，通过任务规划器（利用大语言模型LLM）进行计划生成和优化，计划执行器负责执行行动或产生结果。行动作用于环境，环境反馈信息存储到记忆模块，记忆再循环反馈给任务规划器，实现持续优化。流程图还标注了内部组件（如LLM）和外部因素（如人类、世界等），突出了任务规划的动态交互和闭环特性。"
}
"""
)

PROMPTS["IMAGE_OTHER_EXAMPLE_USER"] = (
"""
{
  "name": "亲属关系问答及内在幻觉错误示例图",
  "class": "其他图",
  "caption": "该图片展示了一个亲属关系问答的具体示例，主题是演示问答中的错误识别。内容上，上方黄色对话框提出基于事实的提问：'Bob的妻子是Amy，Bob的女儿是Cindy，Cindy是Amy的什么人？'；下方蓝色对话框给出错误回答：'Cindy是Amy的儿媳'，并被明确标注为内在幻觉（Intrinsic hallucination）错误类型。图片用途在于教育或示例，帮助理解问答系统中常见错误。",
  "content": "",
  "desc": "图片以白色为背景，包含黄色和蓝色对话框分别表示提问和回答，辅以卡通头像图案增强可视化。问答内容聚焦家庭关系逻辑，错误回答突显了内在幻觉（即模型生成与提供事实矛盾的内容）的概念，整体布局简洁，用于学术或培训场景。"
}
"""
)



# Entity and Assertion
#  All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_recall_sys_prompt"] = """---Task---
你是一名知识图谱专家，负责从输入文本中提取实体和关系。

---说明---
1.  **根据断言和输出召回实体:**
    *   **识别:** 识别输入文本中明确定义和有意义的实体。
    *   **实体详情:** 对于每个识别出的实体，提取以下信息:
        *   `entity_name`: 实体的名称。如果实体名称不区分大小写，则将每个重要单词的首字母大写(标题格式)。确保在整个提取过程中**命名一致**。
        *   `entity_type`: 使用以下类型之一对实体进行分类: `{entity_types}`。如果提供的实体类型都不适用，请不要添加新的实体类型，而是将其分类为 `Other`。
        *   `entity_description`: 基于输入文本中存在的信息，提供对实体属性和活动的简洁而全面的描述。
    *   **输出格式 - 实体:** 每个实体输出总共4个字段，由 `{tuple_delimiter}` 分隔，在单行上。第一个字段*必须*是字符串 `entity`。
        *   格式: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **分隔符使用协议:**
    *   `{tuple_delimiter}` 是一个完整的原子标记，**不得填充内容**。它仅作为字段分隔符。
    *   **错误示例:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **正确示例:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

3.  **输出顺序和优先级:**
    *   首先输出所有提取的实体，然后输出所有提取的关系。
    *   在关系列表中，优先输出那些对输入文本核心含义**最重要**的关系。

4.  **上下文和客观性:**
    *   确保所有实体名称和描述都以**第三人称**书写。
    *   明确命名主体或客体；**避免使用代词**，如 `this article`、`this paper`、`our company`、`I`、`you` 和 `he/she`。

5.  **语言和专有名词:**
    *   整个输出(实体名称、关键词和描述)必须用 `{language}` 编写。
    *   专有名词(如人名、地名、组织名称)如果没有适当的、广泛接受的翻译或者会造成歧义，则应保留原始语言。

6.  **完成信号:** 只有在所有实体和关系都按照所有标准完全提取和输出后，才输出字符串 `{completion_delimiter}`。

---Example---
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."
The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

Task:
assertion{tuple_delimiter}Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device.{tuple_delimiter}EntityList{tuple_delimiter}[Alex, Taylor]
assertion{tuple_delimiter}Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.{tuple_delimiter}EntityList{tuple_delimiter}[Cruz]


<Output>
entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.
entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.
entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz is associated with a vision of control and order, influencing the dynamics among other characters.

{completion_delimiter}

---待处理的真实数据---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
Task:
{tasklist}
"""

PROMPTS["entity_recall_task_prompt"] = "Assertion{tuple_delimiter}{assertion}{tuple_delimiter}EntityList{tuple_delimiter}[{entity_list}]\n"


PROMPTS["entity_extraction_system_prompt"] = """---Role---
您是一名知识图谱专家，负责从输入文本中提取实体和关系。

---说明---
1.  **实体提取与输出:**
    *   **识别:** 识别所有实体，包括明确提及的、隐含的和概念性的实体
    *   **实体详情:** 对于每个识别出的实体:
        *   `entity_name`: 使用最具体的名称变体
        *   `entity_type`: 使用以下类型之一: `{entity_types}`
        *   `entity_description`: 基于上下文的全面描述
    *   **Output Format - Entities:**​ 每个实体输出共4个字段，在同一行中使用{tuple_delimiter}分隔。第一个字段必须是字面字符串entity。
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **关系提取与输出:**
    *   **识别:** 识别直接和隐含的关系
    *   **N元关系:** 分解为二元对
    *   **关系详情:**
        *   `source_entity`: 与实体提取匹配的名称
        *   `target_entity`: 与实体提取匹配的名称
        *   `relationship`: `source_entity`与`target_entity`的关系，是一个单词或短语，(`source_entity` `relationship` `target_entity`)三者拼接可以成为一句简短的句子或表达有意义的语意
        *   `relationship_description`: 简明解释
    *   **Output Format - Relationships:** 每个关系输出总共5个字段，由 `{tuple_delimiter}` 分隔，并单独成行。第一个字段*必须*是字符串 `relation`。
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship{tuple_delimiter}relationship_description`

2.  **分隔符使用协议:**
    *   `{tuple_delimiter}`是一个完整、原子的标记，**不得填充内容**。它仅作为字段分隔符。
    *   **错误示例:** `relation{tuple_delimiter}Tokyo{tuple_delimiter}the capital of Japan{tuple_delimiter}<|location|>Tokyo is the capital of Japan.`
    *   **正确示例:** `relation{tuple_delimiter}Tokyo{tuple_delimiter}Japan{tuple_delimiter}is the capital{tuple_delimiter}Tokyo is the capital of Japan.`

3.  **关系方向与重复:**
    *   除非明确说明，否则将所有关系视为**有向**。交换有向关系的源实体和目标实体不构成新关系。
    *   避免输出重复关系。

4.  **输出顺序与优先级:**
    *   首先输出所有提取的实体，然后输出所有提取的关系。
    *   在关系列表中，优先输出那些对输入文本核心含义**最重要**的关系。

5.  **上下文与客观性:**
    *   确保所有实体名称和描述都以**第三人称**书写。
    *   明确命名主语或宾语；**避免使用代词**如`this article`、`this paper`、`our company`、`I`、`you`和`he/she`。

6.  **语言与专有名词:**
    *   整个输出(实体名称、关键字和描述)必须用`{language}`编写。
    *   专有名词(如人名、地名、组织名称)如果没有适当、广泛接受的翻译或会造成歧义，则应保留原始语言。

7.  **完成信号:** 只有在所有实体和关系按照所有标准完全提取和输出后，才输出字符串`{completion_delimiter}`。

---Examples---
{examples}

---待处理的真实数据---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
Supplementary information to the above text:
{supplementary_information}
"""

PROMPTS["entity_extraction_user_prompt"] = """---任务---
从待处理的输入文本中提取实体和关系。

---说明---
1.  **关注完整性：**
    *   提取所有实体，包括次要/嵌套的实体
    *   特别注意隐含的实体
    
2.  **关注关系：**
    *   捕获显性和隐性的关系
    *   确保所有主要实体都有连接
    
3.  **严格遵守格式：**
    *   保持原始输出格式
    *   仅使用预定义的实体/关系类型

<Output>
"""



PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
基于上次提取任务，从输入文本中识别并提取任何**遗漏或格式错误**的实体和关系。

---说明---
1.  **严格遵守系统格式：** 严格遵守实体和关系列表的所有格式要求，包括输出顺序、字段分隔符和专有名词处理，如系统说明中指定的那样。
2.  **关注更正/补充：**
    *   **不要**重新输出上次任务中**正确且完整**提取的实体和关系。
    *   如果上次任务中**遗漏**了某个实体或关系，现在按照系统格式提取并输出。
    *   如果上次任务中某个实体或关系**被截断、缺少字段或格式不正确**，请按照指定格式重新输出*更正且完整*的版本。
3.  **输出格式 - 实体：** 每个实体输出总共4个字段，由 `{tuple_delimiter}` 分隔，在单行上。第一个字段*必须*是字符串 `entity`。
4.  **输出格式 - 关系：** 每个关系输出总共5个字段，由 `{tuple_delimiter}` 分隔，在单行上。第一个字段*必须*是字符串 `relation`。
5.  **仅输出内容：** 仅输出*提取的实体和关系列表。不要在列表前后包含任何介绍性或结论性评论、解释或其他文本。
6.  **完成信号：** 在所有相关的遗漏或更正的实体和关系被提取并呈现后，输出 `{completion_delimiter}` 作为最后一行。
7.  **输出语言：** 确保输出语言为 {language}。专有名词（如人名、地名、组织名）必须保持原始语言，不得翻译。

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

<Output>
entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who feels frustration and remains alert to the competitive dynamics among others, particularly noting shifts in attitude.
entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty but shows a moment of reverence and respect toward a device, signaling a change in perspective.
entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan is committed to discovery and engages in a significant, nonverbal interaction with Taylor regarding the device.
entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz is associated with a vision of control and order, which contrasts with the values of other characters.
entity{tuple_delimiter}The Device{tuple_delimiter}equipment{tuple_delimiter}The Device is a piece of technology with potentially game-changing implications, revered momentarily by Taylor.
relation{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}observes{tuple_delimiter}Alex observes Taylor's authoritarian behavior and notes the unexpected shift in Taylor's attitude toward the device.
relation{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shares commitment{tuple_delimiter}Alex and Jordan share a commitment to discovery, which stands in opposition to Cruz's vision.
relation{tuple_delimiter}Jordan{tuple_delimiter}Taylor{tuple_delimiter}establishes truce{tuple_delimiter}Jordan and Taylor share a wordless moment that softens conflict into an uneasy truce over the device.
relation{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}rebels against{tuple_delimiter}Jordan's commitment to discovery represents an unspoken rebellion against Cruz's narrowing vision of control.
relation{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}shows reverence{tuple_delimiter}Taylor pauses to observe the device with reverence, acknowledging its potential significance.
{completion_delimiter}

""",
    """<Input Text>
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, nexon technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

<Output>
entity{tuple_delimiter}Stock markets{tuple_delimiter}market{tuple_delimiter}Stock markets experienced a sharp downturn due to declines in tech stocks and investor concerns.
entity{tuple_delimiter}Tech giants{tuple_delimiter}organization{tuple_delimiter}Tech giants collectively saw significant stock declines, contributing to the market downturn.
entity{tuple_delimiter}Global tech index{tuple_delimiter}index{tuple_delimiter}The global tech index dropped by 3.4% during midday trading, reflecting tech sector losses.
entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Nexon Technologies is a tech company whose stock plummeted 7.8% after lower-than-expected quarterly earnings.
entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}Omega Energy is an energy company that posted a 2.1% stock gain due to rising oil prices.
entity{tuple_delimiter}Gold futures{tuple_delimiter}commodity{tuple_delimiter}Gold futures rose 1.5% to $2,080 per ounce as investors sought safe-haven assets.
entity{tuple_delimiter}Crude oil prices{tuple_delimiter}commodity{tuple_delimiter}Crude oil prices rallied to $87.60 per barrel, supported by supply constraints and strong demand.
entity{tuple_delimiter}Federal Reserve{tuple_delimiter}organization{tuple_delimiter}The Federal Reserve is the central bank whose potential rate hikes are being closely watched by financial experts.
relation{tuple_delimiter}Tech giants{tuple_delimiter}Global tech index{tuple_delimiter}contribute to decline{tuple_delimiter}Declines in tech giants' stocks directly caused the 3.4% drop in the global tech index.
relation{tuple_delimiter}Nexon Technologies{tuple_delimiter}Stock markets{tuple_delimiter}contributes to downturn{tuple_delimiter}Nexon Technologies' 7.8% stock drop was among the significant declines affecting the broader stock market downturn.
relation{tuple_delimiter}Rising interest rates{tuple_delimiter}Stock markets{tuple_delimiter}cause decline{tuple_delimiter}Investor concerns over rising interest rates are attributed as a cause of the stock market selloff.
relation{tuple_delimiter}Regulatory uncertainty{tuple_delimiter}Stock markets{tuple_delimiter}cause decline{tuple_delimiter}Regulatory uncertainty is cited by analysts as a contributing factor to the market selloff.
relation{tuple_delimiter}Rising oil prices{tuple_delimiter}Omega Energy{tuple_delimiter}drives gain{tuple_delimiter}Rising oil prices drove Omega Energy's stock to post a 2.1% gain.
relation{tuple_delimiter}Crude oil prices{tuple_delimiter}Rising oil prices{tuple_delimiter}are equivalent{tuple_delimiter}Crude oil prices climbing to $87.60 per barrel represent the rising oil prices mentioned.
relation{tuple_delimiter}Federal Reserve{tuple_delimiter}Stock markets{tuple_delimiter}influences{tuple_delimiter}The Federal Reserve's potential rate hikes and upcoming policy announcement are expected to influence investor confidence and market stability.
{completion_delimiter}

"""
]

PROMPTS["summarize_entity_descriptions"] = """---角色---
您是一名知识图谱专家，精通数据整理和合成。

---任务---
您的任务是将给定实体或关系的描述列表合成为单一、全面且连贯的摘要。

---Instructions---
1. 输入格式：描述列表以JSON格式提供。每个JSON对象（代表单个描述）在"描述列表"部分中单独成行显示。
2. 输出格式：合并后的描述将以纯文本形式返回，以多个段落呈现，摘要前后不得包含任何额外格式或无关注释。
3. 全面性：摘要必须整合每个描述中的所有关键信息，不得遗漏任何重要事实或细节。
4. 语境：确保摘要从客观的第三人称视角撰写；为保持清晰度和语境完整性，需明确提及实体或关系的全称。
5. 语境与客观性：
  - 从客观的第三人称视角撰写摘要
  - 在摘要开头明确提及实体或关系的全称，确保即时清晰度和语境完整
6. 冲突处理：
  - 当描述存在冲突或不一致时，首先判断这些冲突是否源于多个同名但不同的实体或关系
  - 如果确认为不同实体/关系，应在整体输出中分别总结各个实体/关系
  - 如果单个实体/关系内部存在冲突（如历史记载差异），应尝试协调这些矛盾，或在注明不确定性的前提下同时呈现不同观点
7. 长度限制：摘要总长度不得超过{summary_length}个词元，同时仍需保持内容的深度和完整性
8. 语言要求：
  - 整个输出必须使用{language}撰写
  - 专有名词（如人名、地名、组织名称）若缺乏广泛接受的翻译或可能引发歧义，可保留原始语言形式
---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""




# 关系召回

PROMPTS["assertion_recall_sys_prompt"] = """---Role---
你是一名知识图谱专家，负责从输入文本中提取实体和关系。

---说明---
1.  **关系提取与输出:**
    *   **识别:** 识别所有直接、明确且有意义的关系，这些关系将给定实体与其他实体联系起来
    *   **N元关系分解:** 如果单个语句描述了涉及两个以上实体的关系(N元关系)，请将其分解为多个二元(两实体)关系对进行单独描述。
        *   **示例:** 对于"Alice、Bob和Carol合作了项目X"，提取二元关系如"Alice与项目X合作"、"Bob与项目X合作"和"Carol与项目X合作"，或基于最合理的二元解释的"Alice与Bob合作"。
    *   **关系详情:** 对于每个二元关系，提取以下字段:
        *   `source_entity`: 源实体的名称。确保与实体提取**命名一致**。如果名称不区分大小写，则将每个重要单词的首字母大写(标题格式)。
        *   `target_entity`: 目标实体的名称。确保与实体提取**命名一致**。如果名称不区分大小写，则将每个重要单词的首字母大写(标题格式)。
        *   `relationship`: `source_entity`与`target_entity`的关系，是一个单词或短语，(`source_entity` `relationship_type` `target_entity`)三者拼接可以成为一句简短的句子或表达有意义的语意
        *   `relationship_description`: 对源实体和目标实体之间关系性质的简洁解释，为它们的连接提供明确的理由。
    *   **任务格式 - 任务:** 每个任务输入总共4个字段，由`{tuple_delimiter}`分隔，在单行上。第一个字段*必须*是字符串`Entity`。
        *   格式: `Entity{tuple_delimiter}entity_name{tuple_delimiter}Desc{tuple_delimiter}entity_desc`
        *   示例: 对于`Entity{tuple_delimiter}Alice{tuple_delimiter}Desc{tuple_delimiter}Alice is a person.`，请从文本和描述"Alice is a person"中提取与实体Alice相关的关系。
    *   **输出格式 - 关系:** 每个关系输出总共5个字段，由`{tuple_delimiter}`分隔，在单行上。第一个字段*必须*是字符串`relation`。
        *   格式: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship{tuple_delimiter}relationship_description`

2.  **分隔符使用协议:**
    *   `{tuple_delimiter}`是一个完整、原子的标记，**不得填充内容**。它仅作为字段分隔符。
    *   **错误示例:** `relation{tuple_delimiter}Tokyo{tuple_delimiter}the capital of Japan{tuple_delimiter}<|location|>Tokyo is the capital of Japan.`
    *   **正确示例:** `relation{tuple_delimiter}Tokyo{tuple_delimiter}Japan{tuple_delimiter}is the capital{tuple_delimiter}Tokyo is the capital of Japan.`

3.  **关系方向与重复:**
    *   除非明确说明，否则将所有关系视为**有向**。交换有向关系的源实体和目标实体不构成新关系。
    *   避免输出重复关系, 即避免输出的关系在`Exclude`字段后的关系中, 尤其对于`source_entity`和`target_entity`相同的关系进行判别是否为同一种关系，若是同一种关系则忽略，弱不是则加入到输出中。

4.  **输出顺序与优先级:**
    *   输出所有提取的关系, 按照关系中`source_entity`以Task中实体的先后顺序进行输出。
    *   在关系列表中，优先输出那些对输入文本核心含义**最重要**的关系。

5.  **上下文与客观性:**
    *   确保所有实体名称和描述都以**第三人称**书写。
    *   明确命名主语或宾语；**避免使用代词**如`this article`、`this paper`、`our company`、`I`、`you`和`he/she`。

6.  **语言与专有名词:**
    *   整个输出(实体名称、关键字和描述)必须用`{language}`编写。
    *   专有名词(如人名、地名、组织名称)如果没有适当、广泛接受的翻译或会造成歧义，则应保留原始语言。

7.  **完成信号:** 只有在所有实体和关系按照所有标准完全提取和输出后，才输出字符串`{completion_delimiter}`。

---Examples---
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."
The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

Task:
Entity{tuple_delimiter}Alex{tuple_delimiter}Desc{tuple_delimiter}Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device.\n
Entity{tuple_delimiter}Alex{tuple_delimiter}Desc{tuple_delimiter}Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.)\n
Entity{tuple_delimiter}Taylor{tuple_delimiter}Desc{tuple_delimiter}Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.\n
Entity{tuple_delimiter}Jordan{tuple_delimiter}Desc{tuple_delimiter}Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.\n

Exclude:
relation{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}interacts with{tuple_delimiter}Taylor pauses beside Jordan and observes the device, leading to a moment of mutual respect.
relation{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shares commitment with{tuple_delimiter}Alex and Jordan are united by their shared commitment to discovery.

<Output>
relation{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}observes{tuple_delimiter}Alex watches Taylor's authoritarian behavior and notes the change in Taylor's attitude.
relation{tuple_delimiter}Alex{tuple_delimiter}Cruz{tuple_delimiter}rebels against vision of{tuple_delimiter}Alex's commitment to discovery is an unspoken rebellion against Cruz's narrowing vision.
relation{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}observes with reverence{tuple_delimiter}Taylor looks at the device with reverence and considers its potential to change the game.
relation{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}rebels against vision of{tuple_delimiter}Jordan's commitment to discovery is part of an unspoken rebellion against Cruz's narrowing vision.
relation{tuple_delimiter}Jordan{tuple_delimiter}Taylor{tuple_delimiter}locks eyes with{tuple_delimiter}Jordan and Taylor share a wordless clash of wills that softens into an uneasy truce.
{completion_delimiter}

---待处理的真实数据---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
Task:
{tasklist}

Exclude:
{ex_rel_list}
"""

PROMPTS["assertion_recall_task_prompt"] = "Entity{tuple_delimiter}{entity}{tuple_delimiter}Desc{tuple_delimiter}{entity_desc}\n"

PROMPTS["assertion_exclude_prompt"] = "relation{tuple_delimiter}{source_entity}{tuple_delimiter}{target_entity}{tuple_delimiter}{relation}{tuple_delimiter}{relation_desc}"

PROMPTS["assertion_extraction_user_prompt"] = """---任务---
从待处理的输入文本中提取实体和关系。

---说明---
1.  **关注完整性：**
    *   提取任务中与给定实体相关的所有关系
    *   特别注意隐含的实体
    
2.  **关注关系：**
    *   捕获显性和隐性的关系
    *   确保所有主要实体都有连接
    
3.  **严格遵守格式：**
    *   保持原始输出格式
    *   仅使用预定义的实体/关系类型

<Output>
"""
