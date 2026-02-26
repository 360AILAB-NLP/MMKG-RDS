<div align="center">
<img src="assets/logo.png" alt="MMKG-RDS Logo" width="150" height="150" style="vertical-align: middle;">

# MMKG-RDS

### Rasoning Data Synthesis via Deep Mining of Multimodal Knowledge Graphs
</div> 

## 🎯 Overview

Synthesizing high-quality training data is crucial for enhancing domain models' reasoning abilities. Existing methods face limitations in long-tail knowledge coverage, effectiveness verification, and interpretability. Knowledge-graph-based approaches still fall short in functionality, granularity, customizability, and evaluation. To address these issues, we propose MMKG-RDS—a flexible framework for reasoning data synthesis that leverages multimodal knowledge graphs. It supports fine-grained knowledge extraction, customizable path sampling, and multidimensional data quality scoring. We validate MMKG-RDS with the MMKG-RDS-Bench dataset, covering five domains, 17 task types, and 14,950 samples. Experimental results show fine-tuning Qwen3 models (0.6B/8B/32B) on a small number of synthesized samples improves reasoning accuracy by 9.2\%. The framework also generates distinct data, challenging existing models on tasks involving tables and formulas, useful for complex benchmark construction. 

![](./assets/framwork.png)

### 🔍 Key Features
- **📚 Data Preprocessing**: Unified processing of structured data (JSON/CSV) and unstructured documents (PDF/PNG/PPT/DOC), enabling triplet conversion and multimodal content extraction (text, images, tables, formulas).
- **🕸️ Knowledge Graph Construction**: Automated KG construction with fully customizable schemas, supporting entity–relation constraints and the complete pipeline of extraction, disambiguation, and normalization.
- **💾 Flexible Storage**: Compatible with multiple storage backends (Neo4j, NetworkX, JSON), with seamless integration into Neo4j’s visualization and analytical ecosystem.
- **🎯 Reasoning Data Synthesis**: Knowledge-graph-driven data synthesis via subgraph sampling, path generation, and entity fuzzification, enabling controllable reasoning QA generation with balanced difficulty and structure preservation.
- **📊 Quality Analysis & Evaluation**: Multidimensional data quality assessment covering support, difficulty, and complexity, along with fine-grained analytics on token distribution, task types, and domain coverage.


### 🔁 Pipeline
A unified pipeline that preprocesses multimodal data, constructs customizable knowledge graphs, and synthesizes high-quality reasoning datasets with flexible storage and comprehensive quality evaluation.
![](./assets/pipeline.png)

## 🚀 Quick Start

### 1. Environment Setup

1. 安装LibreOffice
```bash
sudo apt-get update
sudo apt-get install libreoffice
```
2. 中文转化乱码，请下载字体并放到文件夹/usr/share/fonts/下
```bash
fc-list :lang=zh # 查看是否安装中文字体
cp ./fonts/msyh.ttf /usr/share/fonts/
```
3. 安装MinerU
```bash
uv pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple
```
4. 安装Python环境
```bash
uv pip install -U -r requirements.txt
```


### 2. Configuration Setup

#### Modify Configuration
Configure in `configs/dev.yaml`:

#### Run the Project

```bash
python main.py
```

这将分为5个阶段，包括：
1. 文档处理及知识图谱构建
2. 数据生成
3. 去重、评分
4. 数据导出
5. 模型评估

## 📁 Project Structure

```
MMKG-RDS  # 多模态知识图谱关系数据合成系统
├─ .gitignore                           # Git忽略配置
├─ main.py                              # 主程序测试, 包含5个阶段
├─ main_law.py                          # 法律领域示例程序
├─ README.md                            # 项目说明文档
├─ requirements.txt                     # Python依赖列表
├─ run.py                               # 运行入口脚本
├─ util                                 # 工具函数目录
│  ├─ any2pdf.py                        # 任意格式转PDF
│  ├─ errors.py                         # 错误处理模块
│  ├─ export2std_data.py                # 导出标准数据格式
│  ├─ json2graph.py                     # JSON转图结构
│  ├─ jsonparser.py                     # JSON解析器
│  ├─ monitor.py                        # 监控模块
│  ├─ pdf2md.py                         # PDF转Markdown
│  ├─ tool.py                           # 通用工具函数
│  └─ tuple2json.py                     # 元组转JSON
├─ schema                               # 模式定义目录
│  ├─ test.json                         # 测试数据文件
│  └─ test.schema                       # 测试模式定义
├─ qafilter                             # QA过滤模块
│  └─ enhanced_refactored_pipeline.py   # 增强重构管道
├─ prompts                              # 提示词模板目录
│  ├─ dataprocess_prompt.py             # 数据处理提示词
│  ├─ datasynthesis_prompt.py           # 数据合成提示词
│  ├─ flowchart.png                     # 流程图
│  ├─ mindmap.png                       # 思维导图
│  ├─ numchart.png                      # 数值图表
│  ├─ other.png                         # 其他图表
│  └─ task_prompt.py                    # 任务提示词
├─ processor                            # 数据处理模块
│  ├─ chunk.py                          # 文本分块处理
│  ├─ edge.py                           # 边数据处理
│  ├─ modal.py                          # 各个节点数据处理
│  ├─ node.py                           # 节点数据处理
│  └─ processor.py                      # 主处理器
├─ output_dir                           # 输出目录
│  ├─ data_filter.json                  # 过滤后数据
│  ├─ data_filter_invalid.json          # 无效数据
│  ├─ data_filter_invalid_report.json   # 无效数据报告
│  ├─ data_filter_statistics.json       # 过滤统计信息
│  ├─ data_gene.json                    # 生成数据
│  └─ graph.graphml                     # 图数据文件
├─ llms                                 # LLM客户端目录
│  ├─ client.py                         # OpenAI客户端
│  ├─ emb.py                            # 嵌入模型客户端
│  └─ vision_client.py                  # 视觉模型客户端
├─ fonts                                # 字体文件目录
│  └─ msyh.ttf                          # 微软雅黑字体
├─ eval                                 # 评估模块目录
│  ├─ eval_up.py                        # llm评估脚本
│  └─ eval_up_vl.py                     # 视觉语言评估脚本
├─ data_synthesis                       # 数据合成核心模块
│  ├─ constants.py                      # 常量定义，任务等
│  ├─ filter.py                         # 数据过滤
│  ├─ generate_qa.py                    # QA生成
│  ├─ information_blur.py               # 实体模糊
│  ├─ net_utils.py                      # networkx工具
│  ├─ rewarite.py                       # 重写模块
│  ├─ subgraph_sampling.py              # 子图采样
│  └─ trace_generate.py                 # 路径生成
├─ data                                 # 原始数据目录
│ ...
│  └─ chemistry                         # 化学领域数据
│     └─ Fundamentals-of-Organic-Chemistry-by-John-McMurry-7th-Edition.pdf
├─ config                               # 配置文件目录
│  ...
│  └─ test.yaml                         # 测试的配置文件
├─ assets                               # 资源文件目录
│  └─ logo.png                          # 项目Logo
└─outputs                               # 输出文件目录
```
