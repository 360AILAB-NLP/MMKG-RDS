#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本脚本实现一个基于大模型的 QA/CoT数据评估，主要功能包括：
# 1. 使用向量嵌入与语义相似度对有效数据进行去重；
# 2. 调用多种评估模型，对每条数据进行：
#    - 支持度评估；
#    - 难度评估；
#    - 复杂度评估；
# 3. 对评估结果做统计分析，生成统计 JSON 和 Plotly HTML 可视化报告；


from ast import Str
import json
import os
import sys
import argparse
import yaml
import requests
import asyncio
import aiohttp
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict, Counter
import logging
from llms.emb import EmbeddingClient
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import logging
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
import requests

"""
日志 
"""
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
ENABLE_LOGGING = True
if ENABLE_LOGGING:
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()],
    )
else:
    logging.basicConfig(
        level=logging.CRITICAL + 1
    )
logger = logging.getLogger(__name__)


class RefactoredEnhancedEvaluationPipeline:
    """：
    加载和合并配置
    对输入的 QA / CoT 数据进行：结构校验、去重、评估（支持度 / 难度 / 复杂度）
    将评估结果与统计分析信息保存到文件
    生成 HTML 可视化报告
    """

    def __init__(self, cfg, **kwargs):
        """
        初始化评估对象
        Args:
            cfg (dict): 已加载好的配置字典
            **kwargs: 预留扩展参数
        """
        self.config = cfg
        self.api_config = self._load_api_config()
        eval_cfg = self.config.get('evaluation', {})
        self.eval_modes = set(eval_cfg.get('modes', ['support', 'difficulty', 'complexity']))
        logger.info(f"评估初始化完成, 启用评估类型: {self.eval_modes}")
        self.config = cfg
        self.api_config = self._load_api_config()
        self.CHAT_SUFFIX = "/chat/completions"
        self.EMBEDDING_SUFFIX = "/embeddings"

    def _load_config(self, config_file: str, cli_args: dict) -> Dict:

        """
        加载配置文件并与命令行参数进行合并
        Args:
            config_file (str): 配置文件路径（YAML）
            cli_args (dict): 从命令行解析得到的参数字典

        Returns:
            Dict: 合并后的完整配置字典
        """
        config_file = os.path.abspath(config_file)
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"已加载配置文件: {config_file}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")
        config.setdefault('llm', {})
        config.setdefault('evaluation', {})
        config['evaluation'].setdefault('deduplication', {})
        config['evaluation'].setdefault('io', {})
        config = self._merge_cli_args(config, cli_args)

        return config

    def _load_api_config(self) -> List[Dict]:
        """
        从配置中加载支持度评估使用的模型配置列表
        Returns:
            List[Dict]: evaluation_models.support_models 配置列表
        """
        return self.config.get('evaluation_models', {}).get('support_models', [])

    def _merge_cli_args(self, config: Dict, cli_args: dict) -> Dict:
        """
        将命令行参数合并到配置中

        Args:
            config (Dict): 原始配置
            cli_args (dict): 命令行参数

        Returns:
            Dict: 合并后的配置
        """
        config.setdefault('llm', {})
        config['llm'].setdefault('pass4', {})
        config.setdefault('evaluation', {})
        if cli_args.get('model'):
            config['llm']['model'] = cli_args['model']
        if cli_args.get('api_key'):
            config['llm']['api_key'] = cli_args['api_key']
        if cli_args.get('base_url'):
            config['llm']['base_url'] = cli_args['base_url']
        if cli_args.get('backend'):
            config['llm']['backend'] = cli_args['backend']
        if cli_args.get('batch_size'):
            config['evaluation']['batch_size'] = cli_args['batch_size']
        return config

    def deduplicate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对输入数据进行去重
        当前策略：
            - 如果 ENABLE_DEDUP 打开，则基于语义相似度进行去重
            - 否则直接返回原始数据
        Args:
            data (List[Dict[str, Any]]): 原始数据列表

        Returns:
            List[Dict[str, Any]]: 去重后的数据列表
        """
        ENABLE_DEDUP = True
        if not ENABLE_DEDUP:
            logger.info("去重功能关闭，跳过")
            return data
        return self._deduplicate_by_semantic_similarity(data)

    def _deduplicate_by_semantic_similarity(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于语义相似度进行去重
        Args:
            data (List[Dict[str, Any]]): 原始数据列表
        Returns:
            List[Dict[str, Any]]: 语义去重后的数据列表
        """
        return data
        embedding_config = self.config.get('embedding_model', {})
        if not embedding_config:
            support_models = self.config.get('evaluation_models', {}).get('support_models', [])
            if support_models:
                embedding_config = support_models[0]
                logger.info(f"使用支持度评估模型作为嵌入模型: {embedding_config['model']}")
            else:
                logger.error("配置中未找到嵌入模型设置，无法使用语义去重")
                return data

        api_key = embedding_config.get('api_key', '')
        base_url = embedding_config.get('base_url', '')
        model_name = embedding_config.get('model', '')
        emb_client = EmbeddingClient(base_url, api_key, model_name)
        threshold = 0.95
        texts = []
        for item in data:
            question = item.get('q', item.get('question', ''))
            answer = item.get('a', item.get('answer', ''))
            cot = item.get('cot', '')
            texts.append(f"{cot}")
        batch_size = 10
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = emb_client.get_embedding(batch_texts)
            if batch_embeddings is None:
                logger.error("获取嵌入向量失败，跳过语义去重")
                return data
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)

        deduplicated_data = []
        seen_indices = set()

        for i in range(len(embeddings)):
            if i in seen_indices:
                continue
            deduplicated_data.append(data[i])
            for j in range(i + 1, len(embeddings)):
                if j in seen_indices:
                    continue
                sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
                if sim >= threshold:
                    seen_indices.add(j)
                    logger.debug(f"发现相似项: {i}和{j}，相似度: {sim:.4f}")
        logger.info(f"语义去重完成 - 原始数据: {len(data)} 条，去重后: {len(deduplicated_data)} 条")
        return deduplicated_data

    def load_input_data(self, input_file: str) -> List[Dict]:
        """
        加载输入数据文件，并进行基础格式校验
        Args:
            input_file (str): 输入文件路径
        Returns:
            List[Dict]: 加载后的数据列表
        Raises:
            FileNotFoundError: 输入文件不存在
            ValueError: 数据格式不符合预期
        """
        try:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"输入文件不存在: {input_file}")

            with open(input_file, 'r', encoding='utf-8') as f:
                if input_file.endswith('.json'):
                    data = json.load(f)
                elif input_file.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    raise ValueError("不支持的文件格式，请使用.json或.jsonl文件")

            if not isinstance(data, list):
                raise ValueError("输入数据格式错误，应为列表")
            return data
        except Exception as e:
            logger.error(f"加载输入数据失败: {str(e)}", exc_info=True)
            raise

    def validate_data_structure(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        验证数据结构是否完整，并区分无效数据
        Args:
            data (List[Dict]): 原始数据列表
        Returns:
            Tuple[List[Dict], List[Dict]]:
                - valid_data: 结构完整的数据列表
                - invalid_data: 结构缺失的数据列表
        """
        required_fields = ['q', 'a', 'cot', 'task_type',
                           'evidence']  # ['q', 'a', 'cot', 'kgpath', 'task_type', 'node_ids', 'node_name', 'relations']
        valid_data = []
        invalid_data = []

        for item in data:
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                error_msg = f"缺失字段: {', '.join(missing_fields)}"
                logger.warning(f"异常数据记录 - {error_msg}")
                invalid_item = {
                    **item,
                    'error': error_msg,
                    'error_type': 'MISSING_FIELD',
                    'is_valid': False,
                    'original_data': item.copy()
                }
                invalid_data.append(invalid_item)
            else:
                valid_data.append({**item, 'is_valid': True})
        logger.info(f"数据结构验证完成 - 有效数据: {len(valid_data)} 条, 异常数据: {len(invalid_data)} 条")
        return valid_data, invalid_data

    async def _post_with_retry(self, session, url: str, headers: Dict, payload: Dict, timeout: int = 60,
                               retries: int = 3) -> Dict:
        """
        带重试机制的 POST 请求封装
        Args:
            session: aiohttp session 对象
            url: 请求地址
            headers: 请求头
            payload: 请求体
            timeout: 超时时间（秒）
            retries: 重试次数
        Returns:
            Dict: 响应的 JSON 数据
        Raises:
            Exception: 重试耗尽后抛出最后一次异常
        """
        last_exception = None
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        for attempt in range(retries):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=client_timeout) as response:
                    response.raise_for_status()
                    return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
                last_exception = e
                if attempt < retries - 1:
                    wait_time = 1 * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"请求最终失败 (尝试 {retries}/{retries}): {url} - {str(e)}")

        if last_exception:
            raise last_exception
        raise Exception("Unknown error in _post_with_retry")

    def build_url(self, base_url: str, suffix: str) -> str:
        """
        根据 base_url 和固定 suffix 构造完整请求 URL
        Args:
            base_url (str): API 基础地址
            suffix (str): 需要拼接的路径后缀（如 /chat/completions）
        Returns:
            str: 拼接后的完整 URL
        Raises:
            ValueError: 当 base_url 为空时抛出
        """
        if not base_url:
            raise ValueError("base_url 不能为空")
        base_url = base_url.rstrip('/')
        if base_url.endswith(suffix) or suffix in base_url:
            return base_url
        return base_url + suffix

    async def evaluate_support(self, question: str, answer: str, cot: str, kgpath: str) -> Dict:
        """
        评估单条数据的支持度
        Args:
            question (str): 问题文本
            answer (str): 参考答案
            cot (str): 推理过程
            kgpath (str): 知识图谱路径描述

        Returns:
            Dict: 评估结果字典，示例：
                {
                    "support_count": int,   # 支持票数
                    "support": "支持" 或 "不支持"
                }
        """
        support_models = self.api_config
        eval_cfg = self.config.get('evaluation', {})
        support_cfg = eval_cfg.get('support', {})

        if not support_models:
            logger.error("支持度评估模型未配置")
            return {"support": "未评估", "support_count": 0}

        model_indices = support_cfg.get('models')
        if model_indices:
            selected = []
            for idx in model_indices:
                if 0 <= idx < len(support_models):
                    selected.append(support_models[idx])
                else:
                    logger.warning(f"支持度评估模型下标超出范围: {idx}")
            if selected:
                support_models = selected
            else:
                logger.error("支持度评估配置的 models 为空或非法")
                return {"support": "未评估", "support_count": 0}

        mode = support_cfg.get('mode', 'majority_vote')
        prompt = f"""
[知识支持度评估]
请严格评估：
1. 问题: {question}
2. 参考答案: {answer}
3. 推理过程: {cot}
4. 知识图谱路径: {kgpath}

评估标准:
- 返回1（支持）当且仅当:
  a) 答案准确回答问题
  b) 推理完全基于知识图谱路径
  c) 无事实错误或幻觉
- 返回0（不支持）如果存在:
  a) 答案错误或未回答问题
  b) 推理与知识图谱路径不一致
  c) 包含知识图谱中不存在的信息

注意: 只返回整数0或1，不要包含其他内容。
"""

        async with aiohttp.ClientSession() as session:
            # -----------------
            # single 模式：只调用一个模型
            # -----------------
            if mode == 'single':
                first_model = support_models[0]
                result = await self._evaluate_single_support_model(session, first_model, prompt)
                support_result = "支持" if result == 1 else "不支持"
                return {
                    "support_count": int(result),
                    "support": support_result
                }

            tasks = []
            for model_cfg in support_models:
                tasks.append(self._evaluate_single_support_model(session, model_cfg, prompt))

            results = await asyncio.gather(*tasks)
            support_count = sum(results)
            support_result = "支持" if support_count >= (len(results) // 2 + 1) else "不支持"

            return {
                "support_count": support_count,
                "support": support_result
            }

    async def _evaluate_single_support_model(self, session, model_cfg, prompt) -> int:
        """
        调用单个模型进行支持度评估（返回 0 或 1）
        已增加重试和超时机制
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {model_cfg.get('api_key', '')}"
            }

            payload = {
                "model": model_cfg['model'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1
            }

            url = self.build_url(model_cfg.get("base_url", ""), self.CHAT_SUFFIX)

            data = await self._post_with_retry(
                session, url, headers, payload, timeout=60, retries=3
            )

            content = data['choices'][0]['message']['content'].strip()
            return int(content) if content in ("0", "1") else 0

        except Exception as e:
            logger.error(f"支持度评估失败: {str(e)}")
            return 0

    async def evaluate_difficult(self, question: str, support: str, answer: str) -> Dict:
        """
        评估难度
        Args:
            question (str): 问题文本
            answer (str): 参考答案

        Returns:
            Dict: 难度评估结果，例如：
                {
                    "level": "simple|medium|hard|未评估",
                    "strong_correct": bool 或 None,
                    "weak_correct": bool 或 None
                }
        """
        difficulty_models = self.config.get('evaluation_models', {}).get('difficulty_models', {})
        if not difficulty_models:
            logger.error("难度评估模型未配置")
            return {
                "level": "未评估",
                "strong_correct": None,
                "weak_correct": None
            }

        strong_model = difficulty_models.get('strong')
        weak_model = difficulty_models.get('weak')

        eval_cfg = self.config.get('evaluation', {})
        difficulty_cfg = eval_cfg.get('difficulty', {})
        mode = difficulty_cfg.get('mode', 'strong_weak')

        strong_correct = None
        weak_correct = None

        if mode in ('strong_weak', 'strong_only'):
            strong_correct = await self._evaluate_difficult_single_model(
                strong_model, question, support, answer
            )

        if mode in ('strong_weak', 'weak_only'):
            weak_correct = await self._evaluate_difficult_single_model(
                weak_model, question, support, answer
            )

        if mode == 'strong_weak':
            if strong_correct and weak_correct:
                difficult_level = "simple"
            elif not strong_correct and weak_correct:
                difficult_level = "hard"
            elif strong_correct and not weak_correct:
                difficult_level = "medium"
            else:
                difficult_level = "hard"
        elif mode == 'strong_only':
            difficult_level = "simple" if strong_correct else "hard"
        elif mode == 'weak_only':
            difficult_level = "simple" if weak_correct else "hard"
        else:
            difficult_level = "未评估"

        return {
            "level": difficult_level,
            "strong_correct": strong_correct,
            "weak_correct": weak_correct
        }

    async def _evaluate_difficult_single_model(self, model_cfg, question: str, support: str, answer: str) -> bool:
        """
        使用单个模型评估参考答案是否正确
        已增加重试和超时机制
        """
        prompt = f"""
现在交给你一个阅读理解的任务，给定一个question 以及对应的参考信息support，你需要基于support中给定的信息来回答question，并返回answer。
请注意：1、answer应该来自于support中的片段，不要发散。
2、请直接将answer按照json的格式返回，如{{"answer":xxxx}}。

question为：{question}
support为：{support}

你的答案是： /no_think
        """

        def match_answer(text):
            import re
            import json
            # print(text)
            try:
                data = json.loads(text)
                return data.get("answer", text)
            except json.JSONDecodeError:
                pass
            # 基础匹配：提取引号内的内容
            # text = '{"answer":-112.12}'
            pattern1 = r'\{"answer":(.*?)\}'
            match = re.search(pattern1, text)
            if match:
                result = match.group(1).strip()
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                elif result.startswith('\'') and result.endswith('\''):
                    result = result[1:-1]
                return result
            else:
                return text

        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {model_cfg.get('api_key', '')}"
            }

            payload = {
                "model": model_cfg['model'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 4096
            }

            url = self.build_url(model_cfg.get("base_url", ""), self.CHAT_SUFFIX)

            async with aiohttp.ClientSession() as session:
                # 使用封装的重试方法，设置超时为 90秒，重试 3次
                data = await self._post_with_retry(
                    session, url, headers, payload, timeout=90, retries=3
                )
                content = data['choices'][0]['message']['content'].strip()
                pred = match_answer(content)
                return pred == answer

        except Exception as e:
            logger.error(f"难度评估失败: {str(e)}")
            return False

    async def evaluate_complex(self, data_item: Dict) -> Dict:
        """
        评估单条数据（指令）的“复杂度分数”
        已增加重试和超时机制
        """
        eval_cfg = self.config.get('evaluation', {})
        complex_cfg = eval_cfg.get('complexity', {})

        if not complex_cfg.get("enabled", True):
            return {"score": 0, "reason": "复杂度评估未启用"}

        eval_models = self.config.get("evaluation_models", {})
        model_cfg = eval_models.get("complexity_model")
        if not model_cfg:
            logger.error("未找到用于复杂度评估的模型配置 (evaluation_models.complexity_model)")
            return {"score": 0, "reason": "复杂度评估模型未配置"}

        api_key = model_cfg.get("api_key", "")
        model_name = model_cfg.get("model", "")
        base_url = model_cfg.get("base_url", "")

        if not base_url:
            logger.error("复杂度评估模型 base_url 为空")
            return {"score": 0, "reason": "复杂度评估 base_url 未配置"}

        url = self.build_url(base_url, self.CHAT_SUFFIX)
        instruction = data_item.get("q", "") or ""
        if not instruction:
            return {"score": 0, "reason": "指令为空，无法评估"}

        prompt = f"""
请评估以下指令的复杂度，从1到5分进行评分（1分为非常简单，5分为非常复杂）。

评分标准基于：
- 语言复杂度：指令的文本难度，包括句子长度、词汇复杂性和可读性。
- 任务复杂度：指令所要求任务的难度，包括所需推理步骤、知识领域和操作复杂性。
- 结构复杂度：指令的结构组织，如是否包含子任务、条件逻辑或模糊性。

请根据以上标准给出一个综合评分，并简要解释理由。

指令：\"{instruction}\"

输出格式：
复杂度评分：[1-5]
理由：
- 总体判断：用 1~2 句话总结为什么是这个分数；
- 语言难度：分析句子结构是否复杂、是否使用专业术语；
- 任务性质：解释执行任务需要多少推理步骤/知识量；
- 结构特征：指出是否存在多层条件、子任务或隐含推理；
- 其他因素：如歧义性、跨领域性、知识依赖程度等。

请确保理由内容丰富、分析充分，而不是简单几句话。
"""
        content = ""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            }

            async with aiohttp.ClientSession() as session:
                data = await self._post_with_retry(
                    session, url, headers, payload, timeout=90, retries=3
                )
                content = data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"复杂度评估请求失败: {str(e)}")
            return {"score": 0, "reason": f"请求失败: {e}"}

        import re
        score = 0
        reason = ""

        m = re.search(r"复杂度评分[:：]\s*([1-5])", content)
        if m:
            score = int(m.group(1))
        else:
            m2 = re.search(r"[：:\s]\s*([1-5])", content)
            if m2:
                score = int(m2.group(1))
            else:
                score = 0

        m_reason = re.search(r"理由[:：]\s*(.*)", content, re.S)
        if m_reason:
            reason = m_reason.group(1).strip()
        else:
            reason = content

        return {
            "score": score,
            "reason": reason,
        }

    async def evaluate_single_item(self, data_item: Dict) -> Dict:
        """
         对单条数据项执行完整评估
        Args:
            data_item (Dict): 原始数据项字典

        Returns:
            Dict: 带有评估结果的 data_item（原内容 + support/difficult/complexity/num_hops 字段）
        """
        try:
            question = data_item.get('q', '')
            answer = data_item.get('a', '')
            cot = data_item.get('cot', '')
            kgpath = data_item.get('kgpath', '')
            evidence = str(data_item.get('evidence', ''))
            eval_cfg = self.config.get('evaluation', {})

            support_evaluation = None
            difficult_evaluation = None
            complex_evaluation = None

            if 'support' in getattr(self, "eval_modes", {'support', 'difficulty', 'complexity'}) \
                    and eval_cfg.get('support', {}).get('enabled', True):
                support_evaluation = await self.evaluate_support(question, answer, cot, evidence)

            if 'difficulty' in getattr(self, "eval_modes", {'support', 'difficulty', 'complexity'}) \
                    and eval_cfg.get('difficulty', {}).get('enabled', True):
                difficult_evaluation = await self.evaluate_difficult(question, evidence, answer)

            if 'complexity' in getattr(self, "eval_modes", {'support', 'difficulty', 'complexity'}) \
                    and eval_cfg.get('complexity', {}).get('enabled', True):
                complex_evaluation = await self.evaluate_complex(data_item)

            if support_evaluation is not None:
                support_evaluation["label"] = support_evaluation.get("support", "未评估")
                data_item["support"] = support_evaluation
            else:
                data_item["support"] = {"label": "未评估", "support_count": 0, "support": "未评估"}

            if difficult_evaluation is not None:
                data_item["difficult"] = difficult_evaluation
            else:
                data_item["difficult"] = {"level": "未评估", "strong_correct": None, "weak_correct": None}

            if complex_evaluation is not None:
                data_item["complexity"] = complex_evaluation
            else:
                data_item["complexity"] = {"score": 0, "reason": "未评估"}
            node_ids = data_item.get("node_ids", [])
            ent_count = 0
            if isinstance(node_ids, (list, tuple)):
                ent_count = sum(1 for node in node_ids if isinstance(node, str) and node.strip().startswith("ent-"))

            num_hops = max(ent_count - 1, 0)

            data_item["num_hops"] = num_hops
            data_item["evaluation_timestamp"] = datetime.now().isoformat()
            return data_item

        except Exception as e:
            logger.error(f"评估数据项失败: {str(e)}")
            data_item["error"] = str(e)
            return data_item

    async def run_pipeline(self, input_file: str = None, output_file: str = None, **kwargs) -> None:
        """
        运行完整评估
        主流程：
            1. 确定输入输出文件路径
            2. 加载输入数据
            3. 结构校验，拆分有效 / 异常数据，并保存异常数据与异常报告
            4. 对有效数据进行语义去重
            5. 按批次并发评估每条数据
            6. 周期性将中间结果写入临时文件
            7. 保存最终评估结果
            8. 生成统计分析与 HTML 可视化报告

        Args:
            input_file (str, optional): 输入文件路径（优先使用函数参数，其次配置中的 input_file）
            output_file (str, optional): 输出文件路径
            **kwargs:
                batch_size (int, optional): 批处理大小
        Returns:
            None
        """
        input_file = input_file or self.config.get('input_file')
        if not input_file:
            raise ValueError("未指定输入文件路径")

        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = Path(input_file)
            output_file = f"{input_path.stem}_enhanced_{timestamp}.json"

        batch_size = kwargs.get('batch_size') or self.config['evaluation']['batch_size']
        data = self.load_input_data(input_file)
        valid_data, invalid_data = self.validate_data_structure(data)
        if invalid_data:
            invalid_file = output_file.replace('.json', '_invalid.json')
            self.save_results(invalid_data, invalid_file)
            logger.info(f"保存异常数据至: {invalid_file} (共{len(invalid_data)}条)")
            from collections import Counter
            error_report = {
                'total_invalid': len(invalid_data),
                'error_types': Counter([item['error_type'] for item in invalid_data]),
                'sample_errors': invalid_data[:min(3, len(invalid_data))]
            }
            report_file = invalid_file.replace('.json', '_report.json')
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(error_report, f, indent=2, ensure_ascii=False)
                logger.info(f"异常数据报告已保存至: {report_file}")
            except Exception as e:
                logger.error(f"保存异常报告失败: {str(e)}", exc_info=True)
        deduplicated_data = self.deduplicate_data(valid_data)

        total_items = len(deduplicated_data)
        results = []
        with tqdm(
                total=total_items,
                desc="Evaluating",
                unit="item",
                ncols=100
        ) as pbar:
            for i in range(0, total_items, batch_size):
                batch_items = deduplicated_data[i:i + batch_size]
                batch_tasks = [self.evaluate_single_item(item) for item in batch_items]
                for coro in asyncio.as_completed(batch_tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)
                self.save_results(results, output_file + '.tmp')

        self.save_results(results, output_file)
        stats = self.analyze_results(results)

        self.save_statistics(stats, output_file, results)
        logger.info(f"统计分析报告已保存至: {output_file.replace('.json', '_statistics.json')}")

    def save_results(self, results: List[Dict], output_file: str) -> None:
        """保存评估结果
        Args:
            results (List[Dict]): 需要保存的结果列表
            output_file (str): 输出文件路径

        Returns:
            None
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            raise

    def analyze_results(self, results: List[Dict]) -> Dict:
        """
        对评估结果进行统计分析
        修改点：预先初始化所有可能的类别，确保计数为0的类别也能出现在统计结果中。
        """
        from collections import defaultdict
        import numpy as np

        if not results:
            logger.warning("无有效结果可分析")
            return {"error": "no_valid_results"}

        total_count = len(results)

        # ==========================================
        # 1. 定义所有固定类别 (Universe of Keys)
        # ==========================================

        # Task Types (您之前提供的完整列表)
        FIXED_TASK_TYPES = [
            "单表格问答", "单表多跳", "单表与单chunk（文本）问答", "单表与多chunk（文本）问答",
            "多表问答", "多表多跳", "多表与多chunk（文本）问答",
            "单chunk（文本）问答", "多chunk（文本）问答", "单chunk（文本）多跳问答", "多chunk（文本）多跳问答",
            "单公式问答", "单公式多跳", "单公式与单chunk（文本）问答", "单公式与多chunk（文本）问答",
            "多公式问答", "多公式多跳", "多公式与多chunk（文本）问答",
            "单chart问答", "单chart多跳", "单chart与单chunk（文本）问答", "单chart与多chunk（文本）问答",
            "多chart问答", "多chart多跳", "多chart与多chunk（文本）问答",
            "单表格&单chart", "单表格&单chart多跳", "单表格&单公式", "单表格&单公式多跳",
            "单chart&单公式", "单chart&单公式多跳"
        ]

        # Support Labels
        FIXED_SUPPORT_LABELS = ["支持", "不支持", "未评估"]

        # Difficulty Levels
        FIXED_DIFFICULTY_LEVELS = ["simple", "medium", "hard", "未评估"]

        # Complexity Scores (包含 0 作为未评估/错误)
        FIXED_COMPLEXITY_SCORES = [0, 1, 2, 3, 4, 5]

        # Token Buckets
        FIXED_TOKEN_BUCKETS = ["0-64", "65-128", "129-256", "257-512", "513+"]

        # ==========================================
        # 2. 初始化统计容器 (全部预填为 0)
        # ==========================================

        # 使用 defaultdict 防止数据中出现意外的新类别报错，但预先填入固定类别
        task_type_distribution = defaultdict(int, {k: 0 for k in FIXED_TASK_TYPES})
        support_distribution = defaultdict(int, {k: 0 for k in FIXED_SUPPORT_LABELS})
        difficulty_distribution = defaultdict(int, {k: 0 for k in FIXED_DIFFICULTY_LEVELS})
        complexity_distribution = defaultdict(int, {k: 0 for k in FIXED_COMPLEXITY_SCORES})
        token_bucket_distribution = defaultdict(int, {k: 0 for k in FIXED_TOKEN_BUCKETS})

        complexity_scores = []
        strong_correct_list = []
        weak_correct_list = []

        hop_distribution = defaultdict(int)  # Hops 不固定，动态统计
        hop_values = []

        token_lengths_total = []
        token_lengths_q = []
        token_lengths_a = []
        token_lengths_cot = []

        # 内部辅助函数：计算基本统计量
        def _safe_stats(values, cast_int=False):
            arr = np.array(values) if values else np.array([0])
            stats = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
            }
            if cast_int:
                stats["min"] = int(stats["min"])
                stats["max"] = int(stats["max"])
            return stats

        # 内部辅助函数：Token分桶
        def _token_bucket(length: int) -> str:
            if length <= 64:
                return "0-64"
            elif length <= 128:
                return "65-128"
            elif length <= 256:
                return "129-256"
            elif length <= 512:
                return "257-512"
            else:
                return "513+"

        # ==========================================
        # 3. 遍历数据进行统计
        # ==========================================
        for item in results:
            # Task Type
            t_type = item.get("task_type", "unknown")
            if "chunk(文本)" in t_type: t_type = t_type.replace("chunk(文本)", "chunk（文本）")
            task_type_distribution[t_type] += 1

            # Support
            s = item.get("support", {})
            s_label = s.get("label", "未评估")
            support_distribution[s_label] += 1

            # Difficulty
            d = item.get("difficult", {})
            d_level = d.get("level", "未评估")
            difficulty_distribution[d_level] += 1

            sc = d.get("strong_correct")
            wc = d.get("weak_correct")
            if sc is not None: strong_correct_list.append(1 if sc else 0)
            if wc is not None: weak_correct_list.append(1 if wc else 0)

            # Complexity
            c = item.get("complexity", {})
            score = c.get("score", 0) or 0
            complexity_scores.append(score)
            complexity_distribution[score] += 1

            # Hops
            num_hops = item.get("num_hops", 0)
            hop_distribution[num_hops] += 1
            hop_values.append(num_hops)

            # Token
            q = str(item.get("q", ""))
            a = str(item.get("a", ""))
            cot = str(item.get("cot", ""))

            q_len = len(q.split())
            a_len = len(a.split())
            cot_len = len(cot.split())
            total_len = q_len + a_len + cot_len

            token_lengths_q.append(q_len)
            token_lengths_a.append(a_len)
            token_lengths_cot.append(cot_len)
            token_lengths_total.append(total_len)

            bucket_label = _token_bucket(total_len)
            token_bucket_distribution[bucket_label] += 1

        # ==========================================
        # 4. 计算聚合指标并返回
        # ==========================================
        complexity_score_stats = _safe_stats(complexity_scores, cast_int=True)
        token_stats_total = _safe_stats(token_lengths_total, cast_int=True)
        token_stats_q = _safe_stats(token_lengths_q, cast_int=True)
        token_stats_a = _safe_stats(token_lengths_a, cast_int=True)
        token_stats_cot = _safe_stats(token_lengths_cot, cast_int=True)
        hop_stats = _safe_stats(hop_values, cast_int=True)

        strong_correct_rate = float(
            sum(strong_correct_list) / len(strong_correct_list)) if strong_correct_list else None
        weak_correct_rate = float(sum(weak_correct_list) / len(weak_correct_list)) if weak_correct_list else None

        support_count = support_distribution.get("支持", 0)
        not_support_count = support_distribution.get("不支持", 0)  # 直接从 dict 获取，不再通过减法计算

        return {
            "total_count": total_count,
            "task_type": {
                "distribution": dict(task_type_distribution)
            },
            "support": {
                "distribution": dict(support_distribution),
                "support_count": int(support_count),
                "not_support_count": int(not_support_count),
            },
            "difficulty": {
                "distribution": dict(difficulty_distribution),
                "strong_correct_rate": strong_correct_rate,
                "weak_correct_rate": weak_correct_rate,
            },
            "complexity": {
                "score": {
                    "values": complexity_scores,
                    "stats": complexity_score_stats,
                    "distribution": dict(complexity_distribution),
                }
            },
            "hops": {
                "distribution": dict(hop_distribution),
                "stats": hop_stats,
            },
            "token": {
                "total": {
                    "stats": token_stats_total,
                    "bucket_distribution": dict(token_bucket_distribution),
                },
                "q": {"stats": token_stats_q},
                "a": {"stats": token_stats_a},
                "cot": {"stats": token_stats_cot},
            },
        }

    def save_statistics(self, stats: Dict, output_file: str, results: List[Dict]) -> None:
        """
        保存统计分析结果
        Args:
            stats (Dict): 统计分析结果字典
            output_file (str): 主评估结果输出文件路径
            results (List[Dict]): 所有评估后的结果列表（用于生成可视化）
        Returns:
            None
        """
        stats_file = output_file.replace('.json', '_statistics.json')
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存统计结果失败: {str(e)}")
            raise

        self.generate_visual_report(stats, output_file, results)

    def generate_visual_report(self, stats: Dict, output_file: str, results: List[Dict]) -> None:
        """
        生成可视化 HTML 报告
        修改点：强制使用固定顺序展示图表，确保数量为0的类别也能显示。
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        report_file = output_file.replace('.json', '_report.html')
        total_count = stats.get("total_count", 0)
        avg_token_len = stats.get("token", {}).get("total", {}).get("stats", {}).get("mean", 0.0)

        # ==========================================
        # 1. 准备各图表数据 (使用固定顺序逻辑)
        # ==========================================

        # --- Helper: 按照固定顺序提取数据 ---
        def get_sorted_data(distribution_dict, fixed_order_list=None, sort_by_key=False):
            """
            如果有 fixed_order_list，则强制按该列表顺序返回 (Key, Value)。
            如果没有，则按 Key 字母序返回 (用于 task_type 等)。
            """
            if fixed_order_list:
                # 强制顺序，即使 dict 里没有（虽然 analyze_results 已经初始化了，但双重保险）
                labels = fixed_order_list
                values = [distribution_dict.get(k, 0) for k in labels]
                return labels, values
            else:
                # 默认排序：按 Key 排序 (sort_by_key=True) 或按 Value 排序
                # 这里默认按 Key 排序以便查阅
                items = sorted(distribution_dict.items(), key=lambda x: x[0])
                return [str(k) for k, v in items], [v for k, v in items]

        # 1. Task Type (数量较多，建议按定义的列表顺序或字母序)
        # 这里我们直接取 dict 里的 keys，因为我们在 analyze_results 里已经定义了 FIXED_TASK_TYPES
        # 为了美观，我们还是按 analyze_results 里的 FIXED_TASK_TYPES 顺序来展示
        FIXED_TASK_TYPES = [
            "单表格问答", "单表多跳", "单表与单chunk（文本）问答", "单表与多chunk（文本）问答",
            "多表问答", "多表多跳", "多表与多chunk（文本）问答",
            "单chunk（文本）问答", "多chunk（文本）问答", "单chunk（文本）多跳问答", "多chunk（文本）多跳问答",
            "单公式问答", "单公式多跳", "单公式与单chunk（文本）问答", "单公式与多chunk（文本）问答",
            "多公式问答", "多公式多跳", "多公式与多chunk（文本）问答",
            "单chart问答", "单chart多跳", "单chart与单chunk（文本）问答", "单chart与多chunk（文本）问答",
            "多chart问答", "多chart多跳", "多chart与多chunk（文本）问答",
            "单表格&单chart", "单表格&单chart多跳", "单表格&单公式", "单表格&单公式多跳",
            "单chart&单公式", "单chart&单公式多跳"
        ]
        task_dist = stats.get("task_type", {}).get("distribution", {})
        # 过滤掉 FIXED_TASK_TYPES 里有但 stats 里没有的key (虽然理论上已初始化)，
        # 或者 stats 里有但 FIXED 里没有的 (unknown类)
        # 这里简单处理：优先用 FIXED 顺序
        task_labels, task_values = get_sorted_data(task_dist, FIXED_TASK_TYPES)

        # 2. Support (固定顺序)
        support_order = ["支持", "不支持", "未评估"]
        support_dist = stats.get("support", {}).get("distribution", {})
        support_labels, support_values = get_sorted_data(support_dist, support_order)

        # 3. Difficulty (固定顺序: 简单 -> 困难)
        difficulty_order = ["simple", "medium", "hard", "未评估"]
        difficulty_dist = stats.get("difficulty", {}).get("distribution", {})
        difficulty_labels, difficulty_values = get_sorted_data(difficulty_dist, difficulty_order)

        # 4. Complexity (固定顺序: 0 -> 5)
        complexity_order = [0, 1, 2, 3, 4, 5]
        complexity_dist = stats.get("complexity", {}).get("score", {}).get("distribution", {})
        complexity_labels, complexity_values = get_sorted_data(complexity_dist, complexity_order)
        # 将 label 转为 string 方便显示
        complexity_labels = [str(l) for l in complexity_labels]

        # 5. Hops (按 0, 1, 2... 顺序)
        hops_dist = stats.get("hops", {}).get("distribution", {})
        # Hops 范围不确定，所以按 key (int) 排序
        hops_items = sorted(hops_dist.items(), key=lambda x: int(x[0]))
        hop_labels = [str(k) for k, v in hops_items]
        hop_values = [v for k, v in hops_items]

        # 6. Token Bucket (固定顺序)
        token_order = ["0-64", "65-128", "129-256", "257-512", "513+"]
        token_dist = stats.get("token", {}).get("total", {}).get("bucket_distribution", {})
        token_bucket_labels, token_bucket_values = get_sorted_data(token_dist, token_order)

        # ==========================================
        # 2. 创建绘图
        # ==========================================
        fig = make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=False,
            subplot_titles=(
                "Task Type Distribution",
                "Support Label Count",
                "Difficulty Level Count",
                "Complexity Score Count",
                "Num Hops Count",
                "Token Length Bucket Count (q+cot+a)",
            ),
            vertical_spacing=0.04,  # 稍微调小间距
        )

        # Row 1: Task Type
        fig.add_trace(
            go.Bar(
                x=task_values, y=task_labels, orientation="h",
                text=[str(v) for v in task_values], textposition="auto",
                name="TaskType", marker_color='#1f77b4'
            ), row=1, col=1
        )

        # Row 2: Support
        fig.add_trace(
            go.Bar(
                x=support_values, y=support_labels, orientation="h",
                text=[str(v) for v in support_values], textposition="auto",
                name="Support", marker_color='#2ca02c'
            ), row=2, col=1
        )

        # Row 3: Difficulty
        fig.add_trace(
            go.Bar(
                x=difficulty_values, y=difficulty_labels, orientation="h",
                text=[str(v) for v in difficulty_values], textposition="auto",
                name="Difficulty", marker_color='#ff7f0e'
            ), row=3, col=1
        )

        # Row 4: Complexity
        fig.add_trace(
            go.Bar(
                x=complexity_values, y=complexity_labels, orientation="h",
                text=[str(v) for v in complexity_values], textposition="auto",
                name="Complexity", marker_color='#d62728'
            ), row=4, col=1
        )

        # Row 5: Hops
        fig.add_trace(
            go.Bar(
                x=hop_values, y=hop_labels, orientation="h",
                text=[str(v) for v in hop_values], textposition="auto",
                name="NumHops", marker_color='#9467bd'
            ), row=5, col=1
        )

        # Row 6: Token
        fig.add_trace(
            go.Bar(
                x=token_bucket_values, y=token_bucket_labels, orientation="h",
                text=[str(v) for v in token_bucket_values], textposition="auto",
                name="TokenBucket", marker_color='#8c564b'
            ), row=6, col=1
        )

        # 设置布局
        # Task Type 比较多，第一张图可能需要稍微高一点，这里平均分配
        fig.update_layout(
            height=1800,  # 增加总高度
            width=1000,
            title_text=(
                f"QA Dataset Evaluation Overview - "
                f"Total: {total_count}, Avg Len: {avg_token_len:.1f}"
            ),
            showlegend=False,
            # 统一字体大小
            font=dict(size=12)
        )

        # 针对 Task Type 这种 Label 很长的图，调整 margin
        fig.update_layout(margin=dict(l=250))

        fig.write_html(report_file)
        logger.info(f"可视化报告已保存至: {report_file}")


async def main_async():
    """
    命令行入口（异步）

    命令行参数：
        --input / -i    : 输入文件路径
        --output / -o   : 输出文件路径
        --config / -c   : 配置文件路径
        --model / -m    : 覆盖配置中的 LLM 模型名称
        --api-key       : 覆盖配置中的 API 密钥
        --base-url      : 覆盖配置中的 API 基础 URL
        --backend       : LLM 后端类型 (openai, ollama, vllm, baidu, custom)
        --batch-size    : 批处理大小
    """
    parser = argparse.ArgumentParser(description="QA和CoT评估管道")
    parser.add_argument('--input', '-i', required=True, help='输入文件路径 (.json 或 .jsonl)')
    parser.add_argument('--output', '-o', help='输出文件路径（可选，自动生成）')
    parser.add_argument('--config', '-c', default='enhanced_evaluation_config.yaml', help='配置文件路径')
    parser.add_argument('--model', '-m', help='LLM模型名称')
    parser.add_argument('--api-key', help='API密钥')
    parser.add_argument('--base-url', help='API基础URL')
    parser.add_argument('--backend', help='LLM后端类型: openai, ollama, vllm, baidu, custom')
    parser.add_argument('--batch-size', type=int, help='批处理大小')
    args = parser.parse_args()

    try:
        pipeline = RefactoredEnhancedEvaluationPipeline(
            config_file=args.config,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        await pipeline.run_pipeline(
            input_file=args.input,
            output_file=args.output,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"评估流程失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
