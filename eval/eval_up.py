from ast import arg
import json
import argparse
import pandas as pd
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
import asyncio
import jieba
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio

import sys
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有的LLM客户端
from llms.client import AsyncLLMClient


# 对中文进行分词
def tokenize_chinese(text: str) -> list[str]:
    words = jieba.cut(text)
    words = [word for word in words if word != ' ']
    return list(words)


class Evaluator:
    def __init__(self, model_configs: List[Dict]):
        self.model_configs = model_configs
        self.clients = self._init_clients()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def _init_clients(self) -> Dict[str, AsyncLLMClient]:
        clients = {}
        for config in self.model_configs:
            client = AsyncLLMClient(
                model=config['model'],
                api_key=config.get('api_key'),
                base_url=config.get('base_url')
            )
            clients[config.get('name', config['model'])] = client
        return clients

    def read_json_file(self, file_path: str) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据格式
            if not isinstance(data, list):
                raise ValueError("JSON文件内容必须是列表格式")
            
            # 验证每个元素是否包含必要字段
            required_fields = ['q', 'evidence', 'a']
            for i, example in enumerate(data):
                if not isinstance(example, dict):
                    raise ValueError(f"第{i+1}个元素不是字典格式")
                
                for field in required_fields:
                    if field not in example:
                        raise ValueError(f"第{i+1}个元素缺少必要字段: {field}")
            
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"JSON文件格式错误: {file_path}")

    def calculate_em(self, pred: str, gold: str) -> float:
        return 1.0 if pred.strip() == gold.strip() else 0.0

    def calculate_acc(self, pred: str, gold: str) -> float:
        pred = pred.strip().lower()
        gold = gold.strip().lower()
        if pred == gold:
            return 1.0
        elif gold in pred:
            return 1.0
        else:
            return 0.0
        # # 更准确的准确率计算，支持部分匹配
        # pred_tokens = pred.strip().lower().split()
        # gold_tokens = gold.strip().lower().split()
        
        # if not gold_tokens:
        #     return 0.0
        
        # common = set(pred_tokens) & set(gold_tokens)
        # # 计算精确匹配率
        # precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        # # 计算召回率
        # recall = len(common) / len(gold_tokens)
        # # 计算F1分数作为准确率
        # if precision + recall == 0:
        #     return 0.0
        # return 2 * (precision * recall) / (precision + recall)

    # 计算F1分数，支持中文和英文
    def calculate_f1(self, pred: str, gold: str) -> float:
        pred_tokens = tokenize_chinese(pred.strip().lower())
        gold_tokens = tokenize_chinese(gold.strip().lower())
        
        if not gold_tokens:
            return 0.0
        
        common = set(pred_tokens) & set(gold_tokens)
        # 计算精确匹配率
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        # 计算召回率
        recall = len(common) / len(gold_tokens)
        # 计算F1分数
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


    def calculate_rouge(self, pred: str, gold: str) -> Dict:
        gold = gold.strip().lower()
        pred = pred.strip().lower()
        gold = tokenize_chinese(gold)
        pred = tokenize_chinese(pred)
        gold = ' '.join(gold)
        pred = ' '.join(pred)
        scores = self.scorer.score(gold, pred)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    async def generate_prediction(self, client: AsyncLLMClient, query: str) -> str:
        try:
            messages = [
                {"role": "system", "content": "你是一个回答问题的助手，请根据提供的信息准确回答问题。"},
                {"role": "user", "content": query}
            ]
            response = await client.agenerate(messages)
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error generating prediction: {e}")
            # raise e
            return None

    async def evaluate_single_example(self, example: Dict, example_index: int, model_list = None) -> Tuple[str, str, Dict]:
        # 拼接q和evidence作为query
        q = """
        问题: {question}
        
        证据: {support}
        
        请根据以上证据，回答问题，答案应该是一个简短的实体词语。
        """
        template = """
现在交给你一个阅读理解的任务，给定一个question 以及对应的参考信息support，你需要基于support中给定的信息来回答question，并返回answer。
请注意：1、answer应该来自于support中的片段，不要发散。
2、请直接将answer按照json的格式返回，如{{"answer":xxxx}}。

question为：{question}
support为：{support}

你的答案是： /no_think
        """
        query = template.format(question=example['q'], support=example['evidence'])
        gold_answer = example['a']
        
        def match_answer(text):
            import re
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
                pattern = r'</think>\s*([^\n]+)'
                match = re.search(pattern, text)
                if match:
                    pred = match.group(1)
                    return pred
                return text
        predictions = {}
        for model_name, client in self.clients.items():
            if model_list and model_name not in model_list:
                continue
            pred = await self.generate_prediction(client, query)
            if pred is None:
                predictions[model_name] = None
                continue
            pred = pred.replace("```json", "").replace("```", "").strip()
            try:
                # print(pred)
                pred = json.loads(pred)['answer']
            except:
                pred = match_answer(pred) #"json parse error"

            predictions[model_name] = pred
        
        return example_index,query, gold_answer, predictions


    def get_example_id(self, example):
        return f"{example['model_name']}_{example['example_id']}"

    async def run_evaluation(self, input_file: str, output_file: str, required: Dict = None):
        data = self.read_json_file(input_file)
        # new_data = []
        # for d in data:
        #     if d.get('domain', 'unknown') == 'history_trace':
        #         new_data.append(d)
        self.start = 0
        total_examples = len(data)
        data = data[self.start:]
        new_data = data
        if required:
            for k, v in required.items():
                new_data = [d for d in new_data if d.get(k) in v]
            data = new_data
            print(f"After filtering, {len(data)} examples remain")
        
        
        # data = data[:8]

        #cache
        json_file = output_file if output_file.endswith('.json') else f"{output_file}.json"
        self.cache_result_map = {}
        self.cache_result = []
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                self.cache_result = json.load(f)
            
            for item in self.cache_result:
                self.cache_result_map[self.get_example_id(item)] = item

        tasks = []
        results = []
        for i, example in enumerate(data):
            i += self.start
            try:
                model_list = self.clients.keys()
                for model_name in model_list:
                    info ={
                        'model_name': model_name,
                        'example_id': i,
                    }
                    if self.cache_result_map.get(self.get_example_id(info), None) is not None:
                        continue
                    task = asyncio.create_task(self.evaluate_single_example(example, i, [model_name]))
                    tasks.append(task)
                
                # query, gold, preds = await self.evaluate_single_example(example, i, total_examples)
                
            except Exception as e:
                # 保存为json文件
                json_file = output_file if output_file.endswith('.json') else f"{output_file}.json"
                if os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                        with open(json_file+"_backup", 'w', encoding='utf-8') as f:
                            json.dump(existing_results, f, ensure_ascii=False, indent=4)
                        existing_results.extend(results)
                        # 去重，根据model_name和query字段
                        results = list({f"{item['model_name']}_{item['query']}": item for item in existing_results}.values())


                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"Evaluation results saved to {json_file}")

                print(f"Error processing example {i+1}: {e}")
                continue
        
        res = await tqdm_asyncio.gather(*tasks, desc="Processing examples")
        for idx, query, gold, preds in res:
            for model_name, pred in preds.items():
                if pred is None:
                    continue
                pred = str(pred)
                gold = str(gold)
                em = self.calculate_em(pred, gold)
                acc = self.calculate_acc(pred, gold)
                rouge = self.calculate_rouge(pred, gold)
                f1 = self.calculate_f1(pred, gold)
                
                results.append({
                    'example_id': idx,
                    'model_name': model_name,
                    'domain': data[idx].get('domain', 'unknown'),
                    'task_type': data[idx].get('task_type', 'unknown'),
                    'query': query,
                    'gold_answer': gold,
                    'prediction': pred,
                    'em': em,
                    'acc': acc,
                    'f1': f1,
                    **rouge
                })
        
        for item in results:
            self.cache_result_map[self.get_example_id(item)] = item
        results = list(self.cache_result_map.values())
        # 保存为json文件
        json_file = output_file if output_file.endswith('.json') else f"{output_file}.json"
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                with open(json_file+"_backup", 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=4)
                existing_results.extend(results)
                # 去重，根据model_name和query字段
                results = list({f"{item['model_name']}_{item['query']}": item for item in existing_results}.values())

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Evaluation results saved to {json_file}")

        # 保存结果到表格
        df = pd.DataFrame(results)
        
        # 保存为CSV格式
        csv_file = output_file if output_file.endswith('.csv') else f"{output_file}.csv"
        # 如果存在，则追加到最后
        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"Evaluation results saved to {csv_file}")
        
        # 尝试保存为Excel格式
        try:
            excel_file = output_file if output_file.endswith('.xlsx') else f"{output_file}.xlsx"
            df.to_excel(excel_file, index=False, engine='openpyxl')
            print(f"Evaluation results saved to {excel_file}")
        except ImportError:
            print("openpyxl not installed, skipping Excel export")

        # 计算并保存每个模型的平均指标
        if not df.empty:
            # 计算平均指标
            grouped = df.groupby('model_name').agg({
                'em': ['mean'],
                'acc': ['mean'],
                'f1': ['mean'],
                'rouge1': ['mean'],
                'rouge2': ['mean'],
                'rougeL': ['mean']
            }).reset_index()
            
            # 重命名列
            grouped.columns = ['model_name', 'em_mean', 'acc_mean', 'f1_mean',
                              'rouge1_mean', 'rouge2_mean', 
                              'rougeL_mean']
            
            # 打印总体指标
            print("\nOverall Evaluation Results (Mean):")
            for _, row in grouped.iterrows():
                print(f"\nModel: {row['model_name']}")
                print(f"  EM: {row['em_mean']:.4f}")
                print(f"  Accuracy: {row['acc_mean']:.4f}")
                print(f"  F1: {row['f1_mean']:.4f}")
                print(f"  ROUGE-1: {row['rouge1_mean']:.4f} ")
                print(f"  ROUGE-2: {row['rouge2_mean']:.4f} ")
                print(f"  ROUGE-L: {row['rougeL_mean']:.4f} ")
            
            # 保存平均指标到单独的文件
            mean_output_file = output_file.replace('.csv', '_mean.csv').replace('.xlsx', '_mean.xlsx')
            grouped.to_csv(mean_output_file.replace('.xlsx', '.csv'), index=False, encoding='utf-8-sig')
            print(f"\nMean evaluation results saved to {mean_output_file}")
        else:
            print("\nNo evaluation results to display")

model_config_list =[
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

def trans2tab(json_file: str, col_fields: str = 'task_type'):
    # 加载evaluation_results.json
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 转换为DataFrame
        df = pd.DataFrame(data)
        # 分组求平均值
        # 分组求平均值
    df = df.groupby(['model_name', col_fields])[['em', 'acc', 'f1', 'rouge1','rouge2', 'rougeL']].mean().reset_index()
    # 转换为透视表
    df = df.pivot(index='model_name', columns=col_fields, values=['em', 'acc', 'f1', 'rouge1','rouge2', 'rougeL'])
    # 输出df行名
    print(df.index)
    # 按model_config_list中name字段排序
    print([model['name'] for model in model_config_list])
    df = df.reindex([model['name'] for model in model_config_list])
    # 保存为Excel文件
    df.to_excel('mean_evaluation_results.xlsx')



def test(args):
    # parser = argparse.ArgumentParser(description='LLM Evaluation Script')
    # parser.add_argument('--input',  help='Input JSON file path')
    # parser.add_argument('--output', default='./result/llm/evaluation_results', help='Output CSV file path')
    # parser.add_argument('--models', required=False, help='Model configurations JSON string')
    
    # args = parser.parse_args()
    
    args.input = args['input'] # "processed_gpt.json"
    args.output = args['output'] # "./result/llm/evaluation_results"
    model_configs = args['model_configs']

    # 使用内置的模型配置
    # model_configs = model_config_list
    out_dir = os.path.dirname(args.output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    evaluator = Evaluator(model_configs)
    asyncio.run(evaluator.run_evaluation(args.input, args.output))
    json_file = args.output if args.output.endswith('.json') else f"{args.output}.json"
    trans2tab(json_file, col_fields='task_type')

def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation Script')
    parser.add_argument('--input',  help='Input JSON file path')
    parser.add_argument('--output', default='./result/llm/evaluation_results', help='Output CSV file path')
    parser.add_argument('--models', required=False, help='Model configurations JSON string')
    
    args = parser.parse_args()
    
    args.input = "processed_gpt.json"

    # 使用内置的模型配置
    model_configs = model_config_list
    out_dir = os.path.dirname(args.output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    evaluator = Evaluator(model_configs)
    asyncio.run(evaluator.run_evaluation(args.input, args.output))
    json_file = args.output if args.output.endswith('.json') else f"{args.output}.json"
    trans2tab(json_file, col_fields='task_type')

if __name__ == '__main__':
    main()