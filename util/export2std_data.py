import json
import os

def convert_to_messages_format(item):
    """
    将单个数据项转换为messages格式
    
    Args:
        item (dict): 包含q, a, cot字段的字典
        
    Returns:
        dict: 包含messages字段的字典
    """
    messages = [
        {
            "role": "system",
            "content": "你是一个有用的助手，需要根据问题提供准确的回答。"
        },
        {
            "role": "user",
            "content": item["q"]
        },
        {
            "role": "assistant",
            "content": f"{item['cot']}\n\n答案：{item['a']}"
        }
    ]
    return {"messages": messages}

# {"system": "<system>", "conversation": [{"human": "<query1>", "assistant": "<response1>"}, {"human": "<query2>", "assistant": "<response2>"}]}
def convert_to_sharegpt_format(item):
    """
    将单个数据项转换为ShareGPT格式
    
    Args:
        item (dict): 包含q, a, cot字段的字典
        
    Returns:
        dict: 包含messages字段的字典
    """
    conversation = [
        {
            "human": item["q"],
            "assistant": f"{item['cot']}\n\n答案：{item['a']}"
        }
    ]
    messages = {
        "system": "你是一个有用的助手，需要根据问题提供准确的回答。",
        "conversation": conversation
    }
    return messages

# query-response
# {"system": "<system>", "query": "<query2>", "response": "<response2>", "history": [["<query1>", "<response1>"]]}
def convert_to_query_response_format(item):
    """
    将单个数据项转换为query-response格式
    
    Args:
        item (dict): 包含q, a, cot字段的字典
        
    Returns:
        dict: 包含query, response, history字段的字典
    """
    query_response = {
        "system": "你是一个有用的助手，需要根据问题提供准确的回答。",
        "query": item["q"],
        "response": f"{item['cot']}\n\n答案：{item['a']}",
        "history": []
    }
    return query_response

# alpaca
# {"system": "<system>", "instruction": "<query-inst>", "input": "<query-input>", "output": "<response>"}
def convert_to_alpaca_format(item):
    """
    将单个数据项转换为alpaca格式
    
    Args:
        item (dict): 包含q, a, cot字段的字典
        
    Returns:
        dict: 包含instruction, input, output字段的字典
    """
    alpaca = {
        "system": "你是一个有用的助手，需要根据问题提供准确的回答。",
        "instruction": item["q"],
        "input": "",
        "output": f"{item['cot']}\n\n答案：{item['a']}"
    }
    return alpaca

def convert_to_sft_x_format(data, format="messages"):
    """
    将数据转换为SFT格式
    
    Args:
        data (list): 包含q, a, cot字段的字典列表
        format (str): 输出格式，目前仅支持"messages"格式
        
    Returns:
        list: 包含messages字段的字典列表
    """
    converted_data = []
    for item in data:
        if format == "messages":
            messages = convert_to_messages_format(item)
        elif format == "sharegpt":
            messages = convert_to_sharegpt_format(item)
        else:
            raise ValueError(f"不支持的格式: {format}")
        converted_data.append(messages)
    return converted_data

def convert_to_std_format(input_path, output_path, mode="sft", format="messages"):
    """
    将输入的JSON数据转换为标准化的messages格式
    
    Args:
        input_path (str): 输入数据文件路径
        output_path (str): 输出数据文件路径
        mode (str): 转换模式，目前仅支持"sft"模式
    """
    # 读取输入数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换数据格式
    if mode == "sft":
        converted_data = convert_to_sft_x_format(data, format)
    else:
        raise ValueError(f"不支持的模式: {mode}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入输出文件（每行一个JSON对象）
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

