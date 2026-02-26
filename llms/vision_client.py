from math import log
import aiohttp
import asyncio
import base64
import mimetypes
import os
from typing import List, Dict, Any, Optional, Union
import requests
import logging
import json
import random
from util.errors import *

logger = logging.getLogger(__name__)

class AsyncVisionClient:
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/QVQ-72B-Preview",
        max_concurrent_requests: int = 5,
        max_tokens: int = 4096,
        max_retry: int = 3
    ):
        """
        异步视觉语言模型客户端
        功能：初始化异步视觉语言模型客户端，配置API连接参数和请求限制
        
        Args:
            base_url (str): API基础URL，默认为"https://api.openai.com/v1"
            api_key (str): OpenAI API密钥，默认为"EMPTY"
            model (str): 使用的视觉模型，默认为"Qwen/QVQ-72B-Preview"
            max_concurrent_requests (int): 最大并发请求数，默认为5
            max_tokens (int): 最大token数，默认为4096
            max_retry (int): 最大重试次数，默认为3
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        发送单个异步请求
        功能：通过aiohttp会话向指定端点发送POST请求并返回响应结果
        
        Args:
            session (aiohttp.ClientSession): aiohttp客户端会话对象
            endpoint (str): API端点路径
            payload (Dict[str, Any]): 请求载荷数据字典
            
        Returns:
            Dict[str, Any]: API响应结果字典
        """
        url = f"{self.base_url}/{endpoint}"
        async with self.semaphore:
            async with session.post(url, json=payload, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()

    
    def _process_image(
        self,
        image_source: Union[str, bytes]
    ) -> Dict[str, str]:
        """
        处理图片输入（网络URL或本地路径）
        功能：根据输入类型（网络URL、本地路径或字节数据）处理图片并转换为Base64编码格式
        
        Args:
            image_source (Union[str, bytes]): 图片URL、本地路径或字节数据
            
        Returns:
            Dict[str, str]: 包含图片类型和编码数据的字典
        """
        # 如果是网络URL
        if isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
            return {"type": "image_url", "image_url": {"url": image_source}}
        
        # 如果是本地文件路径
        if isinstance(image_source, str) and os.path.exists(image_source):
            # 获取MIME类型
            mime_type, _ = mimetypes.guess_type(image_source)
            if not mime_type:
                mime_type = "image/jpeg"  # 默认类型
            
            # 读取并编码图片
            with open(image_source, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            }
        
        # 如果是字节数据
        if isinstance(image_source, bytes):
            # 尝试猜测MIME类型（默认为jpeg）
            mime_type = "image/jpeg"
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64.b64encode(image_source).decode('utf-8')}"
                }
            }
        
        raise ValueError(f"不支持的图片输入类型")

    def _build_payload(
        self,
        system_prompt: str,
        prompt: str,
        image_content: Dict[str, Any],
        model: str = None,
        detail: str = "auto",
        history: List[Dict[str, str]] = [],
        kwargs: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        构建请求负载
        功能：构建发送给视觉语言模型的请求载荷，包含系统提示、用户提示、图片内容等信息
        
        Args:
            system_prompt (str): 系统提示文本
            prompt (str): 用户问题/提示文本
            image_content (Dict[str, Any]): 图片内容字典
            model (str): 模型名称，若为None则使用默认模型
            detail (str): 图片细节级别（"low", "high", "auto"）
            history (List[Dict[str, str]]): 对话历史记录列表
            kwargs (Dict[str, Any]): 其他可选参数字典
            
        Returns:
            Dict[str, Any]: 构建完成的请求载荷字典
        """
        model = model if model else self.model
        # 添加细节级别
        if "image_url" in image_content:
            image_content["image_url"]["detail"] = detail

        # 构造消息
        # his_list = []
        # for his in history:
        #     if  "image_url" in his["content"]:

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            },
        ]
        
        # 构造请求负载
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

        return payload
        
    async def agenerate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        image_sources: List[Union[str, bytes]] = [],
        model: str = None,
        detail: str = "auto",
        require_json: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        并发分析多张图片
        功能：并发处理多个图片分析请求，提高处理效率
        
        Args:
            messages_list (List[List[Dict[str, str]]]): 批量问题/提示列表
            image_sources (List[Union[str, bytes]]): 图片源列表（URL或本地路径）
            model (str): 模型名称，若为None则使用默认模型
            detail (str): 图片细节级别（"low", "high", "auto"）
            require_json (bool): 是否要求JSON格式响应
            **kwargs: 其他可选参数
            
        Returns:
            List[Dict[str, Any]]: API响应结果列表
        """
        if require_json:
            kwargs["response_format"] = {"type": "json_object"}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for prompt, image_source in zip(prompts, image_sources):
                # 处理图片输入
                image_content = self._process_image(image_source)
                
                # 构建请求负载
                payload = self._build_payload(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    image_content=image_content,
                    model=model,
                    detail=detail,
                    kwargs=kwargs,
                )
                
                # 创建任务
                task = asyncio.create_task(
                    self._send_request(session, "chat/completions", payload)
                )
                tasks.append(task)
            
            return await asyncio.gather(*tasks, return_exceptions=True)

    def generate(
        self,
        system_prompt: str = "",
        prompts: str = "",
        image_source: Union[str, bytes] = "",
        model: str = None,
        detail: str = "auto",
        require_json: bool = False,
        history: List[Dict[str, str]] = [],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        分析单张图片
        功能：同步方式分析单张图片，发送请求并返回模型分析结果
        
        Args:
            system_prompt (str): 系统提示文本
            prompts (str): 问题/提示文本
            image_source (Union[str, bytes]): 图片源（URL或本地路径）
            model (str): 模型名称，若为None则使用默认模型
            detail (str): 图片细节级别（"low", "high", "auto"）
            require_json (bool): 是否要求JSON格式响应
            history (List[Dict[str, str]]): 对话历史记录列表
            **kwargs: 其他可选参数
            
        Returns:
            List[Dict[str, Any]]: API响应结果字典
        """
        if require_json:
            kwargs["response_format"] = {"type": "json_object"}
        
        image_content = self._process_image(image_source)
        
        # 构建请求负载
        payload = self._build_payload(
            system_prompt=system_prompt,
            prompt=prompts,
            image_content=image_content,
            model=model,
            detail=detail,
            history=history,
            kwargs=kwargs,
        )
        
        # 发送请求
        response = requests.post(
            self.base_url + "/chat/completions",
            headers=self.headers,
            json=payload
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
        
    async def agenerate(
        self,
        system_prompt: str = "",
        prompts: str = "",
        image_source: Union[str, bytes] = "",
        model: str = None,
        detail: str = "auto",
        require_json: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        异步分析图片
        功能：异步方式分析单张图片，带有重试机制，确保请求成功
        
        Args:
            system_prompt (str): 系统提示文本
            prompts (str): 问题/提示文本
            image_source (Union[str, bytes]): 图片源（URL或本地路径）
            model (str): 模型名称，若为None则使用默认模型
            detail (str): 图片细节级别（"low", "high", "auto"）
            require_json (bool): 是否要求JSON格式响应
            **kwargs: 其他可选参数
            
        Returns:
            List[Dict[str, Any]]: API响应结果字典
        """
        if require_json:
            kwargs["response_format"] = {"type": "json_object"}
        
        image_content = self._process_image(image_source)
            
        # 构建请求负载
        payload = self._build_payload(
            system_prompt=system_prompt,
            prompt=prompts,
            image_content=image_content,
            model=model,
            detail=detail,
            kwargs=kwargs,
        )

        cur_retry = 0
        # 发送请求
        async with aiohttp.ClientSession() as session:
            while cur_retry < self.max_retry:
                cur_retry += 1
                try:
                    response = await self._send_request(session, "chat/completions", payload)
                    ret_text = response['choices'][0]['message']['content'].strip()
                    if require_json:
                        try:
                            json.loads(ret_text)
                        except Exception as e:
                            payload["temperature"] = random.uniform(0.0, 0.1)
                            payload["messages"][-2]["content"]+= "\n请确保输出格式为json且json格式正确。"
                            logger.error(f"Response is not a valid json: {str(e)},try ({cur_retry}/{self.max_retry})again")
                            logger.error(str(e))
                            logger.error(ret_text)
                            continue
                    return response
                except aiohttp.ClientError as e:
                    logger.error(f"Request failed: {str(e)}")
                    continue
                    # raise Exception(f"Request failed: {str(e)}")
                except asyncio.TimeoutError as e:
                    last_exception = e
                    logger.warning(f"请求超时{str(e)},maybe max_concurrent_requests is too large, causing packet loss")
                    continue
                except Exception as e:
                    raise Exception(f"Unexpected error: {str(e)}, img_url:{kwargs['test'], image_source}.")
        raise LLMRetry_Error(f"Max retry reached: {cur_retry}, remove the element from the list")


# 使用示例
async def test1():
    # 替换为你的OpenAI API密钥
    api_key = ""
    client = AsyncVisionClient(
        "https://api-inference.modelscope.cn/v1",
        api_key,
        max_concurrent_requests=3,
        max_tokens=500
    )
    
    # 准备图片和问题
    prompts = [
        "描述这张图片中的内容",
        "描述这张图片中的内容",
        "这张图片是什么风格？"
    ]
    
    # 图片源可以是：网络URL、本地路径或字节数据
    image_sources = [
        # "https://i-blog.csdnimg.cn/direct/d0ace29e2afc4a9eac158fae108b2584.png",  # 网络URL
        r"C:\Users\DELL\Desktop\myapp\data\llms\b.png", # 本地路径
        b"..."                            # 字节数据（实际使用中替换为真实图片数据）
    ]
    
    # 并发发送请求
    responses = await client.agenerate_batch(
        prompts=prompts,
        image_sources=image_sources,
        model = "Qwen/QVQ-72B-Preview",
        detail="auto"  # 高细节模式
    )
    
    # 处理响应
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"图片 {i+1} 分析错误: {str(response)}")
        else:
            content = response['choices'][0]['message']['content'].strip()
            print(f"图片 {i+1} 分析结果:\n{content}\n{'='*50}")


def test2():
    # 替换为你的OpenAI API密钥
    api_key = ""
    client = AsyncVisionClient(
        "https://api-inference.modelscope.cn/v1",
        api_key,
        max_concurrent_requests=3,
        max_tokens=500
    )
    
    # 准备图片和问题
    prompts = "描述这张图片中的内容"
    
    # 图片源可以是：网络URL、本地路径或字节数据
    image_source = "https://i-blog.csdnimg.cn/direct/d0ace29e2afc4a9eac158fae108b2584.png"  # 网络URL
    
    response = client.generate(
        prompts=prompts,
        image_source=image_source,
        model = "Qwen/QVQ-72B-Preview",
        detail="auto"  # 高细节模式
    )
    
    # 处理响应
    content = response['choices'][0]['message']['content'].strip()
    print(f"图片分析结果:\n{content}\n{'='*50}")


if __name__ == "__main__":
    import sys
    # if sys.platform == "win32":
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # asyncio.run(test1())
    test2()