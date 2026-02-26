from openai import OpenAI
from typing import List, Dict, Any
import numpy as np


class EmbeddingClient:
    def __init__(self, base_url="http://localhost:8000/v1", api_key="EMPTY", model="qwen3_embedding"):
        """
        初始化EmbeddingClient实例
        
        功能：创建一个用于获取文本嵌入向量(embeddings)的客户端
        
        Args:
            base_url (str): API服务的基础URL地址
            api_key (str): API访问密钥
            model (str): 使用的嵌入模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url, # 请确保端口与启动命令一致
        )

    def get_embedding(self, textlist: List[str], model: str = None)-> List[List[float]]:
        """
        获取文本列表的嵌入向量表示
        
        功能：将输入的文本列表转换为对应的嵌入向量列表，每个嵌入向量是一个浮点数数组
        
        Args:
            textlist (List[str]): 需要转换为嵌入向量的文本字符串列表
            model (str, optional): 指定使用的嵌入模型名称，如果未指定则使用初始化时设置的模型
            
        Returns:
            List[List[float]]: 返回与输入文本列表对应的嵌入向量列表，每个嵌入向量是numpy数组格式
        """
        model = model or self.model
        res = self.client.embeddings.create(
            model=self.model,  # 请确保与启动时--served-model-name一致
            input=textlist
        )
        embeddings = [np.array(data.embedding) for data in res.data]
        return embeddings


# if __name__ == "__main__":

#     texts = ["深度学习是一种机器学习方法。", "大语言模型近年来发展迅速。"]

#     client = OpenAI(
#         api_key="EMPTY",
#         base_url="http://localhost:8000/v1"  # 请确保端口与启动命令一致
#     )

#     response = client.embeddings.create(
#         model="qwen3_embedding",  # 请确保与启动时--served-model-name一致
#         input=texts
#     )

#     embeddings = [data.embedding for data in response.data]