"""
混合分块器
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ChunkConfig:
    """分块配置参数"""
    max_chunk_size: int = 1000  # 最大块大小（字符数）
    min_chunk_size: int = 200   # 最小块大小（字符数）
    separators: List[str] = None  # 文本分隔符
    overlap_size: int = 50      # 块间重叠大小
    
    def __post_init__(self):
        """
        初始化默认分隔符
        """
        if self.separators is None:
            # 默认分隔符：句号、问号、感叹号、换行符等
            self.separators = ['。', '？', '！', '\n\n', '\n', '；', '，']

class HybridChunker:
    """混合分块器"""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        初始化混合分块器
        
        Args:
            config (Optional[ChunkConfig]): 分块配置参数
        """
        self.config = config or ChunkConfig()
    
    def hybrid_chunking(self, text: str) -> List[Dict[str, Any]]:
        """
        执行混合分块策略
        
        Args:
            text (str): 输入文本
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个块包含文本和元数据
        """
        if not text.strip():
            return []
        
        # 1. 初始粗粒度分割：基于行分隔符
        coarse_chunks = self._coarse_segmentation(text)
        
        # 2. 混合拆分与合并操作
        final_chunks = self._refine_chunks(coarse_chunks)
        
        # 3. 添加块元数据
        chunks_with_metadata = self._add_metadata(final_chunks)
        
        return chunks_with_metadata
    
    def _coarse_segmentation(self, text: str) -> List[str]:
        """
        基于行分隔符进行初始粗粒度分割
        
        Args:
            text (str): 待分割的文本
            
        Returns:
            List[str]: 粗粒度分割后的文本块列表
        """
        # 首先按双换行符分割（段落级）
        paragraphs = re.split(r'\n\s*\n', text)
        
        coarse_chunks = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 如果段落长度合适，直接作为块
            if len(paragraph) <= self.config.max_chunk_size:
                coarse_chunks.append(paragraph)
            else:
                # 否则按单换行符进一步分割
                lines = paragraph.split('\n')
                current_chunk = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 如果当前块加上新行不会超限，则合并
                    if len(current_chunk) + len(line) + 1 <= self.config.max_chunk_size:
                        current_chunk += ("\n" + line) if current_chunk else line
                    else:
                        # 保存当前块并开始新块
                        if current_chunk:
                            coarse_chunks.append(current_chunk)
                        current_chunk = line
                
                if current_chunk:
                    coarse_chunks.append(current_chunk)
        
        return coarse_chunks
    
    def _refine_chunks(self, chunks: List[str]) -> List[str]:
        """
        细化分块：拆分过长块，合并过短块
        
        Args:
            chunks (List[str]): 待细化的文本块列表
            
        Returns:
            List[str]: 细化后的文本块列表
        """
        refined_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            # 如果块过长，进行递归拆分
            if len(chunk) > self.config.max_chunk_size:
                split_chunks = self._split_long_chunk(chunk)
                refined_chunks.extend(split_chunks)
            # 如果块过短，尝试与后续块合并
            elif len(chunk) < self.config.min_chunk_size:
                refined_chunks = self._merge_short_chunks(refined_chunks, chunk)
            else:
                refined_chunks.append(chunk)
        
        return refined_chunks
    
    def _split_long_chunk(self, chunk: str) -> List[str]:
        """
        使用用户定义的分隔符递归拆分过长块
        
        Args:
            chunk (str): 过长的文本块
            
        Returns:
            List[str]: 拆分后的文本块列表
        """
        # 按优先级尝试不同的分隔符
        for separator in self.config.separators:
            if separator in chunk:
                parts = chunk.split(separator)
                result_chunks = []
                current_part = ""
                
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue
                    
                    # 添加分隔符（除了最后一个部分）
                    separator_to_add = separator if i < len(parts) - 1 else ""
                    candidate = part + separator_to_add
                    
                    # 如果当前部分加上新部分不会超限
                    if len(current_part) + len(candidate) <= self.config.max_chunk_size:
                        current_part += candidate
                    else:
                        # 保存当前块
                        if current_part:
                            result_chunks.append(current_part)
                        current_part = candidate
                
                if current_part:
                    result_chunks.append(current_part)
                
                # 检查是否所有结果块都满足大小限制
                if all(len(c) <= self.config.max_chunk_size for c in result_chunks):
                    return result_chunks
        
        # 如果没有找到合适的分隔符，按最大大小强制分割
        return self._split_by_fixed_size(chunk)
    
    def _split_by_fixed_size(self, chunk: str) -> List[str]:
        """
        按固定大小分割文本（最后手段）
        
        Args:
            chunk (str): 待分割的文本
            
        Returns:
            List[str]: 按固定大小分割后的文本块列表
        """
        chunks = []
        start = 0
        chunk_length = len(chunk)
        
        while start < chunk_length:
            end = min(start + self.config.max_chunk_size, chunk_length)
            # 尽量在句子边界处分割
            split_point = self._find_good_split_point(chunk, start, end)
            sub_chunk = chunk[start:split_point].strip()
            if sub_chunk:
                chunks.append(sub_chunk)
            start = split_point
        
        return chunks
    
    def _find_good_split_point(self, text: str, start: int, end: int) -> int:
        """
        在指定范围内寻找合适的分割点
        
        Args:
            text (str): 待分割的文本
            start (int): 分割起始位置
            end (int): 分割结束位置
            
        Returns:
            int: 最佳分割点位置
        """
        if end >= len(text):
            return end
        
        # 优先在标点符号处分割
        for i in range(min(end, len(text) - 1), start, -1):
            if text[i] in ['。', '？', '！', '；', '，', '.', '?', '!', ';', ',']:
                return i + 1
        
        # 其次在空格处分割
        for i in range(min(end, len(text) - 1), start, -1):
            if text[i].isspace():
                return i + 1
        
        # 最后在字符边界处分割
        return end
    
    def _merge_short_chunks(self, chunks: List[str], new_chunk: str) -> List[str]:
        """
        合并过短的块
        
        Args:
            chunks (List[str]): 当前文本块列表
            new_chunk (str): 新的短文本块
            
        Returns:
            List[str]: 合并后的文本块列表
        """
        if not chunks:
            return [new_chunk] if new_chunk else []
        
        # 尝试与上一个块合并
        last_chunk = chunks[-1]
        if len(last_chunk) + len(new_chunk) <= self.config.max_chunk_size:
            merged = last_chunk + "\n" + new_chunk
            chunks[-1] = merged
            return chunks
        else:
            # 无法合并，添加为新块
            chunks.append(new_chunk)
            return chunks
    
    def _add_metadata(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        为每个块添加元数据
        
        Args:
            chunks (List[str]): 文本块列表
            
        Returns:
            List[Dict[str, Any]]: 包含元数据的文本块列表
        """
        result = []
        total_chars = sum(len(chunk) for chunk in chunks)
        
        for i, chunk in enumerate(chunks):
            chunk_info = {
                "src_id": i + 1,
                'text': chunk,
                'char_count': len(chunk),
                'word_count': len(chunk.split()),  # 简单分词
                'position_ratio': sum(len(c) for c in chunks[:i]) / total_chars if total_chars > 0 else 0,
                'is_optimal_size': self.config.min_chunk_size <= len(chunk) <= self.config.max_chunk_size
            }
            result.append(chunk_info)
        
        return result

# 使用示例
def main():
    # 示例文本
    text = """文本分块为了实现对数据采样精确控制并确保与现有大语言模型上下文窗口的兼容性，我们提出了混合分块（Hybrid-Chunking），这是一种结构感知且自适应的预处理策略，能够灵活地将源文档分割为连贯的文本块。该方法允许用户配置块大小并自定义文本分隔符，从而适应多种内容类型，如纯文本、代码片段或表格数据。该过程首先基于行分隔符进行初始粗粒度分割，随后执行混合的拆分与合并操作：对于过长的块，使用用户定义的分隔符递归拆分；而对于相邻的短段落，若需满足长度约束且不破坏语义单元，则进行合并。此外，针对自动化规则可能无法最优处理的边缘情况，我们提供了一个可视化文本分块界面，使用户能够进行细粒度的手动调整，确保文本分割的精确性。用户可根据特定文档内容灵活定制分块策略和阈值。这种混合设计在自动化与用户控制之间取得了平衡，显著提升了生成文本块的一致性和可靠性。"""

    # 配置分块参数
    config = ChunkConfig(
        max_chunk_size=300,  # 最大300字符
        min_chunk_size=100,  # 最小100字符
        separators=['。', '，', '；', '\n'],  # 中文友好分隔符
        overlap_size=20
    )
    
    # 创建分块器
    chunker = HybridChunker(config)
    
    # 执行分块
    chunks = chunker.hybrid_chunking(text)
    
    # 输出结果
    print(f"原始文本长度: {len(text)} 字符")
    print(f"分块数量: {len(chunks)}")
    print("\n分块结果:")
    print("-" * 50)
    
    for i, chunk_info in enumerate(chunks, 1):
        print(f"块 {chunk_info["src_id"]}:")
        print(f"字符数: {chunk_info['char_count']}")
        print(f"位置比例: {chunk_info['position_ratio']:.2%}")
        print(f"内容: {chunk_info['text'][:100]}...")  # 显示前100字符
        print("-" * 30)

if __name__ == "__main__":
    main()