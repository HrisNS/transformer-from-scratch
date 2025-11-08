import re
from collections import Counter
import torch

class WordTokenizer:
    def __init__(self, max_vocab_size=2000):
        self.vocab = ['<pad>', '<unk>', '<s>', '</s>']
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.max_vocab_size = max_vocab_size
        self.vocab_size = len(self.vocab)
    
    def build_vocab(self, texts):
        """从文本构建词汇表"""
        word_counts = Counter()
        
        for text in texts:
            # 简单的分词
            words = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
            word_counts.update(words)
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.max_vocab_size - len(self.vocab))
        
        for word, count in most_common:
            if word not in self.word_to_idx:
                idx = len(self.vocab)
                self.vocab.append(word)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.vocab)
        print(f"词汇表大小: {self.vocab_size}")
    
    def encode(self, text, max_length=None, padding=False, return_tensors=None):
        # 分词
        words = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
        
        # 转换为索引
        indices = [self.word_to_idx.get(word, self.unk_token_id) for word in words]
        
        if max_length:
            if len(indices) < max_length and padding:
                indices = indices + [self.pad_token_id] * (max_length - len(indices))
            elif len(indices) > max_length:
                indices = indices[:max_length]
        
        if return_tensors == 'pt':
            return torch.tensor([indices])
        return indices
    
    def decode(self, indices):
        # 处理单个索引的情况
        if isinstance(indices, int):
            if indices in self.idx_to_word:
                return self.idx_to_word[indices]
            else:
                return '<unk>'
        
        # 处理tensor
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        
        # 处理嵌套列表
        if isinstance(indices, list) and all(isinstance(i, list) for i in indices):
            indices = indices[0]
        
        # 处理单个索引的列表
        if isinstance(indices, list) and len(indices) == 1:
            idx = indices[0]
            if idx in self.idx_to_word:
                return self.idx_to_word[idx]
            else:
                return '<unk>'
        
        # 处理多个索引
        words = []
        for idx in indices:
            if idx == self.pad_token_id:
                continue
            if idx in self.idx_to_word:
                words.append(self.idx_to_word[idx])
            else:
                words.append('<unk>')
        
        return ' '.join(words)
