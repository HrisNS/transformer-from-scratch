import os
import re
import requests
from torch.utils.data import Dataset

class ImprovedShakespeareDataset(Dataset):
    def __init__(self, tokenizer, max_length=64, split='train'):
        if not os.path.exists('tinyshakespeare.txt'):
            self._download_dataset()
        
        with open('tinyshakespeare.txt', 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
        
        # 更好的句子分割
        sentences = self._split_into_sentences(text)
        
        # 构建词汇表
        if split == 'train':
            tokenizer.build_vocab(sentences)
        
        # 数据分割
        total_sentences = len(sentences)
        train_size = int(total_sentences * 0.9)
        if split == 'train':
            self.sentences = sentences[:train_size]
        else:
            self.sentences = sentences[train_size:]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"{split}集: {len(self.sentences)} 个句子")
    
    def _split_into_sentences(self, text):
        """改进的句子分割"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        return sentences
    
    def _download_dataset(self):
        print("下载Shakespeare数据集...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open('tinyshakespeare.txt', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("下载成功!")
        except Exception as e:
            print(f"下载失败: {e}")
            test_text = """
Now is the winter of our discontent.
Made glorious summer by this sun of York.
And all the clouds that lour'd upon our house.
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths.
Our bruised arms hung up for monuments.
Our stern alarums changed to merry meetings.
What news my lord? The king is dead.
So sudden? I have important news for you.
Speak clearly good citizen and tell me all.
We are accounted poor citizens but honest.
The patricians are good men and true.
What is the matter with this world?
To be or not to be that is the question.
Whether tis nobler in the mind to suffer.
The slings and arrows of outrageous fortune.
Or to take arms against a sea of troubles.
And by opposing end them to die to sleep.
"""
            with open('tinyshakespeare.txt', 'w', encoding='utf-8') as f:
                f.write(test_text)
            print("使用模拟数据")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.tokenizer.encode(sentence, max_length=self.max_length, padding=True, return_tensors='pt').squeeze(0)
        labels = input_ids.clone()
        return {'input_ids': input_ids, 'labels': labels}
