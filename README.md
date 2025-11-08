# transformer-from-scratch

基于Transformer的词级莎士比亚文本生成模型，使用PyTorch实现。


## 硬件要求

- **最低配置**: 4GB RAM, 2GB GPU内存
- **推荐配置**: 8GB RAM, 4GB+ GPU内存
- **存储空间**: 至少500MB可用空间

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练模型

```bash
cd src
python -c "
import torch
torch.manual_seed(42)
from main import main
main()
"
```

## 生成文本
```bash
cd src
python generate.py
```
## 重现实验的精确命令行
```bash
# 设置环境变量
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

# 运行完整训练（包含随机种子设置）
python -c "
import torch
import random
import numpy as np

# 固定随机种子
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 导入并运行
from src.train import ImprovedTrainer
from src.model import ImprovedWordLevelTransformer  
from src.tokenizer import WordTokenizer
from src.dataset import ImprovedShakespeareDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 配置参数
config = {
    'vocab_size': 2000,
    'd_model': 128, 
    'n_heads': 8,
    'n_layers': 3,
    'max_seq_len': 64,
    'batch_size': 32,
    'lr': 5e-5,
    'epochs': 15,
    'dropout': 0.2
}

# 初始化组件
tokenizer = WordTokenizer(max_vocab_size=2000)
train_dataset = ImprovedShakespeareDataset(tokenizer, config['max_seq_len'], 'train')
val_dataset = ImprovedShakespeareDataset(tokenizer, config['max_seq_len'], 'validation')

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

model = ImprovedWordLevelTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=config['d_model'],
    n_heads=config['n_heads'], 
    n_layers=config['n_layers'],
    max_seq_len=config['max_seq_len'],
    dropout=config['dropout']
).to(device)

optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

# 训练
trainer = ImprovedTrainer(model, train_loader, val_loader, optimizer, device, tokenizer)
trainer.train(epochs=config['epochs'])

# 生成测试
from src.generate import comprehensive_generation_test
comprehensive_generation_test(model, tokenizer, device)
"
```
