
##  scripts/run.sh

```bash
#!/bin/bash

# Word-Level Transformer 训练脚本
# 设置随机种子以确保结果可重现

set -e  # 遇到错误立即退出

echo "=== Word-Level Transformer 训练脚本 ==="
echo "设置随机种子: 42"

# 设置Python路径
export PYTHONPATH=src

# 设置随机种子
python -c "
import torch
import random
import numpy as np

# 固定所有随机种子
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('随机种子设置完成')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
"

echo "开始训练..."
python -c "
from src.train import ImprovedTrainer
from src.model import ImprovedWordLevelTransformer
from src.tokenizer import WordTokenizer
from src.dataset import ImprovedShakespeareDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 配置
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

# 初始化
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
"

echo "训练完成!"
echo "开始生成测试..."

python -c "
from src.generate import comprehensive_generation_test
from src.model import ImprovedWordLevelTransformer
from src.tokenizer import WordTokenizer
from src.dataset import ImprovedShakespeareDataset
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'improved_results/best_model.pt'

if os.path.exists(model_path):
    # 重新加载模型进行生成测试
    tokenizer = WordTokenizer(max_vocab_size=2000)
    train_dataset = ImprovedShakespeareDataset(tokenizer, 64, 'train')
    
    model = ImprovedWordLevelTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=3,
        max_seq_len=64,
        dropout=0.2
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    comprehensive_generation_test(model, tokenizer, device)
else:
    print('模型文件不存在，跳过生成测试')
"

echo "=== 脚本执行完成 ==="
