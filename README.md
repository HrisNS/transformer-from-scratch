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
