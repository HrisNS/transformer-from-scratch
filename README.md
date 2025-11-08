# transformer-from-scratch

基于Transformer的词级莎士比亚文本生成模型，使用PyTorch实现。

## 项目结构

word-transformer-shakespeare/
├── src/ # 源代码
│ ├── tokenizer.py # 词级tokenizer
│ ├── model.py # Transformer模型定义
│ ├── dataset.py # 数据集处理
│ ├── train.py # 训练器类
│ └── generate.py # 文本生成函数
├── scripts/
│ └── run.sh # 运行脚本
├── requirements.txt # 依赖包
└── README.md # 项目说明


## 硬件要求

- **最低配置**: 4GB RAM, 2GB GPU内存
- **推荐配置**: 8GB RAM, 4GB+ GPU内存
- **存储空间**: 至少500MB可用空间

## 安装依赖

```bash
pip install -r requirements.txt
