# transformer-from-scratch
精确运行命令（含随机种子）：

# 安装
pip install -r requirements.txt

# 训练
python src/train.py --dataset tiny_shakespeare --model_dim 128 --n_layers 2 --n_heads 4 \
    --batch_size 32 --seq_len 64 --lr 3e-4 --epochs 10 --seed 42 --save_dir results/


硬件要求：一块 GPU (推荐)，但小模型在 CPU 上也能跑（较慢）。

完整重现实验的 exact 命令行（上面即为 exact）。

说明随机性控制：torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)。
