import torch
import os

def comprehensive_generation_test(model, tokenizer, device):
    """综合生成测试"""
    print("\n" + "="*50)
    print("综合生成测试")
    print("="*50)
    
    test_cases = [
        {"prompt": "To be", "temp": 0.8, "penalty": 1.6},
        {"prompt": "The king", "temp": 0.9, "penalty": 1.5},
        {"prompt": "My lord", "temp": 1.0, "penalty": 1.4},
        {"prompt": "What is", "temp": 1.1, "penalty": 1.3},
        {"prompt": "I have", "temp": 0.7, "penalty": 1.7},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: '{case['prompt']}'")
        print(f"参数: temperature={case['temp']}, repetition_penalty={case['penalty']}")
        
        generated = smart_generate_text(
            model, tokenizer, device, case['prompt'],
            temperature=case['temp'],
            repetition_penalty=case['penalty']
        )
        print(f"生成结果: {generated}")
        print("-" * 40)

def smart_generate_text(model, tokenizer, device, prompt, max_length=25, temperature=0.9, top_p=0.95, repetition_penalty=1.3):
    """智能生成函数，专门解决重复问题"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_words = prompt.split()
    generated_tokens = input_ids[0].tolist()
    
    # 记录最近生成的词以避免重复
    recent_words = []
    
    with torch.no_grad():
        for i in range(max_length):
            current_input = input_ids
            if current_input.size(1) >= model.pos_embedding.num_embeddings:
                current_input = current_input[:, -model.pos_embedding.num_embeddings:]
            
            outputs = model(current_input)
            next_token_logits = outputs['logits'][:, -1, :]
            
            # 更强的重复惩罚
            for token_id in set(generated_tokens[-8:]):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repetition_penalty
                else:
                    next_token_logits[0, token_id] *= repetition_penalty
            
            # 特别惩罚连续重复
            if len(recent_words) >= 2 and len(set(recent_words[-2:])) == 1:
                last_word = recent_words[-1]
                if last_word in tokenizer.word_to_idx:
                    last_token = tokenizer.word_to_idx[last_word]
                    next_token_logits[0, last_token] = next_token_logits[0, last_token] / 3.0
            
            # 应用温度
            next_token_logits = next_token_logits / max(temperature, 0.1)
            
            # Nucleus (top-p) 采样
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[0, indices_to_remove] = -float('Inf')
            
            # 采样
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # 确保概率有效
            if torch.any(torch.isnan(probs)) or torch.sum(probs) == 0:
                probs = torch.softmax(torch.ones_like(next_token_logits), dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            next_word = tokenizer.decode(next_token[0].item())
            
            # 更新记录
            generated_words.append(next_word)
            generated_tokens.append(next_token.item())
            recent_words.append(next_word)
            if len(recent_words) > 4:
                recent_words.pop(0)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 智能停止条件
            if next_word in ['.', '!', '?'] and i > 5:
                break
            if len(generated_words) >= 4 and len(set(generated_words[-3:])) == 1:
                break
            if i >= 8 and len(set(generated_words[-4:])) == 1:
                break
    
    return ' '.join(generated_words)

def load_model_for_generation(model_path, device):
    """加载训练好的模型用于生成"""
    from src.model import ImprovedWordLevelTransformer
    from src.tokenizer import WordTokenizer
    
    # 加载tokenizer
    tokenizer = WordTokenizer(max_vocab_size=2000)
    
    # 重新构建词汇表（需要训练数据）
    from src.dataset import ImprovedShakespeareDataset
    train_dataset = ImprovedShakespeareDataset(tokenizer, 64, 'train')
    
    # 创建模型
    model = ImprovedWordLevelTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=3,
        max_seq_len=64,
        dropout=0.2
    ).to(device)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "improved_results/best_model.pt"
    
    if os.path.exists(model_path):
        model, tokenizer = load_model_for_generation(model_path, device)
        comprehensive_generation_test(model, tokenizer, device)
    else:
        print(f"模型文件 {model_path} 不存在，请先训练模型")
