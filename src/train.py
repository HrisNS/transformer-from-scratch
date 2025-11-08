import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class ImprovedTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, tokenizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.tokenizer = tokenizer
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']
            
            if not torch.isnan(loss) and loss > 1e-4:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, labels=labels)
                total_loss += outputs['loss'].item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs, save_path="improved_results"):
        os.makedirs(save_path, exist_ok=True)
        
        print("开始训练...")
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss
                }, f"{save_path}/best_model.pt")
                print(f"保存最佳模型，验证损失: {val_loss:.4f}")
            
            # 每2个epoch测试生成效果
            if epoch % 2 == 0:
                self.test_generation_during_training(epoch)
        
        self.plot_losses(save_path)
    
    def test_generation_during_training(self, epoch):
        """训练过程中的生成测试"""
        print(f"\n--- Epoch {epoch} 生成测试 ---")
        test_prompts = ["To be", "The king", "My lord"]
        
        for prompt in test_prompts:
            generated = smart_generate_text(
                self.model, self.tokenizer, self.device, prompt,
                temperature=0.9, repetition_penalty=1.5
            )
            print(f"  '{prompt}' -> '{generated}'")
    
    def plot_losses(self, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Improved Word-Level Transformer Training')
        plt.legend()
        plt.savefig(f"{save_path}/loss_curve.png")
        plt.show()

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
