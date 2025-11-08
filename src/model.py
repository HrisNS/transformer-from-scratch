import torch
import torch.nn as nn

class ImprovedWordLevelTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=3, max_seq_len=64, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入 + 位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 更深的Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出层 + LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, labels=None):
        batch_size, seq_len = x.size()
        seq_len = min(seq_len, self.pos_embedding.num_embeddings)
        x = x[:, :seq_len]
        
        # 嵌入
        token_embeddings = self.embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.pos_embedding(positions)
        
        x = token_embeddings + position_embeddings
        
        # Transformer
        x = self.transformer(x)
        x = self.layer_norm(x)
        
        # 输出
        logits = self.output(x)
        
        loss = None
        if labels is not None:
            labels = labels[:, :seq_len]
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return {'logits': logits, 'loss': loss}
