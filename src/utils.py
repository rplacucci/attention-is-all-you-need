import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, x, target, target_mask):      # x: [B*T, V] log-probs, target: [B*T] indices
        # Only keep valid positions
        valid_idx = target_mask.bool()
        x = x[valid_idx]
        target = target[valid_idx]
        true_dist = torch.full_like(x, fill_value=self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        loss = self.criterion(x, true_dist.detach())
        # Normalize by number of non-padding tokens
        denom = valid_idx.sum().item()
        if denom > 0:
            loss = loss / denom
        return loss

def causal_mask(seq_len):
    attn_shape = (1, seq_len, seq_len)                             
    attn_mask = torch.tril(torch.ones(attn_shape, dtype=torch.bool))
    return attn_mask

def causal_shift(tgt, tgt_mask):
    tgt_x = tgt[:, :-1]
    tgt_y = tgt[:,  1:]
    tgt_x_mask = tgt_mask[:, :-1].to(torch.bool)
    tgt_y_mask = tgt_mask[:,  1:].to(torch.bool)

    T = tgt_x.size(1)
    causal = causal_mask(T).to(tgt.device)  # [1, T, T]
    tgt_x_mask = tgt_x_mask[:, None, None, :] & causal  # [batch, 1, T, T]

    return tgt_x, tgt_x_mask, tgt_y, tgt_y_mask
    
def greedy_decode(model, src, src_mask, max_len, sos_id=0):
    # Encode source
    memory = model.encode(src, src_mask)

    # init target sequence with <sos>
    tgt = torch.full((1, 1), sos_id, dtype=src.dtype, device=src.device)

    # greedy decode
    for _ in range(max_len - 1):
        tgt_mask = causal_mask(tgt.size(1)).to(src.device)
        out = model.decode(memory, src_mask, tgt, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_token = torch.max(prob, dim=1)
        next_token = next_token.item()
        next_token_tensor = torch.full((1, 1), next_token, dtype=src.dtype, device=src.device)
        tgt = torch.cat([tgt, next_token_tensor], dim=1)

    return tgt
