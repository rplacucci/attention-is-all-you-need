import torch

def beam_search_decode(
    model,
    src,
    src_mask,
    max_len,
    device,
    sos_token_id,
    eos_token_id,
    pad_token_id,
    beam_size=4,
    length_penalty_alpha=0.6,
):
    """
    Barebones beam search faithful to Vaswani et al.:
    - Beam size (default 4).
    - Google NMT length normalization with alpha=0.6.
    - No repetition penalties, no n-gram blocking, no forced min length.
    """
    def length_norm(L, alpha):
        # Google NMT style: ((5+L)/6)^alpha
        return ((5.0 + L) / 6.0) ** alpha

    def causal_mask(T, device):
        # [1, T, T] with True in lower triangle (allow attending to <= position)
        return torch.tril(torch.ones((T, T), dtype=torch.bool, device=device)).unsqueeze(0)

    batch_size = src.size(0)
    # Each item: (seq[tensor 1xT], sum_logprobs: float, finished: bool)
    beams = [
        [(torch.full((1, 1), sos_token_id, dtype=torch.long, device=device), 0.0, False)]
        for _ in range(batch_size)
    ]

    for _ in range(max_len):
        new_beams = []
        all_finished = True
        for i in range(batch_size):
            candidates = []
            for seq, raw_score, finished in beams[i]:
                if finished:
                    candidates.append((seq, raw_score, True))
                    continue

                all_finished = False
                T = seq.size(1)
                tgt_mask = causal_mask(T, device=device)

                with torch.no_grad():
                    logits = model(src[i:i+1], seq, src_mask[i:i+1].to(torch.bool), tgt_mask.to(torch.bool))
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                for k in range(beam_size):
                    next_token = topk_indices[k].item()
                    next_logp = topk_log_probs[k].item()
                    next_seq = torch.cat([seq, torch.tensor([[next_token]], device=device)], dim=1)
                    next_finished = (next_token == eos_token_id)
                    candidates.append((next_seq, raw_score + next_logp, next_finished))

            # Rank by normalized score (exclude <s> when computing length)
            def norm_key(h):
                seq_t, s, _ = h
                L_gen = max(1, seq_t.size(1) - 1)
                return s / length_norm(L_gen, length_penalty_alpha)

            candidates.sort(key=norm_key, reverse=True)
            new_beams.append(candidates[:beam_size])
        beams = new_beams
        if all_finished:
            break

    # Finalize: prefer finished; otherwise take the best by normalized score
    results = []
    for i in range(batch_size):
        finished = [h for h in beams[i] if h[2]]
        pool = finished if finished else beams[i]
        best = max(pool, key=lambda h: h[1] / length_norm(max(1, h[0].size(1) - 1), length_penalty_alpha))
        results.append(best[0].squeeze(0))

    # Pad to same length
    max_seq_len = max(seq.size(0) for seq in results)
    padded = []
    for seq in results:
        pad_len = max_seq_len - seq.size(0)
        if pad_len > 0:
            seq = torch.cat([seq, torch.full((pad_len,), pad_token_id, dtype=seq.dtype, device=seq.device)])
        padded.append(seq)
    return torch.stack(padded, dim=0)