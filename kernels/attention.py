import torch

def paged_attention_v1(
    query: torch.Tensor,        # [num_seqs, num_heads, head_dim]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables: torch.Tensor, # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor,
    scale: float
) -> torch.Tensor: # [num_seqs, num_heads, head_dim]
    num_seqs, num_heads, head_dim = query.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    out = torch.zeros(num_seqs, num_heads, head_dim, device=query.device, dtype=query.dtype)

    for i in range(num_seqs):
        context_len = context_lens[i]
        num_blocks = (context_len + block_size - 1) // block_size
        
        block_idx = block_tables[i, :num_blocks]

        keys = key_cache[block_idx]      # [num_used_blocks, num_kv_heads, block_size, head_dim]
        values = value_cache[block_idx]  # [num_used_blocks, num_kv_heads, block_size, head_dim]
        
        # Shape: [num_used_blocks * block_size, num_kv_heads, head_dim]
        keys = keys.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        values = values.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)

        # Shape: [context_len, num_kv_heads, head_dim]
        keys = keys[:context_len]
        values = values[:context_len]

        # query: [num_heads, head_dim] -> [num_heads, 1, head_dim]
        q = query[i].unsqueeze(1)
        # key: [num_kv_heads, head_dim, context_len]
        k = keys.transpose(0, 1).transpose(1, 2)
        # values: [num_kv_heads, context_len, head_dim]
        v = values.transpose(0, 1)

        if num_kv_heads != num_heads:
            k = torch.repeat_interleave(k, num_heads // num_kv_heads, dim=0)
            v = torch.repeat_interleave(v, num_heads // num_kv_heads, dim=0)

        # score: [num_heads, 1, context_len]
        score = torch.matmul(q, k) * scale
        attn = torch.softmax(score, dim=-1)

        # out: [num_heads, head_dim]
        out[i] = torch.matmul(attn, v).squeeze(1)

    return out
