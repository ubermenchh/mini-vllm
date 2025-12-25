import torch

def paged_attention_v1(
    query: torch.Tensor,        # [num_seqs, num_heads, head_dim]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables: torch.Tensor, # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor, # [num_seqs]
    scale: float
) -> torch.Tensor:              # Returns [num_seqs, num_heads, head_dim]
    
    num_seqs, num_heads, head_dim = query.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    out = torch.zeros(num_seqs, num_heads, head_dim, device=query.device, dtype=query.dtype)

    for i in range(num_seqs):
        context_len = context_lens[i]
        num_blocks = (context_len + block_size - 1) // block_size
        
        # 1. Fetch Block Indices for this sequence
        # block_idx: [num_used_blocks]
        block_idx = block_tables[i, :num_blocks]

        # 2. Gather Keys/Values from the Cache
        # keys/values: [num_used_blocks, num_kv_heads, block_size, head_dim]
        keys = key_cache[block_idx]      
        values = value_cache[block_idx]  
        
        # 3. Reshape to flat sequence
        # From [num_blocks, heads, block_size, dim] -> [num_blocks * block_size, heads, dim]
        keys = keys.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        values = values.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)

        # 4. Trim Padding/Garbage from the last block
        # keys/values: [context_len, num_kv_heads, head_dim]
        keys = keys[:context_len]
        values = values[:context_len]

        # 5. Prepare Query
        # q: [num_heads, 1, head_dim] (Add sequence dimension for matmul)
        q = query[i].unsqueeze(1)
        
        # 6. Transpose Keys/Values for Attention
        # k: [num_kv_heads, head_dim, context_len]
        k = keys.transpose(0, 1).transpose(1, 2)
        # v: [num_kv_heads, context_len, head_dim]
        v = values.transpose(0, 1)

        # 7. Handle GQA (Grouped Query Attention) Broadcasting
        if num_kv_heads != num_heads:
            # Repeat KV heads to match Query heads
            # k/v: [num_heads, head_dim, context_len]
            k = torch.repeat_interleave(k, num_heads // num_kv_heads, dim=0)
            v = torch.repeat_interleave(v, num_heads // num_kv_heads, dim=0)

        # 8. Compute Attention Scores
        # score: [num_heads, 1, context_len]
        score = torch.matmul(q, k) * scale
        attn = torch.softmax(score, dim=-1)

        # 9. Compute Output
        # out[i]: [num_heads, head_dim] (Squeeze back the sequence dimension)
        out[i] = torch.matmul(attn, v).squeeze(1)

    return out
