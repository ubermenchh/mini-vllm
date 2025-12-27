import torch
import triton
import triton.language as tl


def paged_attention_v1(
    query: torch.Tensor,        # [num_seqs, num_heads, head_dim]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables: torch.Tensor, # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor, # [num_seqs]
    scale: float
) -> torch.Tensor:              # [num_seqs, num_heads, head_dim]
    
    num_seqs, num_heads, head_dim = query.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    out = torch.zeros(num_seqs, num_heads, head_dim, device=query.device, dtype=query.dtype)

    for i in range(num_seqs):
        context_len = context_lens[i]
        num_blocks = (context_len + block_size - 1) // block_size
        
        # block_idx: [num_used_blocks]
        block_idx = block_tables[i, :num_blocks]

        # keys/values: [num_used_blocks, num_kv_heads, block_size, head_dim]
        keys = key_cache[block_idx]      
        values = value_cache[block_idx]  
        
        # [num_blocks, heads, block_size, dim] -> [num_blocks * block_size, heads, dim]
        keys = keys.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        values = values.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)

        # keys/values: [context_len, num_kv_heads, head_dim]
        keys = keys[:context_len]
        values = values[:context_len]

        # q: [num_heads, 1, head_dim]
        q = query[i].unsqueeze(1)
        
        # k: [num_kv_heads, head_dim, context_len]
        k = keys.transpose(0, 1).transpose(1, 2)
        # v: [num_kv_heads, context_len, head_dim]
        v = values.transpose(0, 1)

        if num_kv_heads != num_heads: # For GQA
            # k/v: [num_heads, head_dim, context_len]
            k = torch.repeat_interleave(k, num_heads // num_kv_heads, dim=0)
            v = torch.repeat_interleave(v, num_heads // num_kv_heads, dim=0)

        # score: [num_heads, 1, context_len]
        score = torch.matmul(q, k) * scale
        attn = torch.softmax(score, dim=-1)

        # out[i]: [num_heads, head_dim]
        out[i] = torch.matmul(attn, v).squeeze(1)

    return out

@triton.jit
def _paged_attention_kernel(
    q_ptr,      # [num_seqs, num_heads, head_dim]
    k_ptr,      # [num_blocks, num_kv_heads, block_size, head_dim]
    v_ptr,      # [num_blocks, num_kv_heads, block_size, head_dim]
    block_ptr,  # [num_seqs, max_num_blocks]
    lens_ptr,   # [num_seqs]
    out_ptr,    # [num_seqs, num_heads, head_dim]
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_head, stride_k_dim, stride_k_x,
    stride_v_batch, stride_v_head, stride_v_dim, stride_v_x,
    stride_out_batch, stride_out_head, stride_out_dim,
    stride_b_batch, stride_b_block,
    num_heads,
    num_kv_heads,
    scale: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)

    offs_dim = tl.arange(0, head_dim)
    q_ptr_offset = (
        pid_batch * stride_q_batch + 
        pid_head * stride_q_head +
        offs_dim * stride_q_dim
    )
    q = tl.load(q_ptr + q_ptr_offset) # load query vector into SRAM

    m_i = -float("inf") # max score so far
    l_i = 0.0           # denominator of softmax
    acc = tl.zeros([head_dim], dtype=tl.float32)

    context_len = tl.load(lens_ptr + pid_batch)
    num_blocks = (context_len + block_size - 1) // block_size

    for block_idx in range(num_blocks):
        block_table_ptr = block_ptr + pid_batch * stride_b_batch + block_idx * stride_b_block
        physical_block_id = tl.load(block_table_ptr)

        offs_block = tl.arange(0, block_size)

        kv_head_idx = pid_head // (num_heads // num_kv_heads)

        k_ptr_base = k_ptr + physical_block_id * stride_k_batch + kv_head_idx * stride_k_head
        v_ptr_base = v_ptr + physical_block_id * stride_v_batch + kv_head_idx * stride_v_head

        k_offsets = offs_block[:, None] * stride_k_x + offs_dim[None, :] * stride_k_dim
        v_offsets = offs_block[:, None] * stride_v_x + offs_dim[None, :] * stride_v_dim

        k = tl.load(k_ptr_base + k_offsets)
        v = tl.load(v_ptr_base + v_offsets)

        score = tl.sum(q[None, :] * k, axis=1)
        score *= scale

        token_indices = block_idx * block_size + offs_block
        mask = token_indices < context_len
        score = tl.where(mask, score, -float("inf"))

        m_prev = m_i
        m_i = tl.maximum(m_i, tl.max(score, axis=0))

        p = tl.exp(score - m_i)

        l_i = l_i * tl.exp(m_prev - m_i) + tl.sum(p, axis=0)

        acc = acc * tl.exp(m_prev - m_i) + tl.sum(p[:, None] * v, axis=0)

    acc = acc / l_i

    out_ptr_offset = (
        pid_batch * stride_out_batch + 
        pid_head * stride_out_head +
        offs_dim * stride_out_dim
    )
    tl.store(out_ptr + out_ptr_offset, acc.to(q.dtype))

def paged_attention_triton(query, key_cache, value_cache, block_tables, context_lens, scale):
    num_seqs, num_heads, head_dim = query.shape
    num_blocks, num_kv_heads, block_size, _ = key_cache.shape

    output = torch.empty_like(query)

    grid = (num_seqs, num_heads)

    _paged_attention_kernel[grid](
        query, key_cache, value_cache, block_tables, context_lens,
        output,
        query.stride(0), query.stride(1), query.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(3), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(3), value_cache.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        num_heads, num_kv_heads, scale, block_size, head_dim
    )

    return output
