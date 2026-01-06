
import torch
from transformers import AutoModelForCausalLM

from mini_vllm.kernels.attention import paged_attention_triton


def rotate_half(x: torch.Tensor):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

class ModelExecutor:
    def __init__(self, model_name: str, num_gpu_blocks: int, block_size: int):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda"
        )

        self.num_heads = getattr(self.model.config, "num_attention_heads", getattr(self.model.config, "n_head", None))
        self.num_layers = getattr(self.model.config, "num_hidden_layers", getattr(self.model.config, "n_layer", None))
        self.hidden_size = getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", None))

        head_dim = getattr(self.model.config, "head_dim", None)
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim

        self.num_kv_heads = getattr(self.model.config, "num_key_value_heads", self.num_heads)

        self._init_kv_cache()

        if hasattr(self.model, "model"):
            self.layers = self.model.model.layers
        elif hasattr(self.model, "transformer"):
            self.layers = self.model.transformer.h
        else:
            raise ValueError("Could not find layers attribute in model")

        for i, layer in enumerate(self.layers):
            attn_module = None
            if hasattr(self.layers, "self_attn"):
                attn_module = layer.self_attn
            elif hasattr(self.layers, "attn"):
                attn_module = layer.attn

            if attn_module is None:
                if "self_attn" in layer._modules:
                    attn_module = layer._modules["self_attn"]
                elif "attn" in layer._modules:
                    attn_module = layer._modules["attn"]

            if not attn_module:
                raise ValueError(f"Unknown attention module. Available modules: {layer._modules.keys()}")

            attn_module.layer_idx = i
            object.__setattr__(attn_module, "forward", self._make_forward(attn_module))

    def _init_kv_cache(self):
        self.kv_cache = [torch.zeros(
            2,
            self.num_gpu_blocks,
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
            device=self.model.device,
            dtype=self.model.dtype
        ) for _ in range(self.num_layers)]

    def _make_forward(self, module):
        def forward_wrapper(hidden_states, *args, **kwargs):
            return self._paged_attention_forward(module, hidden_states, *args, **kwargs)
        return forward_wrapper

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, context_lens: torch.Tensor, 
                block_tables: torch.Tensor, is_prefill: bool=False, start_pos: torch.Tensor=None):
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.is_prefill = is_prefill
        self.start_pos = start_pos

        outputs = self.model(input_ids=input_ids, position_ids=position_ids)
        return outputs.logits

    def _chunked_attention(self, query, key, value, start_pos):
        head_dim = query.shape[-1]
        chunk_len = query.shape[2]
        kv_len = key.shape[2]

        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale

        mask = torch.ones(chunk_len, kv_len, device=query.device, dtype=torch.bool)
        for i in range(chunk_len):
            q_pos = start_pos + i
            mask[i, q_pos+1:] = False

        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(attn_weights, value)
        return output

    def _paged_attention_forward(self, module, hidden_states: torch.Tensor, *args, **kwargs):
        # hidden_states: [batch_size, seq_len, hidden_size]
        if hasattr(module, "c_attn"):
            # GPT2 style
            qkv = module.c_attn(hidden_states)
            query, key, value = qkv.split(self.head_dim * self.num_heads, dim=2)
        else:
            # Llama style
            query = module.q_proj(hidden_states)
            key = module.k_proj(hidden_states)
            value = module.v_proj(hidden_states)

        # Q: [batch_size, seq_len, num_heads * head_dim]
        # K: [batch_size, seq_len, num_kv_heads * head_dim]
        batch_size, seq_len, _ = hidden_states.shape
        executor = self

        # query: [batch_size, num_heads, seq_len, head_dim]
        # key:   [batch_size, num_kv_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, executor.num_heads, executor.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, executor.num_kv_heads, executor.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, executor.num_kv_heads, executor.head_dim).transpose(1, 2)

        rotary_emb = getattr(self.model.model, "rotary_emb", None)
        if rotary_emb is not None:
            pos_ids = kwargs.get("position_ids")
            if pos_ids is not None:
                # cos, sin: [batch_size, seq_len, head_dim]
                cos, sin = rotary_emb(value, pos_ids)
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
            else:
                pos_emb = kwargs.get("position_embeddings")
                if pos_emb is not None:
                    cos, sin = pos_emb
                    query, key = apply_rotary_pos_emb(query, key, cos, sin)

        layer_idx = module.layer_idx
        # k_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        k_cache = executor.kv_cache[layer_idx][0]
        v_cache = executor.kv_cache[layer_idx][1]

        attn_outputs = []

        if seq_len > 1: #executor.is_prefill:
            # PREFILL PHASE - Chunked
            for i in range(batch_size):
                start_pos = executor.start_pos[i].item()
                chunk_len = seq_len
                end_pos = start_pos + chunk_len
                block_table = executor.block_tables[i]

                if start_pos > 0:
                    cached_k_list = []
                    cached_v_list = []
                    for t in range(start_pos):
                        block_idx = block_table[t // executor.block_size]
                        offset = t % executor.block_size
                        # [num_kv_heads, head_dim]
                        cached_k_list.append(k_cache[block_idx, :, offset, :])
                        cached_v_list.append(v_cache[block_idx, :, offset, :])

                    # [start_pos, num_kv_heads, head_dim]
                    cached_k = torch.stack(cached_k_list, dim=0)
                    cached_v = torch.stack(cached_v_list, dim=0)

                    # [num_kv_heads, start_pos, head_dim]
                    cached_k = cached_k.permute(1, 0, 2)
                    cached_v = cached_v.permute(1, 0, 2)

                    # key[i] -> [num_kv_heads, ]
                    key_full = torch.cat([cached_k, key[i]], dim=1)
                    value_full = torch.cat([cached_v, value[i]], dim=1)
                else:
                    key_full = key[i]
                    value_full = value[i]

                if executor.num_kv_heads != executor.num_heads:
                    key_exp = torch.repeat_interleave(key_full.unsqueeze(0), executor.num_heads // executor.num_kv_heads, dim=1)
                    value_exp = torch.repeat_interleave(value_full.unsqueeze(0), executor.num_heads // executor.num_kv_heads, dim=1)
                else:
                    key_exp = key_full.unsqueeze(0)
                    value_exp = value_full.unsqueeze(0)

                q_i = query[i:i+1]

                attn_out = self._chunked_attention(q_i, key_exp, value_exp, start_pos)
                
                if i == 0:
                    attn_outputs = [attn_out]
                else:
                    attn_outputs.append(attn_out)

                k_chunk = key[i]
                v_chunk = value[i]
                actual_chunk_len = executor.context_lens[i].item() - start_pos
                for t in range(actual_chunk_len):
                    abs_pos = start_pos + t
                    block_idx = block_table[abs_pos // executor.block_size]
                    offset = abs_pos % executor.block_size
                    k_cache[block_idx, :, offset, :] = k_chunk[:, t, :]
                    v_cache[block_idx, :, offset, :] = v_chunk[:, t, :]
            
            attn_output = torch.cat(attn_outputs, dim=0)
            # attn_output: [batch_size, heads, seq_len, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            if hasattr(module, "c_proj"):
                output = module.c_proj(attn_output.view(batch_size * seq_len, -1))
            else:
                output = module.o_proj(attn_output.view(batch_size * seq_len, -1))
            return output.view(batch_size, seq_len, -1), None
        else:
            # DECODE PHASE: Read from cache
            # query: [batch_size, num_heads, head_dim]
            query = query.squeeze(2)
            key = key.squeeze(2)
            value = value.squeeze(2)

            last_token_indices = executor.context_lens - 1
            block_indices = executor.block_tables.gather(1, (last_token_indices // executor.block_size).unsqueeze(1).long()).squeeze(1)
            block_offsets = last_token_indices % executor.block_size

            # k_cache[block_idx, :, offset, :] = key
            k_cache[block_indices, :, block_offsets, :] = key
            v_cache[block_indices, :, block_offsets, :] = value

            scale = 1.0 / (executor.head_dim ** 0.5)
            
            # attn_output = paged_attention_v1(query, k_cache, v_cache, executor.block_tables, executor.context_lens, scale)
            attn_output = paged_attention_triton(query, k_cache, v_cache, executor.block_tables, executor.context_lens, scale)

            if hasattr(module, "c_proj"):
                output = module.c_proj(attn_output.view(batch_size, -1))
            else:
                output = module.o_proj(attn_output.view(batch_size, -1))
            return output.view(batch_size, 1, -1), None
