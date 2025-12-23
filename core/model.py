from typing import Optional

import torch
from transformers import AutoModelForCausalLM


class ModelExecutor:
    def __init__(self, model_name: str, num_gpu_blocks: int, block_size: int):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
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

    def _init_kv_cache(self):
        self.kv_cache = [torch.zeros(
            2,
            self.num_gpu_blocks,
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
            device=self.model.device,
            dtype=torch.float16
        ) for _ in range(self.num_layers)]

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
