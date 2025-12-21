import torch
from transformers import AutoModelForCausalLM


class ModelExecutor:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="cuda"
        )

    def forward(self, input_ids: torch.Tensor):
        outputs = self.model(input_ids=input_ids)
        return outputs.logits