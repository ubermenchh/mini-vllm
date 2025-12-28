import unittest
from unittest.mock import MagicMock, patch

import torch

from mini_vllm.llm_engine import LLMEngine


class MockConfig:
    num_attention_heads = 4
    num_hidden_layers = 1
    hidden_size = 16
    head_dim = 4
    num_key_value_heads = 4
    n_head = 4
    n_layer = 1
    n_embd = 16

class MockAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.float16)
        self.k_proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.float16)
        self.v_proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.float16)
        self.o_proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.float16)
        self.layer_idx = 0

class MockLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockAttention()
    
    def forward(self, hidden_states, *args, **kwargs):
        # We must call self_attn to trigger the patched method
        result = self.self_attn(hidden_states)
        if isinstance(result, tuple):
            return result[0]
        return result

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([MockLayer()])
        self.config = MockConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def forward(self, input_ids):
        # Create dummy embeddings with correct seq_len
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_states = torch.randn(batch_size, seq_len, 16, device=self.device, dtype=torch.float16)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        return hidden_states

class MockHFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockModel()
        self.config = MockConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def forward(self, input_ids):
        # Call the internal model
        hidden_states = self.model(input_ids)
        
        # Return logits as expected by ModelExecutor
        # Logits should have shape [batch, seq_len, vocab_size]
        batch_size, seq_len, _ = hidden_states.shape
        logits = torch.randn(batch_size, seq_len, 100, device=self.device, dtype=torch.float16)
        return MagicMock(logits=logits)

class TestPagedAttentionIntegration(unittest.TestCase):
    @patch("core.model.AutoModelForCausalLM")
    @patch("core.llm_engine.AutoTokenizer")
    def test_end_to_end_step(self, mock_tokenizer_cls, mock_model_cls):
        # 1. Setup HF Mocks
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_hf_model = MockHFModel().to(device).to(torch.float16)
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        
        mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
        # Prompt: 3 tokens [1, 2, 3]
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode.return_value = "generated"

        # 2. Init Engine (This triggers ModelExecutor init and PATCHING)
        # block_size=2, so 3 tokens = 2 blocks:
        # Block 0: indices 0, 1
        # Block 1: index 2
        engine = LLMEngine("dummy", block_size=2, num_gpu_blocks=10)
        
        # 3. Add Request (Prefill phase)
        engine.add_request("test prompt")
        
        # 4. Run Step 1 (Prefill)
        print("Running Prefill step...")
        with torch.no_grad():
            engine.step()
        
        # Verify Prefill cache writing
        seq_id = 0
        block_table = engine.block_manager.get_block_table(seq_id)
        kv_cache = engine.model_executor.kv_cache
        
        # Check Block 0 (Tokens 0 and 1)
        b0_idx = block_table[0]
        # Data at Block 0, offset 0 (Token 0)
        assert kv_cache[0][0][b0_idx, :, 0, :].abs().sum() > 0.0
        # Data at Block 0, offset 1 (Token 1)
        assert kv_cache[0][0][b0_idx, :, 1, :].abs().sum() > 0.0
        
        # Check Block 1 (Token 2)
        b1_idx = block_table[1]
        assert kv_cache[0][0][b1_idx, :, 0, :].abs().sum() > 0.0

        # 5. Run Step 2 (Decode)
        # Now context_len is 4 (3 prompt + 1 generated)
        print("Running Decode step...")
        with torch.no_grad():
            engine.step()
        
        # Check if 4th token (index 3) was written to Block 1, offset 1
        assert kv_cache[0][0][b1_idx, :, 1, :].abs().sum() > 0.0
        
        print("All assertions passed!")


if __name__ == "__main__":
    unittest.main()
