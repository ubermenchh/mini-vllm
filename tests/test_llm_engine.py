from unittest.mock import patch

import torch

from core.llm_engine import LLMEngine


# Mock the heavy dependencies
@patch("core.llm_engine.AutoTokenizer")
@patch("core.llm_engine.ModelExecutor")
def test_engine_step(mock_model_cls, mock_tokenizer_cls):
    # 1. Setup Mocks
    # Mock Tokenizer
    mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
    mock_tokenizer.encode.return_value = [1, 2, 3] # "Hello world"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode.return_value = "Hello world generated"
    
    # Mock Model
    mock_model_instance = mock_model_cls.return_value
    # Return random logits: [batch=1, seq_len=4, vocab=100]
    # We need to ensure the shape matches what step() expects
    def mock_forward(input_ids):
        batch, seq_len = input_ids.shape
        return torch.randn(batch, seq_len, 100) 
    
    mock_model_instance.forward.side_effect = mock_forward

    # 2. Initialize Engine
    # Use block_size=2 to test allocation easily
    engine = LLMEngine("dummy-model", block_size=2, num_gpu_blocks=10)
    
    # 3. Add Request
    req_id = engine.add_request("Hello world")
    
    # 4. Step 1
    outputs = engine.step()
    
    # 5. Verify
    assert req_id in outputs
    # Sequence length should increase from 3 to 4
    seq = engine.scheduler.running[0].get_seqs()[0]
    assert seq.get_len() == 4
    # Block manager should have been called
    # (3 tokens -> 2 blocks. 4 tokens -> 2 blocks. No new allocation yet).
    assert len(engine.block_manager.block_tables[int(req_id)]) == 2
    
    print("Test passed!")