from unittest.mock import patch
import torch
from core.llm_engine import LLMEngine

@patch("core.llm_engine.AutoTokenizer")
@patch("core.llm_engine.ModelExecutor")
def test_mixed_batch_bug(mock_model_cls, mock_tokenizer_cls):
    # Setup Mocks
    mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
    mock_tokenizer.encode.side_effect = lambda x: [1, 2, 3] if "A" in x else [4, 5, 6]
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode.return_value = "generated"

    # Mock Model Instance
    mock_model_instance = mock_model_cls.return_value
    
    # We want to capture the inputs to forward
    forward_inputs = []
    def mock_forward(input_ids, context_lens, block_tables, is_prefill):
        forward_inputs.append({
            "input_ids": input_ids,
            "is_prefill": is_prefill
        })
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        # Return dummy logits with correct shape
        return torch.randn(batch_size, seq_len, 100) 
        
    mock_model_instance.forward.side_effect = mock_forward

    engine = LLMEngine("dummy", block_size=16, num_gpu_blocks=10)

    # 1. Add Request A
    engine.add_request("Prompt A") # ids: [1, 2, 3]

    # 2. Step 1 (Prefill A)
    engine.step()
    
    print("Step 1 done")
    assert forward_inputs[-1]["is_prefill"]
    assert forward_inputs[-1]["input_ids"].shape == (1, 3) # Batch 1, Seq 3

    # 3. Add Request B
    engine.add_request("Prompt B") # ids: [4, 5, 6]

    # 4. Step 2 (Decode A + Prefill B)
    # This is where the bug should manifest
    engine.step()
    
    print("Step 2 done")
    last_call = forward_inputs[-1]
    
    # The engine takes is_prefill from the first group.
    # If Scheduler puts A first, is_prefill will be False.
    # If Scheduler puts B first, is_prefill will be True.
    # Scheduler uses deque and appends to running. 
    # A is already in running. B is added to running. So A is first.
    
    is_prefill_arg = last_call["is_prefill"]
    input_ids_arg = last_call["input_ids"]
    
    print(f"Is Prefill: {is_prefill_arg}")
    print(f"Input Shape: {input_ids_arg.shape}")
    print(f"Input IDs: {input_ids_arg}")

    # Expectation: 
    # is_prefill should be False (because A is first).
    # Request B should have been processed as prefill (3 tokens), but treated as decode (1 token).
    
    # Check Request B inputs (index 1)
    # B's tokens are [4, 5, 6]. 
    # If treated as decode, we only see [6].
    
    # input_ids_arg shape will be [2, 1] because is_prefill=False forces stack with unsqueeze(1) or similar logic in engine
    
    if not is_prefill_arg:
        # We expected mixed batch handling, but if engine forces False, B is broken.
        b_input = input_ids_arg[1]
        if len(b_input) == 1 and b_input[0] == 6:
            print("BUG CONFIRMED: Request B (prefill) was processed as decode (last token only).")
        else:
            print("Something else happened.")
    else:
        print("Unexpected: is_prefill was True. Did A finish?")

if __name__ == "__main__":
    test_mixed_batch_bug()
