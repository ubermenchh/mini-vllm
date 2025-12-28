# mini-vllm

A minimal implementation of vLLM's core ideas: PagedAttention and continuous batching.

## Installation

```bash
pip install mini-vllm
```
**Requirements**: Python 3.10+, CUDA-capable GPU

## Quick Start

```python
from mini_vllm import LLMEngine

# Initialize the engine
engine = LLMEngine(
    model_name="meta-llama/Llama-3.2-1B",
    block_size=16,
    num_gpu_blocks=100
)

# Add a request
req_id = engine.add_request("The meaning of life is")

# Generate tokens
while True:
    outputs = engine.step()
    if not outputs:
        break
    
    # Check if generation is complete
    if req_id in outputs:
        print(outputs[req_id])
```

## Benchmarks

**Hardware**: NVIDIA A100 (Modal)  
**Model**: `meta-llama/Llama-3.2-1B`  
**Max tokens per request**: 50  
**Prompt**: `"The meaning of life is"`

### mini-vllm Performance

| Batch Size | Duration | Total Tokens | Throughput |
|------------|----------|--------------|------------|
| 1          | 4.59s    | 50           | 10.90 tokens/sec |
| 4          | 1.01s    | 250          | 248.48 tokens/sec |
| 16         | 1.20s    | 1050         | 872.23 tokens/sec |

### Comparison with vLLM

| Batch Size | mini-vllm | vLLM | Ratio (vLLM/mini) |
|------------|-----------|------|-------------------|
| 1          | 10.90 tokens/sec | 213.73 tokens/sec | 19.6x |
| 4          | 248.48 tokens/sec | 977.46 tokens/sec | 3.9x |
| 16         | 872.23 tokens/sec | 3510.41 tokens/sec | 4.0x |


### References
- [vLLM repo](https://github.com/vllm-project/vllm)
- [vLLM paper](https://arxiv.org/abs/2309.06180)
- [continuous-batching-llm-inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [paged-attention-minimal](https://github.com/tspeterkim/paged-attention-minimal)
