import os
import time

import modal
from dotenv import load_dotenv

load_dotenv()

app = modal.App("mini-vllm-benchmark")

# Image for mini-vllm
mini_vllm_image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "triton",
        "transformers",
        "huggingface-hub",
        "einops",
        "numpy<2.0.0",
        "accelerate",
        "python-dotenv",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("core", remote_path="/root/core")
    .add_local_dir("kernels", remote_path="/root/kernels")
)

# Image for vLLM
vllm_image = (
    modal.Image.debian_slim()
    .pip_install(
        "vllm",
        "python-dotenv",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_NAME = "meta-llama/Llama-3.2-1B"
GPU_CONFIG = "A100"
PROMPT = "The meaning of life is"


@app.cls(
    gpu=GPU_CONFIG,
    image=mini_vllm_image,
    timeout=600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.getenv("HF_TOKEN")})],
)
class MiniVLLMBenchmark:
    @modal.enter()
    def load_model(self):
        from core.llm_engine import LLMEngine
        print(f"[mini-vllm] Loading model {MODEL_NAME}...")
        self.engine = LLMEngine(MODEL_NAME, num_gpu_blocks=5000)
        print("[mini-vllm] Model loaded.")

    @modal.method()
    def run_throughput_test(self, num_requests: int = 4, steps: int = 50):
        req_ids = []
        for i in range(num_requests):
            req_id = self.engine.add_request(PROMPT)
            req_ids.append(req_id)
        
        start_time = time.time()
        total_tokens_generated = 0
        
        for _ in range(steps):
            outputs = self.engine.step()
            total_tokens_generated += len(outputs)
            if not outputs:
                break

        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "duration": duration,
            "total_tokens": total_tokens_generated,
            "tokens_per_sec": total_tokens_generated / duration,
            "num_requests": num_requests
        }


@app.cls(
    gpu=GPU_CONFIG,
    image=vllm_image,
    timeout=600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.getenv("HF_TOKEN")})],
)
class VLLMBenchmark:
    @modal.enter()
    def load_model(self):
        from vllm import LLM
        print(f"[vLLM] Loading model {MODEL_NAME}...")
        self.llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.9)
        print("[vLLM] Model loaded.")

    @modal.method()
    def run_throughput_test(self, num_requests: int = 4, max_tokens: int = 50):
        from vllm import SamplingParams
        
        # Create prompts for batch
        prompts = [PROMPT] * num_requests
        sampling_params = SamplingParams(
            temperature=0,  # Greedy to match mini-vllm
            max_tokens=max_tokens
        )
        
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        duration = end_time - start_time
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        
        return {
            "duration": duration,
            "total_tokens": total_tokens,
            "tokens_per_sec": total_tokens / duration,
            "num_requests": num_requests
        }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("BENCHMARK: mini-vllm vs vLLM")
    print("=" * 60)
    
    scenarios = [
        {"requests": 1, "steps": 50},
        {"requests": 4, "steps": 50},
        {"requests": 16, "steps": 50},
    ]
    
    mini_results = []
    vllm_results = []
    
    # Run mini-vllm benchmarks
    print("\n" + "=" * 60)
    print("Running mini-vllm benchmarks...")
    print("=" * 60)
    mini_benchmark = MiniVLLMBenchmark()
    
    for s in scenarios:
        print(f"\n--- mini-vllm: Batch Size {s['requests']} ---")
        stats = mini_benchmark.run_throughput_test.remote(s['requests'], s['steps'])
        mini_results.append(stats)
        print(f"Duration: {stats['duration']:.4f}s")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Throughput: {stats['tokens_per_sec']:.2f} tokens/sec")
    
    # Run vLLM benchmarks
    print("\n" + "=" * 60)
    print("Running vLLM benchmarks...")
    print("=" * 60)
    vllm_benchmark = VLLMBenchmark()
    
    for s in scenarios:
        print(f"\n--- vLLM: Batch Size {s['requests']} ---")
        stats = vllm_benchmark.run_throughput_test.remote(s['requests'], s['steps'])
        vllm_results.append(stats)
        print(f"Duration: {stats['duration']:.4f}s")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Throughput: {stats['tokens_per_sec']:.2f} tokens/sec")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Batch':<8} {'mini-vllm':<18} {'vLLM':<18} {'Ratio':<10}")
    print(f"{'Size':<8} {'(tokens/sec)':<18} {'(tokens/sec)':<18} {'(vLLM/mini)':<10}")
    print("-" * 60)
    
    for i, s in enumerate(scenarios):
        mini_tps = mini_results[i]['tokens_per_sec']
        vllm_tps = vllm_results[i]['tokens_per_sec']
        ratio = vllm_tps / mini_tps if mini_tps > 0 else 0
        print(f"{s['requests']:<8} {mini_tps:<18.2f} {vllm_tps:<18.2f} {ratio:<10.2f}x")
