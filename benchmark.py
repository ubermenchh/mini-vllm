import time
import os
import modal
from dotenv import load_dotenv

load_dotenv()

app = modal.App("mini-vllm-benchmark")

image = (
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

MODEL_NAME = "meta-llama/Llama-3.2-1B"
GPU_CONFIG = "A100"

@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.getenv("HF_TOKEN")})],
)
class BenchmarkEngine:
    @modal.enter()
    def load_model(self):
        from core.llm_engine import LLMEngine
        print(f"Loading model {MODEL_NAME}...")
        self.engine = LLMEngine(MODEL_NAME, num_gpu_blocks=5000)
        print("Model loaded.")

    @modal.method()
    def run_throughput_test(self, num_requests: int = 4, steps: int = 50):
        # 1. Add Requests
        prompt = "The meaning of life is"
        req_ids = []
        for i in range(num_requests):
            req_id = self.engine.add_request(prompt)
            req_ids.append(req_id)
        
        print(f"Added {num_requests} requests. Starting generation...")
        
        # 2. Measure Generation Loop
        # We run for a fixed number of steps to measure raw token throughput
        # disregarding EOS logic for this specific throughput test.
        
        start_time = time.time()
        total_tokens_generated = 0
        
        for _ in range(steps):
            # This processes one token for ALL active requests
            # In a real batch, this is 'num_requests' tokens per step
            outputs = self.engine.step()
            
            # Count how many requests actually generated a token this step
            # outputs contains {req_id: text} for all running groups
            total_tokens_generated += len(outputs)
            
            if not outputs:
                break

        end_time = time.time()
        duration = end_time - start_time
        
        tokens_per_sec = total_tokens_generated / duration
        
        return {
            "duration": duration,
            "total_tokens": total_tokens_generated,
            "tokens_per_sec": tokens_per_sec,
            "num_requests": num_requests
        }

@app.local_entrypoint()
def main():
    print("Starting Benchmark on Modal...")
    benchmark = BenchmarkEngine()
    
    # Run a few scenarios
    scenarios = [
        {"requests": 1, "steps": 50},
        {"requests": 4, "steps": 50},
        {"requests": 16, "steps": 50},
    ]
    
    for s in scenarios:
        print(f"\n--- Running Scenario: Batch Size {s['requests']} ---")
        stats = benchmark.run_throughput_test.remote(s['requests'], s['steps'])
        
        print(f"Duration: {stats['duration']:.4f}s")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Throughput: {stats['tokens_per_sec']:.2f} tokens/sec")
