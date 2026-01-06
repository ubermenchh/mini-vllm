import logging
import os
import time

import modal
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = modal.App("mini-vllm")

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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) # to avoid model download every time
    .add_local_dir("mini_vllm", remote_path="/root/mini_vllm")
)

# MODEL_NAME = "google/gemma-3-270m"
# MODEL_NAME = "openai-community/gpt2"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
GPU_CONFIG = "A100"

@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.getenv("HF_TOKEN")})],
)
class InferenceEngine:
    @modal.enter()
    def load_model(self):
        # import torch
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import AutoTokenizer

        from mini_vllm.llm_engine import LLMEngine

        logger.info(f"Loading model {MODEL_NAME}...")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_NAME,
        #     dtype=torch.bfloat16,
        #     device_map="auto"
        # )
        self.engine = LLMEngine(MODEL_NAME, num_gpu_blocks=5000)

        logger.info(f"Model loaded in {time.time() - t0:.2f}s")

    @modal.method()
    def generate(self, prompt: str):
        import time
        t0= time.time()

        # inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=20
        # )

        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        req_id = self.engine.add_request(prompt)

        final_text = ""
        token_count = 0
        while True:
            outputs = self.engine.step()
            token_count += 1
            if req_id in outputs:
                final_text = outputs[req_id]
                pass
            else:
                if final_text:
                    break
                if not outputs:
                    break
            
            if token_count > 256:
                break

        duration = time.time() - t0

        return {
            "text": final_text,
            "duration_sec": duration
        }

    @modal.method()
    def generate_batch(self, prompts: list):
        import time
        t0 = time.time()

        req_ids = []
        for prompt in prompts:
            req_ids.append(self.engine.add_request(prompt))

        current_texts = {req_id: "" for req_id in req_ids}
        finished = {}
        token_count = 0

        while len(finished) < len(req_ids):
            outputs = self.engine.step()
            token_count += 1

            for req_id in req_ids:
                if req_id in outputs:
                    current_texts[req_id] = outputs[req_id]
                elif req_id not in finished and current_texts[req_id]:
                    finished[req_id] = current_texts[req_id]

            if token_count > 256:
                for req_id in req_ids:
                    if req_id not in finished:
                        finished[req_id] = current_texts[req_id]
                break

        return {
            "texts": finished,
            "duration_sec": time.time() - t0
        }

@app.local_entrypoint()
def main():
    # Long context to trigger chunked prefill (>512 tokens)
    long_context = """Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future.

The primary aim is to allow computers to learn automatically without human intervention or assistance and adjust actions accordingly. Machine learning algorithms are often categorized as supervised or unsupervised. Supervised learning algorithms can apply what has been learned in the past to new data using labeled examples to predict future events.

Deep learning is a subset of machine learning that uses neural networks with many layers. These deep neural networks attempt to simulate the behavior of the human brain in processing data for use in decision making. Deep learning is especially useful for image recognition, natural language processing, and speech recognition.

Reinforcement learning is another paradigm where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. Unlike supervised learning, reinforcement learning does not require labeled input/output pairs and does not require sub-optimal actions to be explicitly corrected.

Transfer learning is a machine learning method where a model developed for one task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point for computer vision and natural language processing tasks.

Neural networks are computing systems inspired by biological neural networks that constitute animal brains. These systems learn to perform tasks by considering examples, generally without being programmed with task-specific rules. They consist of layers of interconnected nodes or neurons that process information using connectionist approaches to computation."""

    prompt = [
        f"{long_context}\n\nBased on the above context, explain what is gradient descent and how it is used in training neural networks. Answer:",
    ]
    
    logger.info(f"Prompt token count (approx): {len(prompt[0].split())}")

    engine = InferenceEngine()
    results = engine.generate_batch.remote(prompt)
    for req_id, text in results["texts"].items():
        logger.info(f"[{req_id}]: ...{text[-800:]}")  # Print last 800 chars

    # logger.info("--- Result ---")
    # logger.info(f"Output: {result['text']}")
    # logger.info(f"Time: {result['duration_sec']:.2f}")
