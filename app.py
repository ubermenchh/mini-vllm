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
    # prompt = [
    #     "The meaning of life is to find",
    #     "The meaning of life is to give"
    # ]
    prompt = [
        "You are a helpful AI assistant. You always provide detailed, accurate, and thoughtful answers to questions. Please explain the concept of gravity.",
        "You are a helpful AI assistant. You always provide detailed, accurate, and thoughtful answers to questions. Please describe how plants grow.",
    ]
    logger.info(f"Sending prompt: '{prompt}'")

    engine = InferenceEngine()
    results = engine.generate_batch.remote(prompt)
    for req_id, text in results["texts"].items():
        logger.info(f"[{req_id}]: {text}")

    # logger.info("--- Result ---")
    # logger.info(f"Output: {result['text']}")
    # logger.info(f"Time: {result['duration_sec']:.2f}")
