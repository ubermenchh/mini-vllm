import torch
from transformers import AutoTokenizer

from core.block_manager import BlockAllocator, BlockSpaceManager
from core.model import ModelExecutor
from core.scheduler import Scheduler
from core.sequence import Sequence, SequenceGroup


class LLMEngine:
    def __init__(self, model_name: str, block_size: int=16, num_gpu_blocks: int=100):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks=num_gpu_blocks, block_size=block_size)
        self.block_manager = BlockSpaceManager(self.allocator)
        self.scheduler = Scheduler(self.block_manager)

        self.model_executor = ModelExecutor(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.request_counter = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add_request(self, prompt: str) -> str:
        # generate ids
        req_id = str(self.request_counter)
        self.request_counter += 1
        seq_id = int(req_id)

        # tokenize
        token_ids = self.tokenizer.encode(prompt)

        # create objects
        seq = Sequence(seq_id, prompt, token_ids)
        group = SequenceGroup(req_id, [seq], arrival_time=0)

        # add to scheduler
        self.scheduler.add_sequence_group(group)
        return req_id

    def step(self):
        # 1. Schedule
        running_groups = self.scheduler.schedule()
        if not running_groups:
            return {}

        # 2. Prepare Inputs (Naive Batching)
        all_token_ids = []
        for group in running_groups:
            seq = group.get_seqs()[0]
            all_token_ids.append(seq.get_token_ids())

        # max length for padding
        max_len = max([len(ids) for ids in all_token_ids])
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_inputs = []
        for tokens in all_token_ids:
            num_pad = max_len - len(tokens)

            padded_inputs.append(tokens + [pad_id] * num_pad)

        input_tensor = torch.tensor(padded_inputs, device=self.device, dtype=torch.long)

        # 3. Model Forward
        logits = self.model_executor.forward(input_tensor)

        # 4. Sample and update
        outputs = {}
        for i, group in enumerate(running_groups):
            seq = group.get_seqs()[0]

            last_idx = len(seq.get_token_ids()) - 1

            next_token_logits = logits[i, last_idx, :] # Shape: [vocab_size]

            next_token_id = torch.argmax(next_token_logits).item()

            seq.append_token_id(next_token_id, 1.0)

            self.block_manager.append_slot(seq.seq_id, seq.get_len() - 1)

            outputs[group.request_id] = self.tokenizer.decode(seq.get_token_ids())

        return outputs