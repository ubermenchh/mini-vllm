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

        self.model_executor = ModelExecutor(model_name, num_gpu_blocks, block_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eos_token_id = self.tokenizer.eos_token_id

        self.request_counter = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add_request(self, prompt: str) -> str:
        req_id = str(self.request_counter)
        self.request_counter += 1
        seq_id = int(req_id)

        token_ids = self.tokenizer.encode(prompt)

        seq = Sequence(seq_id, prompt, token_ids)
        group = SequenceGroup(req_id, [seq], arrival_time=0)

        self.scheduler.add_sequence_group(group)
        return req_id

    @torch.inference_mode()
    def step(self):
        running_groups = self.scheduler.schedule()
        if not running_groups:
            return {}

        first_seq = running_groups[0].get_seqs()[0]
        is_prefill = len(first_seq.output_token_ids) == 0

        input_ids_list = []
        position_ids_list = []
        context_lens_list = []
        block_tables_list = []

        for group in running_groups:
            seq = group.get_seqs()[0]
            
            if is_prefill:
                input_ids_list.append(seq.get_token_ids())
                position_ids_list.append(list(range(seq.get_len())))
            else:
                input_ids_list.append(seq.get_token_ids()[-1])
                position_ids_list.append(seq.get_len() - 1)
            context_lens_list.append(seq.get_len())
            block_tables_list.append(self.block_manager.get_block_table(seq.seq_id))

        max_num_blocks = max([len(block_table) for block_table in block_tables_list])
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        if is_prefill:
            # input_tensor: [batch_size, max_seq_len]
            input_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in input_ids_list],
                batch_first=True,
                padding_value=pad_id
            ).to(self.device)
            # position_tensor: [batch_size, max_seq_len]
            position_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in position_ids_list],
                batch_first=True,
                padding_value=pad_id
            ).to(self.device)
        else:
            # input_tensor: [batch_size, 1]
            input_tensor = torch.tensor(input_ids_list, device=self.device, dtype=torch.long).unsqueeze(1) 
            # position_tensor: [batch_size, 1]
            position_tensor = torch.tensor(position_ids_list, device=self.device, dtype=torch.long).unsqueeze(1)

        padded_block_tables = []
        for blocks in block_tables_list:
            num_pad = max_num_blocks - len(blocks)
            padded_block_tables.append(blocks + [-1] * num_pad)

        # context_lens_tensor: [batch_size]
        context_lens_tensor = torch.tensor(context_lens_list, device=self.device, dtype=torch.int64)   
        # block_tables_tensor: [batch_size, max_num_blocks]
        block_tables_tensor = torch.tensor(padded_block_tables, device=self.device, dtype=torch.int64) 

        # logits: [batch_size, seq_len, vocab_size]
        logits = self.model_executor.forward(input_tensor, position_tensor, context_lens_tensor, block_tables_tensor, is_prefill)

        outputs = {}
        for i, group in enumerate(running_groups):
            seq = group.get_seqs()[0]
    
            # logit_idx = (context_lens_list[i] - 1) if is_prefill else 0
            next_token_logits = logits[i, -1, :] # Shape: [vocab_size]
            # next_token_id = torch.argmax(next_token_logits).item()
            
            temperature = 0.8
            #
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            #
            # top_p = 0.9
            # sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            # cum_probs = torch.cumsum(sorted_probs, dim=-1)
            # sorted_indices_to_remove = cum_probs > top_p
            # sorted_indices_to_remove[0] = False # Keep the top token
            # sorted_probs[sorted_indices_to_remove] = 0
            # sorted_probs = sorted_probs / sorted_probs.sum()
            #
            # sampled_idx = torch.multinomial(sorted_probs, num_samples=1).item()
            # next_token_id = sorted_indices[sampled_idx].item()

            seq.append_token_id(next_token_id, 1.0)

            from core.sequence import SequenceStatus

            if next_token_id == self.eos_token_id:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.free(seq.seq_id)
            else:
                self.block_manager.append_slot(seq.seq_id, seq.get_len() - 1)

            outputs[group.request_id] = self.tokenizer.decode(seq.get_token_ids())

        return outputs
