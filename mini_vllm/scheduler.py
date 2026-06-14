import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

from mini_vllm.block_manager import BlockSpaceManager
from mini_vllm.sequence import SequenceGroup

BLOCK_SIZE = 16
logger = logging.getLogger(__name__)

@dataclass
class SchedulerOutput:
    scheduled_groups: List[SequenceGroup]
    num_prefill_tokens: Dict[int, int] = field(default_factory=dict)
    num_decode_tokens: Dict[int, int] = field(default_factory=dict)

class Scheduler:
    def __init__(self, block_manager: BlockSpaceManager):
        self.block_manager = block_manager
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []

    def add_sequence_group(self, seq_group: SequenceGroup):
        self.waiting.append(seq_group)

    def schedule(self, chunk_size: int=128) -> SchedulerOutput:
        output = SchedulerOutput(scheduled_groups=[])
        self.running = [g for g in self.running if not g.is_finished()]

        for seq_group in self.running:
            seq = seq_group.get_seqs()[0]
            if not seq.is_prefill_complete():
                chunk_tokens = seq.get_next_prefill_chunks(chunk_size)
                logger.info(f"CHUNK: seq={seq.seq_id}, start={seq.num_prefilled_tokens}, chunk_len={len(chunk_tokens)}, total_prompt={len(seq.prompt_token_ids)}")
                num_tokens = len(chunk_tokens)

                output.num_prefill_tokens[seq.seq_id] = num_tokens
                output.scheduled_groups.append(seq_group)

            else:
                output.num_decode_tokens[seq.seq_id] = 1
                output.scheduled_groups.append(seq_group)

        while self.waiting:
            seq_group = self.waiting[0]
            seq = seq_group.get_seqs()[0]

            if self.block_manager.can_allocate_request(seq.prompt_token_ids):
                self.waiting.popleft()
                self.block_manager.allocate_request(seq.seq_id, seq.prompt_token_ids)
                self.running.append(seq_group)
                
                first_chunk = seq.get_next_prefill_chunks(chunk_size)
                output.num_prefill_tokens[seq.seq_id] = len(first_chunk)
                output.scheduled_groups.append(seq_group)
            else:
                break

        return output
