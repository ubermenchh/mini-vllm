from collections import deque
from typing import Deque, List

from core.block_manager import BlockSpaceManager
from core.sequence import SequenceGroup


class Scheduler:
    def __init__(self, block_manager: BlockSpaceManager):
        self.block_manager = block_manager
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []

    def add_sequence_group(self, seq_group: SequenceGroup):
        self.waiting.append(seq_group)

    def schedule(self) -> List[SequenceGroup]:
        self.running = [g for g in self.running if not g.is_finished()]

        while self.waiting:
            seq_group = self.waiting[0]
            seq = seq_group.get_seqs()[0]

            token_ids = seq.get_token_ids()

            if self.block_manager.can_allocate_request(token_ids):
                self.waiting.popleft()
                self.block_manager.allocate_request(seq.seq_id, token_ids)
                self.running.append(seq_group)
            else:
                break

        return self.running