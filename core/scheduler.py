from typing import List, Deque
from collections import deque
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

    def schedule(self):
        while self.waiting:
            seq_group = self.waiting[0]

            seq = seq_group.get_seqs()[0]

            if self.block_manager.can_allocate(seq.get_len()):
                self.waiting.popleft()
                self.block_manager.allocate(seq.seq_id, seq.get_len())
                self.running.append(seq_group)
            else:
                break

        return self.running