from dataclasses import dataclass
from typing import Deque, Dict, List
import collections

BLOCK_SIZE = 16

@dataclass
class PhysicalTokenBlock:
    device: str
    block_num: int
    block_size: int
    ref_count: int = 0

class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int, device: str="cuda"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device

        self.free_blocks: Deque[PhysicalTokenBlock] = collections.deque()
        for i in range(num_blocks):
            self.free_blocks.append(PhysicalTokenBlock(device=device, block_num=i, block_size=block_size))

        self.all_blocks: Dict[int, PhysicalTokenBlock] = {
            b.block_num: b for b in self.free_blocks
        }

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks available")

        block = self.free_blocks.popleft()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock):
        if block.ref_count == 0:
            raise ValueError(f"Block {block.block_num} is already free")

        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


class BlockSpaceManager:
    def __init__(self, block_allocator: BlockAllocator):
        self.allocator = block_allocator
        # map: seq_id --> list of PhysicalTokenBlock
        self.block_tables: Dict[int, List[PhysicalTokenBlock]] = {}

    def allocate(self, seq_id: int, num_tokens: int):
        # free existing blocks if seq_id already exists
        if seq_id in self.block_tables:
            self.free(seq_id)

        # calculate the number of blocks needed to be allocated
        num_blocks = (num_tokens + self.allocator.block_size - 1) // self.allocator.block_size

        # allocate each block in the block table
        blocks = []
        try:
            for _ in range(num_blocks):
                blocks.append(self.allocator.allocate())
            self.block_tables[seq_id] = blocks
        except ValueError:
            for block in blocks:
                self.allocator.free(block)
            raise

    def append_slot(self, seq_id: int, current_num_tokens: int):
        # check if seq_id exists
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} not found in block tables")
        # if we have space in the previous block, do nothing
        if current_num_tokens % self.allocator.block_size != 0:
            return
        
        # allocate more blocks if there is no space
        self.block_tables[seq_id].append(self.allocator.allocate())

    def free(self, seq_id):
        # if seq_id is not in the block tables, do nothing
        if seq_id not in self.block_tables:
            return 

        # iterate and free the blocks in the block table
        for physical_block in self.block_tables[seq_id]:
            self.allocator.free(physical_block)

        # delete the entry from the dict as well
        del self.block_tables[seq_id]