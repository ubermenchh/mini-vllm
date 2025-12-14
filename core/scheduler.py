from dataclasses import dataclass
from typing import Deque, Dict
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