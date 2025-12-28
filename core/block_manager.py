import collections
import logging
from dataclasses import dataclass
from typing import Deque, Dict, List

logger = logging.getLogger(__name__)

BLOCK_SIZE = 16

@dataclass
class PhysicalTokenBlock:
    device: str
    block_num: int
    block_size: int
    ref_count: int = 0

class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int=BLOCK_SIZE, device: str="cuda"):
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
        self.cached_blocks: Dict[int, PhysicalTokenBlock] = {}

    def _compute_block_hash(self, token_ids: List[int]) -> int:
        return hash(tuple(token_ids))

    def _is_block_cached(self, block: PhysicalTokenBlock) -> bool:
        return block in self.cached_blocks.values()

    def allocate_with_prefix_cache(self, seq_id: int, token_ids: List[int]):
        if seq_id in self.block_tables:
            self.free(seq_id)

        blocks = []
        block_size = self.allocator.block_size

        for i in range(0, len(token_ids), block_size):
            chunk = token_ids[i:i+block_size]

            if len(chunk) == block_size:
                content_hash = self._compute_block_hash(chunk)

                if content_hash in self.cached_blocks:
                    logger.info(f"CACHE HIT: block {i//block_size} for seq {seq_id}")
                    block = self.cached_blocks[content_hash]
                    block.ref_count += 1
                else:
                    logger.info(f"CACHE MISS: block {i//block_size} for seq {seq_id}")
                    block = self.allocator.allocate()
                    self.cached_blocks[content_hash] = block
            
                blocks.append(block)
            else:
                blocks.append(self.allocator.allocate())
        
        self.block_tables[seq_id] = blocks

    def allocate(self, seq_id: int, num_tokens: int):
        if seq_id in self.block_tables:
            self.free(seq_id)

        num_blocks = (num_tokens + self.allocator.block_size - 1) // self.allocator.block_size

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
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} not found in block tables")
        # if we have space in the previous block, do nothing
        if current_num_tokens % self.allocator.block_size != 0:
            return
        
        # allocate more blocks if there is no space
        self.block_tables[seq_id].append(self.allocator.allocate())

    def free(self, seq_id):
        if seq_id not in self.block_tables:
            return 

        # iterate and free the blocks in the block table
        for block in self.block_tables[seq_id]:
            if self._is_block_cached(block):
                block.ref_count -= 1
            else:
                self.allocator.free(block)

        # delete the entry from the dict as well
        del self.block_tables[seq_id]

    def can_allocate(self, num_tokens: int) -> bool:
        num_blocks_needed = (num_tokens + self.allocator.block_size - 1) // self.allocator.block_size
        return len(self.allocator.free_blocks) >= num_blocks_needed

    def can_allocate_with_cache(self, token_ids: List[int]) -> bool:
        block_size = self.allocator.block_size
        blocks_needed = 0

        for i in range(0, len(token_ids), block_size):
            chunk = token_ids[i:i+block_size]

            if len(chunk) == block_size:
                content_hash = self._compute_block_hash(chunk)
                if content_hash not in self.cached_blocks:
                    blocks_needed += 1
            else:
                blocks_needed += 1

        return self.allocator.get_num_free_blocks() >= blocks_needed

    def get_block_table(self, seq_id: int) -> List[int]:
        if seq_id not in self.block_tables:
            return []
        return [block.block_num for block in self.block_tables[seq_id]]
