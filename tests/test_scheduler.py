import pytest
from mini_vllm.block_manager import BlockAllocator, PhysicalTokenBlock

def test_allocator_initialization():
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    assert allocator.get_num_free_blocks() == 10
    
def test_allocate_single_block():
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    block = allocator.allocate()
    
    assert isinstance(block, PhysicalTokenBlock)
    assert block.block_num == 0
    assert block.ref_count == 1
    assert allocator.get_num_free_blocks() == 9

def test_allocate_all_blocks():
    allocator = BlockAllocator(num_blocks=2, block_size=16)
    allocator.allocate()
    allocator.allocate()
    
    assert allocator.get_num_free_blocks() == 0
    
    # Next allocation should fail
    with pytest.raises(ValueError, match="Out of memory"):
        allocator.allocate()

def test_free_block():
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    block = allocator.allocate()
    
    assert allocator.get_num_free_blocks() == 9
    
    allocator.free(block)
    assert block.ref_count == 0
    assert allocator.get_num_free_blocks() == 10
    
    # Re-allocating should give us the same block back (LIFO/FIFO depending on deque)
    # Your implementation uses popleft (FIFO) for allocate and append (end) for free
    # So it behaves like a queue.
    
def test_double_free_error():
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    block = allocator.allocate()
    
    allocator.free(block)
    
    with pytest.raises(ValueError, match="already free"):
        allocator.free(block)