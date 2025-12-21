from core.block_manager import BlockAllocator, BlockSpaceManager
from core.scheduler import Scheduler
from core.sequence import Sequence, SequenceGroup


def test_simple_simulation_flow():
    block_size = 4
    num_blocks = 10
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    block_manager = BlockSpaceManager(allocator)
    scheduler = Scheduler(block_manager)

    seq = Sequence(seq_id=1, prompt="Hello World", prompt_token_ids=[101, 102])
    seq_group = SequenceGroup(request_id="req_1", seqs=[seq], arrival_time=0.0)

    scheduler.add_sequence_group(seq_group)
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 0

    running_groups = scheduler.schedule()

    assert len(running_groups) == 1
    assert running_groups[0] == seq_group
    assert len(scheduler.waiting) == 0

    assert allocator.get_num_free_blocks() == 9
    assert len(block_manager.block_tables[1]) == 1

    seq.append_token_id(201, 1.0)
    block_manager.append_slot(seq.seq_id, seq.get_len() - 1)

    seq.append_token_id(202, 1.0)

    seq.append_token_id(203, 1.0)

    block_manager.append_slot(seq.seq_id, 4)

    assert allocator.get_num_free_blocks() == 8
    assert len(block_manager.block_tables[1]) == 2

    print("Simulation Test Passed!!!")

if __name__=="__main__":
    test_simple_simulation_flow()