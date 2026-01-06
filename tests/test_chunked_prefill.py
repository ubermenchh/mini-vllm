import pytest

from mini_vllm.block_manager import BlockAllocator, BlockSpaceManager
from mini_vllm.scheduler import Scheduler
from mini_vllm.sequence import Sequence, SequenceGroup


class TestChunkedPrefillScheduler:
    """Tests for chunked prefill scheduling logic."""

    def setup_method(self):
        """Setup fresh allocator, block_manager, and scheduler for each test."""
        self.allocator = BlockAllocator(num_blocks=100, block_size=16)
        self.block_manager = BlockSpaceManager(self.allocator, prefix_cache=False)
        self.scheduler = Scheduler(self.block_manager)

    def _create_sequence(self, seq_id: int, num_tokens: int) -> SequenceGroup:
        """Helper to create a sequence group with fake tokens."""
        fake_tokens = list(range(num_tokens))
        seq = Sequence(seq_id, f"prompt_{seq_id}", fake_tokens)
        return SequenceGroup(str(seq_id), [seq], arrival_time=0)

    def test_single_chunk_short_prompt(self):
        """Prompt shorter than chunk_size should complete in one prefill step."""
        group = self._create_sequence(seq_id=0, num_tokens=100)
        self.scheduler.add_sequence_group(group)

        output = self.scheduler.schedule(chunk_size=512)

        assert len(output.scheduled_groups) == 1
        assert 0 in output.num_prefill_tokens
        assert output.num_prefill_tokens[0] == 100  # All tokens in one chunk

    def test_two_chunks_for_long_prompt(self):
        """Prompt longer than chunk_size should require multiple prefill steps."""
        group = self._create_sequence(seq_id=0, num_tokens=1000)
        self.scheduler.add_sequence_group(group)
        seq = group.get_seqs()[0]

        # First schedule: chunk 1
        output1 = self.scheduler.schedule(chunk_size=512)
        assert output1.num_prefill_tokens[0] == 512
        assert seq.num_prefilled_tokens == 0  # Not updated yet by scheduler

        # Simulate llm_engine updating num_prefilled_tokens
        seq.num_prefilled_tokens += output1.num_prefill_tokens[0]
        assert seq.num_prefilled_tokens == 512

        # Second schedule: chunk 2
        output2 = self.scheduler.schedule(chunk_size=512)
        assert output2.num_prefill_tokens[0] == 488  # Remaining tokens

        seq.num_prefilled_tokens += output2.num_prefill_tokens[0]
        assert seq.num_prefilled_tokens == 1000
        assert seq.is_prefill_complete()

    def test_transition_to_decode_after_prefill(self):
        """After prefill completes, sequence should move to decode phase."""
        group = self._create_sequence(seq_id=0, num_tokens=100)
        self.scheduler.add_sequence_group(group)
        seq = group.get_seqs()[0]

        # Prefill step
        output1 = self.scheduler.schedule(chunk_size=512)
        assert 0 in output1.num_prefill_tokens
        seq.num_prefilled_tokens += output1.num_prefill_tokens[0]

        # Decode step (prefill complete)
        output2 = self.scheduler.schedule(chunk_size=512)
        assert 0 in output2.num_decode_tokens
        assert output2.num_decode_tokens[0] == 1
        assert 0 not in output2.num_prefill_tokens

    def test_multiple_sequences_with_chunks(self):
        """Multiple sequences should all get scheduled."""
        group1 = self._create_sequence(seq_id=0, num_tokens=600)
        group2 = self._create_sequence(seq_id=1, num_tokens=300)

        self.scheduler.add_sequence_group(group1)
        self.scheduler.add_sequence_group(group2)

        output = self.scheduler.schedule(chunk_size=512)

        # Both should be scheduled
        assert len(output.scheduled_groups) == 2
        assert 0 in output.num_prefill_tokens
        assert 1 in output.num_prefill_tokens
        assert output.num_prefill_tokens[0] == 512  # First chunk of seq 0
        assert output.num_prefill_tokens[1] == 300  # All of seq 1

    def test_block_allocation_incremental(self):
        """Blocks should be allocated incrementally per chunk."""
        group = self._create_sequence(seq_id=0, num_tokens=1000)
        self.scheduler.add_sequence_group(group)
        seq = group.get_seqs()[0]

        initial_free = self.allocator.get_num_free_blocks()

        # First chunk: 512 tokens = ceil(512/16) = 32 blocks
        output1 = self.scheduler.schedule(chunk_size=512)
        blocks_after_chunk1 = self.allocator.get_num_free_blocks()
        assert initial_free - blocks_after_chunk1 == 32

        seq.num_prefilled_tokens += output1.num_prefill_tokens[0]

        # Second chunk: 488 tokens = ceil(488/16) = 31 blocks
        # But some might fit in existing blocks, so we check incremental
        output2 = self.scheduler.schedule(chunk_size=512)
        blocks_after_chunk2 = self.allocator.get_num_free_blocks()

        # Total blocks for 1000 tokens = ceil(1000/16) = 63
        total_allocated = initial_free - blocks_after_chunk2
        assert total_allocated == 63

    def test_insufficient_blocks_for_chunk(self):
        """If not enough blocks, sequence should not be scheduled."""
        # Only 2 blocks available (can hold 32 tokens)
        small_allocator = BlockAllocator(num_blocks=2, block_size=16)
        small_block_manager = BlockSpaceManager(small_allocator, prefix_cache=False)
        scheduler = Scheduler(small_block_manager)

        group = self._create_sequence(seq_id=0, num_tokens=100)  # Needs 7 blocks
        scheduler.add_sequence_group(group)

        output = scheduler.schedule(chunk_size=512)

        # Should not be scheduled due to insufficient blocks
        assert len(output.scheduled_groups) == 0


class TestSequenceChunking:
    """Tests for Sequence chunking methods."""

    def test_get_next_prefill_chunks(self):
        """get_next_prefill_chunks should return correct slices."""
        tokens = list(range(1000))
        seq = Sequence(0, "test", tokens)

        # First chunk
        chunk1 = seq.get_next_prefill_chunks(chunk_size=512)
        assert chunk1 == list(range(512))
        assert len(chunk1) == 512

        # Simulate processing
        seq.num_prefilled_tokens = 512

        # Second chunk
        chunk2 = seq.get_next_prefill_chunks(chunk_size=512)
        assert chunk2 == list(range(512, 1000))
        assert len(chunk2) == 488

    def test_is_prefill_complete(self):
        """is_prefill_complete should track prefill progress."""
        tokens = list(range(100))
        seq = Sequence(0, "test", tokens)

        assert not seq.is_prefill_complete()

        seq.num_prefilled_tokens = 50
        assert not seq.is_prefill_complete()

        seq.num_prefilled_tokens = 100
        assert seq.is_prefill_complete()

    def test_get_num_computed_tokens(self):
        """get_num_computed_tokens should sum prefilled + output tokens."""
        tokens = list(range(100))
        seq = Sequence(0, "test", tokens)

        seq.num_prefilled_tokens = 100
        assert seq.get_num_computed_tokens() == 100

        seq.append_token_id(999, 1.0)
        assert seq.get_num_computed_tokens() == 101


class TestChunkedPrefillIntegration:
    """Integration tests simulating full chunked prefill flow."""

    def test_full_prefill_decode_cycle(self):
        """Simulate complete prefill (chunked) -> decode cycle."""
        allocator = BlockAllocator(num_blocks=100, block_size=16)
        block_manager = BlockSpaceManager(allocator, prefix_cache=False)
        scheduler = Scheduler(block_manager)

        # Create sequence with 1500 tokens (will need 3 chunks at 512)
        tokens = list(range(1500))
        seq = Sequence(0, "test", tokens)
        group = SequenceGroup("0", [seq], arrival_time=0)
        scheduler.add_sequence_group(group)

        prefill_steps = 0
        total_prefilled = 0

        # Run prefill chunks
        while not seq.is_prefill_complete():
            output = scheduler.schedule(chunk_size=512)
            assert len(output.scheduled_groups) == 1

            if 0 in output.num_prefill_tokens:
                chunk_size = output.num_prefill_tokens[0]
                seq.num_prefilled_tokens += chunk_size
                total_prefilled += chunk_size
                prefill_steps += 1

        assert prefill_steps == 3  # 512 + 512 + 476 = 1500
        assert total_prefilled == 1500
        assert seq.is_prefill_complete()

        # Now should be in decode mode
        output = scheduler.schedule(chunk_size=512)
        assert 0 in output.num_decode_tokens
        assert output.num_decode_tokens[0] == 1

    def test_mixed_prefill_decode_batch(self):
        """Test batch with some sequences prefilling, others decoding."""
        allocator = BlockAllocator(num_blocks=200, block_size=16)
        block_manager = BlockSpaceManager(allocator, prefix_cache=False)
        scheduler = Scheduler(block_manager)

        # Seq 0: Long prompt (needs multiple chunks)
        seq0 = Sequence(0, "test0", list(range(1000)))
        group0 = SequenceGroup("0", [seq0], arrival_time=0)
        scheduler.add_sequence_group(group0)

        # First schedule for seq0
        output1 = scheduler.schedule(chunk_size=512)
        seq0.num_prefilled_tokens += output1.num_prefill_tokens[0]

        # Seq 1: Short prompt (single chunk, will complete prefill)
        seq1 = Sequence(1, "test1", list(range(100)))
        group1 = SequenceGroup("1", [seq1], arrival_time=0)
        scheduler.add_sequence_group(group1)

        # Second schedule: seq0 continues prefill, seq1 starts
        output2 = scheduler.schedule(chunk_size=512)
        
        # seq0 should continue prefilling (488 more tokens)
        assert 0 in output2.num_prefill_tokens
        assert output2.num_prefill_tokens[0] == 488

        # seq1 should prefill all 100 tokens
        assert 1 in output2.num_prefill_tokens
        assert output2.num_prefill_tokens[1] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

