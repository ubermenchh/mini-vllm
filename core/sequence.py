from enum import Enum
from typing import List, Optional

class SequenceStatus(Enum):
    WAITING = 1
    RUNNING = 2
    FINISHED = 3

class Sequence:
    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
    ):
        self.seq_id = seq_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = []
        self.status = SequenceStatus.WAITING

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)

    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_token_ids(self):
        return self.prompt_token_ids + self.output_token_ids

class SequenceGroup:
    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        arrival_time: float
    ):
        self.request_id = request_id

        if len(seqs) != len(set(seq.seq_id for seq in seqs)):
            raise ValueError("Duplicate seq_id found in sequences")

        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.arrival_time = arrival_time

    def get_seqs(self, status: Optional[SequenceStatus]=None) -> List[Sequence]:
        if not status:
            return list(self.seqs_dict.values())
        return [seq for seq in self.seqs_dict.values() if seq.status == status]

    def is_finished(self) -> bool:
        return all([seq.status == SequenceStatus.FINISHED for seq in self.seqs_dict.values()])