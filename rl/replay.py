
# -*- coding: utf-8 -*-
"""
序列经验回放（用于 n-step 目标）
"""
import random
from typing import List
from dataclasses import dataclass

@dataclass
class Transition:
    s: object
    E_t: set
    r: float
    s_next: object
    done: bool

class ReplaySeq:
    def __init__(self, capacity:int=100_000):
        self.buf: List[Transition]=[]
        self.cap=capacity; self.idx=0

    def add(self, tr:Transition):
        if len(self.buf) < self.cap:
            self.buf.append(tr)
        else:
            self.buf[self.idx] = tr
            self.idx = (self.idx+1) % self.cap

    def sample_sequences(self, batch:int, n:int) -> List[List[Transition]]:
        out=[]
        L = len(self.buf)
        assert L > n+1, "replay too small"
        for _ in range(batch):
            i = random.randrange(0, L-n-1)
            seq = self.buf[i:i+n+1]
            out.append(seq)
        return out
