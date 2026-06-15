from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    # Optional debug
    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # TODO(student): yield RolloutBatch minibatches of size minibatch_size.
    # Requirements:
    # - Let N = batch.input_ids.shape[0] be the number of sampled completions.
    # - If shuffle=True, permute indices with torch.randperm using the provided generator.
    # - Otherwise iterate in the original order 0, 1, ..., N-1.
    # - Slice ALL tensor fields consistently with the same minibatch indices.
    # - Keep task_names / completion_texts aligned with the same indices when present.
    # - If device is not None, move the minibatch to that device before yielding.
    N = batch.input_ids.shape[0]
    if shuffle==True:
        indices = torch.randperm(N, generator=generator).tolist()
    else:
        indices = list(range(N))
    
    for start in range(0, N, minibatch_size):
        end = min(start + minibatch_size, N)
        mb_indices = indices[start:end]
        
        # 切片张量字段
        mb_input_ids = batch.input_ids[mb_indices]
        mb_attention_mask = batch.attention_mask[mb_indices]
        mb_completion_mask = batch.completion_mask[mb_indices]
        mb_old_logprobs = batch.old_logprobs[mb_indices]
        mb_ref_logprobs = batch.ref_logprobs[mb_indices]
        mb_rewards = batch.rewards[mb_indices]
        mb_advantages = batch.advantages[mb_indices]
        
        # 切片列表字段
        mb_task_names = [batch.task_names[i] for i in mb_indices]
        mb_completion_texts = [batch.completion_texts[i] for i in mb_indices]
        
        # 移动到指定设备（如果提供）
        if device is not None:
            mb_input_ids = mb_input_ids.to(device)
            mb_attention_mask = mb_attention_mask.to(device)
            mb_completion_mask = mb_completion_mask.to(device)
            mb_old_logprobs = mb_old_logprobs.to(device)
            mb_ref_logprobs = mb_ref_logprobs.to(device)
            mb_rewards = mb_rewards.to(device)
            mb_advantages = mb_advantages.to(device)
        
        # 创建新的 RolloutBatch 并 yield
        yield RolloutBatch(
            input_ids=mb_input_ids,
            attention_mask=mb_attention_mask,
            completion_mask=mb_completion_mask,
            old_logprobs=mb_old_logprobs,
            ref_logprobs=mb_ref_logprobs,
            rewards=mb_rewards,
            advantages=mb_advantages,
            task_names=mb_task_names,
            completion_texts=mb_completion_texts,
        )
    
        