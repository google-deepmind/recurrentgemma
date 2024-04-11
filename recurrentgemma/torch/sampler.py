# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A sampler for a Griffin model."""
from collections.abc import Sequence
from typing import NamedTuple
import jaxtyping as jt

from recurrentgemma.torch import array_typing as at
import recurrentgemma.torch.griffin as griffin_lib
import torch

import sentencepiece as spm


@at.typed
class _SamplingState(NamedTuple):
  """Internal sampling state.

  Attributes:
    decoding_step: The number of the current decoding step.
    num_input_tokens: The number of tokens in the prompt.
    token_buffer: Fixed-size buffer for accumulating the output tokens.
    cache: Model state for conditioning the model on autoregressively.
    done: Whether decoding is done on the current sequence.
    total_sampling_steps: Total number of sampling steps including the prompt.
    logits_buffer: Fixed-size buffer for accumulating the output logits.
  """

  decoding_step: jt.Integer[torch.Tensor, ""]
  positions: jt.Integer[torch.Tensor, "*b 1"]
  token_buffer: at.Tokens
  cache: griffin_lib.Cache
  done: jt.Bool[torch.Tensor, "*b"]
  total_sampling_steps: jt.Integer[torch.Tensor, ""]
  logits_buffer: at.TokenLogits | None = None


class SamplerOutput(NamedTuple):
  """Output of the sampler.

  Attributes:
    text: Decoded samples from the model.
    logits: Per-step logits used during sampling.
    tokens: Tokens corresponding to the generated samples.
  """

  text: list[str]
  logits: list[list[float]]
  tokens: list[list[int]]


class Sampler:
  """Sampler for a Griffin model."""

  def __init__(
      self,
      model: griffin_lib.Griffin,
      vocab: spm.SentencePieceProcessor,
  ):
    """Initializes a sampler for a Griffin model.

    Args:
      model: An instance of the Griffin model.
      vocab: Vocabulary of the model.
    """
    self.model = model
    self.vocab = vocab
    self._eos_token = torch.tensor([self.vocab.eos_id()], device=self.device)

  @property
  def dtype(self) -> torch.dtype:
    return next(self.model.parameters()).dtype

  @property
  def device(self) -> torch.device:
    return next(self.model.parameters()).device

  @at.typed
  def _sample_step(self, sampler_state: _SamplingState) -> _SamplingState:
    """Performs a single sampling step.

    Args:
      sampler_state: The current state of the sampler.

    Returns:
      The updated sampler state.
    """
    decoding_step = sampler_state.decoding_step
    last_token = sampler_state.token_buffer[:, decoding_step]
    last_token = last_token[:, None]

    logits, cache = self.model(
        last_token,
        sampler_state.positions + decoding_step,
        sampler_state.cache,
    )

    next_token_candidate = torch.argmax(logits, axis=-1)  # [B, 1]
    next_token_candidate = next_token_candidate[:, 0]  # [B,]

    sampler_state.token_buffer[:, decoding_step + 1] = next_token_candidate

    if sampler_state.logits_buffer is not None:
      next_logits = torch.squeeze(logits, 1)
      sampler_state.logits_buffer[:, decoding_step] = next_logits

    done_now = torch.equal(
        sampler_state.token_buffer[:, decoding_step + 1], self._eos_token
    )

    return _SamplingState(
        decoding_step=decoding_step + 1,
        token_buffer=sampler_state.token_buffer,
        logits_buffer=sampler_state.logits_buffer,
        cache=cache,
        positions=sampler_state.positions,
        done=sampler_state.done | done_now,
        total_sampling_steps=sampler_state.total_sampling_steps,
    )

  @at.typed
  def init_cache(self, batch_size: int) -> griffin_lib.Cache:
    """Initializes an empty cache for the model."""
    return self.model.init_cache(batch_size, dtype=self.dtype)

  @at.typed
  def init_sample_state(
      self,
      total_sampling_steps: int,
      last_token: jt.Integer[torch.Tensor, "*b"],
      positions: jt.Integer[torch.Tensor, "*b 1"],
      cache: griffin_lib.Cache,
      include_logits: bool = False,
  ) -> _SamplingState:
    """Initializes the sampling state given input prompts."""
    buffer_size = total_sampling_steps + 1
    batch_size = last_token.shape[0]
    token_buffer = torch.full(
        (batch_size, buffer_size),
        self.vocab.pad_id(),
        dtype=torch.int32,
        device=self.device,
    )
    token_buffer[:, 0] = last_token

    if include_logits:
      # The last logit from the buffer will always be a zero
      logits_buffer = torch.zeros(
          (batch_size, buffer_size, self.model.config.vocab_size),
          dtype=self.dtype,
          device=self.device,
      )
    else:
      logits_buffer = None

    return _SamplingState(
        positions=positions + 1,
        token_buffer=token_buffer,
        logits_buffer=logits_buffer,
        cache=cache,
        decoding_step=torch.zeros([], dtype=torch.int32, device=self.device),
        done=torch.zeros((batch_size,), dtype=torch.bool, device=self.device),
        total_sampling_steps=torch.tensor(
            total_sampling_steps,
            dtype=torch.int32,
            device=self.device,
        ),
    )

  @at.typed
  def tokenize(self, input_string: str) -> torch.Tensor:
    """Tokenizes the input string."""
    input_ids = self.vocab.EncodeAsIds(input_string)
    input_ids = torch.tensor(
        [self.vocab.bos_id()] + torch.tensor(input_ids).tolist(),
        dtype=torch.int32,
        device=self.device,
    )
    return input_ids

  @at.typed
  def _sample_fn(
      self,
      initial_sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal sampling function (to be jitted)."""

    sampler_state = initial_sampling_state
    while (
        (sampler_state.decoding_step < sampler_state.total_sampling_steps) &
        torch.any(torch.logical_not(sampler_state.done))
    ):
      sampler_state = self._sample_step(sampler_state)

    return sampler_state

  @at.typed
  def _get_padded_tokens(
      self,
      tokens: Sequence[torch.Tensor],
  ) -> at.Tokens:
    """Returns an array of padded tokens."""
    max_input_length = max(len(input_ids) for input_ids in tokens)

    pad_values = [
        torch.full(
            [max_input_length - len(input_ids)],
            self.vocab.pad_id(),
            dtype=input_ids.dtype,
            device=self.device,
        )
        for input_ids in tokens
    ]

    padded_tokens = [
        torch.concatenate([pad, input_ids], dim=0)
        for input_ids, pad in zip(tokens, pad_values)
    ]
    padded_tokens = torch.stack(padded_tokens, axis=0)
    return padded_tokens

  @torch.no_grad
  def __call__(
      self,
      input_strings: Sequence[str],
      total_generation_steps: int,
      echo: bool = False,
      return_logits: bool = False,
  ) -> SamplerOutput:
    """Samples a completion of the input string.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      total_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      echo: whether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    # Prefill Stage
    factory_kwargs = dict(device=self.device, dtype=torch.int32)
    all_input_ids = [self.tokenize(x) for x in input_strings]
    input_lengths = torch.tensor(
        [len(input_ids) for input_ids in all_input_ids],
        **factory_kwargs,
    )
    padded_tokens = self._get_padded_tokens(all_input_ids)
    batch_size, prompt_length = padded_tokens.shape
    positions = torch.arange(prompt_length, **factory_kwargs)
    positions = torch.repeat_interleave(positions[None], batch_size, dim=0)
    # Make all positions to end with the corresponding sequence `length - 1`.
    positions = positions - prompt_length + input_lengths[:, None]
    positions = torch.clip(positions, min=0)

    logits, cache = self.model(
        tokens=padded_tokens,
        segment_pos=positions,
    )

    # Sampling Stage
    next_token = torch.argmax(logits, axis=-1)  # [B, Max Prompt Length]
    next_token = next_token[:, -1]

    initial_sampling_state = self.init_sample_state(
        positions=positions[:, -1:],
        last_token=next_token,
        total_sampling_steps=total_generation_steps,
        cache=cache,
        include_logits=return_logits,
    )

    sampling_state = self._sample_fn(initial_sampling_state)

    token_buffer = sampling_state.token_buffer[:, :-1]
    logits_buffer = (
        sampling_state.logits_buffer[:, :-1, :] if return_logits else None
    )

    if echo:
      out_tokens = torch.concatenate(
          [padded_tokens, token_buffer],
          axis=1,
      ).tolist()
      out_logits = (
          torch.concatenate([logits, logits_buffer], axis=1).tolist()
          if return_logits
          else []
      )
    else:
      out_tokens = token_buffer.tolist()
      out_logits = logits_buffer.tolist() if logits_buffer is not None else []

    decoded_outputs = [self.vocab.DecodeIds(tokens) for tokens in out_tokens]
    result = SamplerOutput(
        text=decoded_outputs,
        logits=out_logits,
        tokens=out_tokens,
    )
    return result
