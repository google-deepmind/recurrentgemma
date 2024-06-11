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
import dataclasses
from typing import Generic, NamedTuple, TypeVar

import jaxtyping as jt
from recurrentgemma import common
from recurrentgemma.torch import array_typing as at
import recurrentgemma.torch.griffin as griffin_lib
import torch

import sentencepiece as spm


Cache = TypeVar("Cache")


@at.typed
@dataclasses.dataclass
class SamplingState(Generic[Cache]):
  """Internal sampling state.

  Attributes:
    tokens_buffer: Fixed-size buffer for accumulating the output tokens.
    step: The number of the current decoding step.
    total_steps: Total number of sampling steps.
    positions: The position of the latest token in the sequence.
    cache: Model state for conditioning the model on autoregressively.
    done: Whether decoding is done on the current sequence.
    logits_buffer: Fixed-size buffer for accumulating the output logits.
  """
  tokens_buffer: jt.Integer[torch.Tensor, "*b l"]
  step: jt.Integer[torch.Tensor, ""]
  total_steps: jt.Integer[torch.Tensor, ""]
  positions: jt.Integer[torch.Tensor, "*b 1"]
  cache: Cache
  done: jt.Bool[torch.Tensor, "*b"]
  logits_buffer: jt.Float[torch.Tensor, "*b l v"] | None = None


class SamplerOutput(NamedTuple):
  """Output of the sampler.

  Attributes:
    text: Decoded samples from the model.
    logits: Per-step logits used during sampling.
    tokens: Tokens corresponding to the generated samples.
  """

  text: list[str]
  logits: list[torch.Tensor]
  tokens: list[torch.Tensor]


class Sampler:
  """Sampler for a Griffin model."""

  def __init__(
      self,
      model: griffin_lib.Griffin,
      vocab: spm.SentencePieceProcessor,
      greedy_sampling: bool = True,
      is_it_model: bool = False,
  ):
    """Initializes a sampler for a Griffin model.

    Args:
      model: An instance of the Griffin model.
      vocab: Vocabulary of the model.
      greedy_sampling: If `True` will sample the `argmax` from the logits, else
        will sample from the categorical distribution defined by the logits.
      is_it_model: if the passed model is instruction tuned
    """
    self.model = model
    self.vocab = vocab
    self.greedy_sampling = greedy_sampling
    self._eos_token = torch.tensor([self.vocab.eos_id()], device=self.device)
    self._is_it_model = is_it_model

  @property
  def dtype(self) -> torch.dtype:
    return next(self.model.parameters()).dtype

  @property
  def device(self) -> torch.device:
    return next(self.model.parameters()).device

  @property
  def vocab_size(self) -> int:
    return self.model.config.vocab_size

  @at.typed
  def apply_model(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: bool = True,
      return_cache: bool = True,
  ) -> tuple[at.TokenLogits | None, Cache | None]:
    return self.model(
        tokens=tokens,
        segment_pos=segment_pos,
        cache=cache,
        return_logits=return_logits,
        return_cache=return_cache,
    )

  @at.typed
  def _sample_from_logits(
      self,
      logits: jt.Float[torch.Tensor, "*b v"],
  ) -> jt.Integer[torch.Tensor, "*b"]:
    """Samples from the logits categorical distribution."""
    if self.greedy_sampling:
      return torch.argmax(logits, dim=-1)
    else:
      return torch.distributions.Categorical(logits=logits).sample()

  @at.typed
  def _sample_step(
      self,
      sampler_state: SamplingState,
      end_sampling_at_eos_token: bool = True,
  ) -> SamplingState:
    """Performs a single sampling step.

    Args:
      sampler_state: The current state of the sampler.
      end_sampling_at_eos_token: Whether to stop sampling for every sequence if
        the model produces an EOS token.

    Returns:
      The updated sampler state.
    """
    step = sampler_state.step
    tokens_buffer = sampler_state.tokens_buffer
    logits_buffer = sampler_state.logits_buffer

    # Process last token.
    last_token = sampler_state.tokens_buffer[:, step][:, None]
    logits, cache = self.apply_model(
        tokens=last_token,
        segment_pos=sampler_state.positions,
        cache=sampler_state.cache,
        return_logits=True,
        return_cache=True,
    )

    # Compute and fill next token.
    next_token = self._sample_from_logits(logits[:, 0])
    tokens_buffer[:, step + 1] = next_token

    # Optionally fill the logits.
    if logits_buffer is not None:
      logits_buffer[:, step + 1] = logits[:, 0]

    # Optionally terminate sampling.
    if end_sampling_at_eos_token:
      done_now = torch.equal(next_token, self._eos_token)
    else:
      done_now = False

    return SamplingState(
        tokens_buffer=tokens_buffer,
        step=step + 1,
        total_steps=sampler_state.total_steps,
        positions=sampler_state.positions + 1,
        cache=cache,
        done=sampler_state.done | done_now,
        logits_buffer=logits_buffer,
    )

  @at.typed
  def tokenize(self, input_string: str) -> torch.Tensor:
    """Tokenizes the input string."""
    if self._is_it_model:
      input_string = common.apply_it_formatter(input_string)

    input_ids = self.vocab.EncodeAsIds(input_string)
    input_ids = torch.tensor(
        [self.vocab.bos_id()] + input_ids,
        dtype=torch.int32,
        device=self.device,
    )
    return input_ids

  @at.typed
  def _sample_fn(
      self,
      sampler_state: SamplingState,
      end_sampling_at_eos_token: bool = True,
  ) -> SamplingState:
    """Internal sampling function (to be jitted)."""
    # This is -1, since we make the first sampling from the prompt.
    while (
        (sampler_state.step < sampler_state.total_steps - 1) &
        torch.any(torch.logical_not(sampler_state.done))
    ):
      sampler_state = self._sample_step(
          sampler_state, end_sampling_at_eos_token
      )

    return sampler_state

  @at.typed
  def _prompt_processing_fn(
      self,
      tokens: at.Tokens,
      input_lengths: at.NumTokens,
      total_generation_steps: int,
      return_logits: bool,
      echo: bool,
  ) -> SamplingState:
    """Pre-processes the prompt."""
    factory_kwargs = dict(device=self.device, dtype=torch.int32)
    batch_size, prompt_length = tokens.shape

    # Make all positions to end with the corresponding sequence `length - 1`.
    positions = torch.arange(prompt_length, **factory_kwargs)
    positions = torch.repeat_interleave(positions[None], batch_size, dim=0)
    positions = positions - prompt_length + input_lengths[:, None]
    positions = torch.clip(positions, min=-1)

    # Actual prompt processing.
    if total_generation_steps == 0:
      # No sampling.
      prev_logits, cache = self.apply_model(
          tokens=tokens,
          segment_pos=positions,
          cache=None,
          return_logits=return_logits and echo,
          return_cache=False,
      )
      logits = None

    elif prompt_length == 1:
      # Just a single BOS token.
      logits, cache = self.apply_model(
          tokens=tokens,
          segment_pos=positions,
          cache=None,
          return_logits=return_logits,
          return_cache=True,
      )
      prev_logits = logits[:, :0]

    else:
      # Process everything except the last token separately, since unless
      # `return_logits=True` and `echo=True`, we don't need `prev_logits`.
      prev_logits, cache = self.apply_model(
          tokens=tokens[:, :-1],
          segment_pos=positions[:, :-1],
          cache=None,
          return_logits=return_logits and echo,
          return_cache=True,
      )
      # Process the last token for logits
      logits, cache = self.apply_model(
          tokens=tokens[:, -1:],
          segment_pos=positions[:, -1:],
          cache=cache,
          return_logits=True,
          return_cache=total_generation_steps > 1,
      )

    # Tokens buffer for samples.
    tokens_buffer = torch.full(
        (batch_size, total_generation_steps),
        self.vocab.pad_id(),
        **factory_kwargs,
    )

    if logits is not None:
      # Sample the next token and update the tokens buffer.
      next_token = self._sample_from_logits(logits[:, 0])
      tokens_buffer[:, 0] = next_token

    if return_logits:
      # Logits buffer for samples.
      logits_buffer = torch.zeros(
          (batch_size, total_generation_steps, self.vocab_size),
          dtype=self.dtype, device=self.device,
      )

      if logits is not None:
        # Updated the logits buffer with the ones used for the next token.
        logits_buffer[:, 0] = logits[:, 0]

    else:
      logits_buffer = None

    step = torch.tensor(0, **factory_kwargs)
    total_steps = torch.tensor(total_generation_steps, **factory_kwargs)

    if echo:
      # Append the tokens to start of the token buffer.
      tokens_buffer = torch.concatenate([tokens, tokens_buffer], dim=1)

      if return_logits:
        if logits is None:
          # No sampling, so all logits are coming from the prompt.
          logits_buffer = prev_logits
        else:
          # Append the logits from the prompt to the start of the logits buffer.
          all_logits = [prev_logits, logits, logits_buffer]
          logits_buffer = torch.concatenate(all_logits, dim=1)

      # Update the step and the total steps accordingly.
      step = step + prompt_length
      total_steps = total_steps + prompt_length

    return SamplingState(
        tokens_buffer=tokens_buffer,
        step=step,
        total_steps=total_steps,
        positions=positions[:, -1:] + 1,
        cache=cache,
        done=torch.zeros((batch_size,), dtype=torch.bool),
        logits_buffer=logits_buffer,
    )

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
    padded_tokens = torch.stack(padded_tokens, dim=0)
    return padded_tokens

  @torch.no_grad
  def __call__(
      self,
      input_strings: Sequence[str],
      total_generation_steps: int,
      echo: bool = False,
      return_logits: bool = False,
      end_sampling_at_eos_token: bool = True,
  ) -> SamplerOutput:
    """Samples a completion of the input string.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      total_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      echo: whether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      end_sampling_at_eos_token: Whether to stop sampling for every sequence if
        the model produces an EOS token.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    if total_generation_steps < 0:
      raise ValueError("total_generation_steps must be at least 0.")

    # Create a batched array from inputs.
    all_input_ids = [self.tokenize(x) for x in input_strings]
    input_lengths = torch.tensor(
        [len(input_ids) for input_ids in all_input_ids],
        device=self.device,
        dtype=torch.int32,
    )
    padded_tokens = self._get_padded_tokens(all_input_ids)
    _, pad_length = padded_tokens.shape
    pad_lengths = pad_length - input_lengths

    # Prefill processing stage.
    sampling_state = self._prompt_processing_fn(
        padded_tokens,
        input_lengths,
        total_generation_steps,
        return_logits,
        echo,
    )

    # Sampling stage.
    if total_generation_steps > 1:
      sampling_state = self._sample_fn(
          sampling_state,
          end_sampling_at_eos_token,
      )

    # Text decoding.
    tokens = [
        tokens[l:]
        for tokens, l in zip(sampling_state.tokens_buffer, pad_lengths)
    ]

    if return_logits:
      logits = [
          logits[l:]
          for logits, l in zip(sampling_state.logits_buffer, pad_lengths)
      ]
    else:
      logits = []

    return SamplerOutput(
        text=[self.vocab.DecodeIds(seq.tolist()) for seq in tokens],
        tokens=tokens,
        logits=logits,
    )
