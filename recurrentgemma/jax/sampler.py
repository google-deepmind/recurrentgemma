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

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import jaxtyping as jt
from recurrentgemma.jax import array_typing as at

import sentencepiece as spm


Cache = TypeVar("Cache")


@at.typed
@struct.dataclass
class SamplingState(Generic[Cache]):
  """Internal sampling state.

  Attributes:
    tokens_buffer: Fixed-size buffer for accumulating the output tokens.
    rng: Jax PRNGKey for non-deterministic sampling.
    step: The number of the current decoding step.
    total_steps: Total number of sampling steps.
    positions: The position of the latest token in the sequence.
    cache: Model state for conditioning the model on autoregressively.
    done: Whether decoding is done on the current sequence.
    logits_buffer: Fixed-size buffer for accumulating the output logits.
  """

  tokens_buffer: jt.Integer[jt.Array, "*b l"]
  rng: jt.PRNGKeyArray | None
  step: jt.Integer[jt.Array, ""]
  total_steps: jt.Integer[jt.Array, ""]
  positions: jt.Integer[jt.Array, "*b 1"]
  cache: Cache
  done: jt.Bool[jt.Array, "*b"]
  logits_buffer: jt.Float[jt.Array, "*b l v"] | None = None


@struct.dataclass
class SamplerOutput:
  """Output of the sampler.

  Attributes:
    text: Decoded samples from the model.
    logits: Per-step logits used during sampling.
    tokens: Tokens corresponding to the generated samples.
  """

  text: list[str]
  logits: list[list[float]]
  tokens: list[list[int]]


class Sampler(Generic[Cache]):
  """Sampler for a Griffin model."""

  def __init__(
      self,
      model: nn.Module,
      vocab: spm.SentencePieceProcessor,
      params: at.Params,
      jit_compile: bool = True,
      deterministic_sampling: bool = True,
  ):
    """Initializes a sampler for a Griffin model.

    Args:
      model: An instance of the Griffin model.
      vocab: Vocabulary of the model.
      params: Parameters of the model.
      jit_compile: Whether to jit compile all Jax functions.
      deterministic_sampling: If `True` will sample the `argmax` from the
        logits, else will sample from the categorical distribution defined by
        the logits.
    """
    self.model = model
    self.vocab = vocab
    self.params = params
    self.deterministic_sampling = deterministic_sampling
    self.jit_compile = jit_compile
    self._compiled_prompt_processing_fn = jax.jit(
        self._prompt_processing_fn,
        donate_argnums=[1, 2, 3],
        static_argnums=[4, 5, 6],
    )
    self._compiled_sample_fn = jax.jit(
        self._sample_fn,
        donate_argnums=[1],
        static_argnums=[2],
    )

  @property
  def dtype(self) -> jnp.dtype:
    return jax.tree_util.tree_leaves(self.params)[0].dtype

  @property
  def prompt_processing_fn(self) -> Callable[..., SamplingState[Cache]]:
    if self.jit_compile:
      return self._compiled_prompt_processing_fn
    else:
      return self._prompt_processing_fn

  @property
  def sample_fn(self) -> Callable[..., SamplingState[Cache]]:
    if self.jit_compile:
      return self._compiled_sample_fn
    else:
      return self._sample_fn

  @at.typed
  def apply_model(
      self,
      params: at.Params,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
  ) -> tuple[at.TokenLogits, Cache]:
    return self.model.apply(
        {"params": params},
        tokens=tokens,
        segment_pos=segment_pos,
        cache=cache,
    )

  @at.typed
  def _sample_from_logits(
      self,
      rng: jt.PRNGKeyArray | None,
      logits: jt.Float[jt.Array, "*b v"],
  ) -> tuple[jt.Integer[jt.Array, "*b"], jt.PRNGKeyArray | None]:
    """Samples from the logits categorical distribution."""
    if self.deterministic_sampling:
      return jnp.argmax(logits, axis=-1), rng
    else:
      assert rng is not None
      rng, next_rng = jax.random.split(rng)
      return jax.random.categorical(next_rng, logits), rng

  @at.typed
  def _sample_step(
      self,
      params: at.Params,
      sampler_state: SamplingState[Cache],
      end_sampling_at_eos_token: bool = True,
  ) -> SamplingState[Cache]:
    """Performs a single sampling step.

    Args:
      params: Parameters of the model.
      sampler_state: The current state of the sampler.
      end_sampling_at_eos_token: Whether to stop sampling for every sequence if
        the model produces an EOS token.

    Returns:
      The updated sampler state.
    """
    step = sampler_state.step
    tokens_buffer = sampler_state.tokens_buffer
    logits_buffer = sampler_state.logits_buffer

    # Process last token
    last_token = sampler_state.tokens_buffer[:, step][:, None]
    logits, cache = self.apply_model(
        params=params,
        tokens=last_token,
        segment_pos=sampler_state.positions,
        cache=sampler_state.cache,
    )

    # Compute and fill next token
    next_token, rng = self._sample_from_logits(sampler_state.rng, logits[:, 0])
    tokens_buffer = tokens_buffer.at[:, step + 1].set(next_token)

    # Optionally fill the logits
    if logits_buffer is not None:
      logits_buffer = sampler_state.logits_buffer.at[:, step + 1].set(
          logits[:, 0]
      )

    # Optionally terimnate sampling
    if end_sampling_at_eos_token:
      done_now = jnp.equal(next_token, self.vocab.eos_id())
    else:
      done_now = False

    return SamplingState(
        tokens_buffer=tokens_buffer,
        rng=rng,
        step=step + 1,
        total_steps=sampler_state.total_steps,
        positions=sampler_state.positions + 1,
        cache=cache,
        done=sampler_state.done | done_now,
        logits_buffer=logits_buffer,
    )

  @at.typed
  def tokenize(self, input_string: str) -> jax.Array:
    """Tokenizes the input string."""
    input_ids = self.vocab.EncodeAsIds(input_string)
    input_ids = jnp.array(
        [self.vocab.bos_id()] + jnp.array(input_ids).tolist(), dtype=jnp.int32
    )
    return input_ids

  @at.typed
  def _sample_fn(
      self,
      params: at.Params,
      initial_sampling_state: SamplingState,
      end_sampling_at_eos_token: bool = True,
  ) -> SamplingState:
    """Internal sampling function (to be jitted)."""

    def sample_with_params(sampler_state: SamplingState) -> SamplingState:
      return self._sample_step(params, sampler_state, end_sampling_at_eos_token)

    def cond_fn(sampler_state: SamplingState) -> jax.Array:
      # This is -1, since we make the first sampling from the prompt
      cond1 = sampler_state.step < sampler_state.total_steps - 1
      cond2 = jnp.any(jnp.logical_not(sampler_state.done))
      return jnp.logical_and(cond1, cond2)

    return jax.lax.while_loop(
        cond_fn, sample_with_params, initial_sampling_state
    )

  @at.typed
  def _prompt_processing_fn(
      self,
      params: at.Params,
      tokens: at.Tokens,
      rng: jt.PRNGKeyArray | None,
      input_lengths: at.NumTokens,
      total_generation_steps: int,
      return_logits: bool,
      echo: bool,
  ) -> SamplingState:
    """Pre-processes the prompt."""
    batch_size, prompt_length = tokens.shape

    # Make all positions to end with the corresponding sequence `length - 1`.
    positions = jnp.repeat(jnp.arange(prompt_length)[None], batch_size, axis=0)
    positions = positions - prompt_length + input_lengths[:, None]
    positions = jnp.maximum(positions, 0)

    if prompt_length == 1:
      logits, cache = self.apply_model(params, tokens, positions)
      prev_logits = logits[:, :0]

    else:
      # Process everything except the last token separately, since unless
      # `return_logits=True` and `echo=True`, we don't need `prev_logits`.
      prev_logits, cache = self.apply_model(
          params, tokens[:, :-1], positions[:, :-1]
      )
      # Process the last token for logits
      logits, cache = self.apply_model(
          params, tokens[:, -1:], positions[:, -1:], cache
      )

    # Create the newly sampled tokens buffer
    tokens_buffer = jnp.full(
        (batch_size, total_generation_steps),
        self.vocab.pad_id(),
        dtype=jnp.int32,
    )
    if logits is not None:
      next_token, rng = self._sample_from_logits(rng, logits[:, 0])
      tokens_buffer = tokens_buffer.at[:, 0].set(next_token)

    if return_logits:
      # Create the newly sampled logits buffer
      logits_buffer = jnp.zeros(
          (batch_size, total_generation_steps, logits.shape[-1]),
          dtype=self.dtype,
      )
      if logits is not None:
        logits_buffer = logits_buffer.at[:, 0].set(logits[:, 0])

    else:
      logits_buffer = None

    step = jnp.array(0, dtype=jnp.int32)
    total_steps = jnp.array(total_generation_steps, dtype=jnp.int32)

    if echo:
      # Append the tokens and logits to the start of the buffers and update
      # accordingly the step and total_steps
      tokens_buffer = jnp.concatenate([tokens, tokens_buffer], axis=1)
      if return_logits:
        logits_buffer = jnp.concatenate(
            [prev_logits, logits, logits_buffer], axis=1
        )
      step = step + prompt_length
      total_steps = total_steps + prompt_length

    return SamplingState(
        tokens_buffer=tokens_buffer,
        rng=rng,
        step=step,
        total_steps=total_steps,
        positions=positions[:, -1:] + 1,
        cache=cache,
        done=jnp.zeros((batch_size,), dtype=jnp.bool_),
        logits_buffer=logits_buffer,
    )

  @at.typed
  def _get_padded_tokens(
      self,
      tokens: Sequence[jax.Array],
  ) -> at.Tokens:
    """Returns an array of padded tokens."""
    max_input_length = max(len(input_ids) for input_ids in tokens)

    pad_values = [
        max_input_length - len(input_ids) for input_ids in tokens
    ]

    padded_tokens = [
        jnp.pad(input_ids, (pad, 0), constant_values=self.vocab.pad_id())
        for input_ids, pad in zip(tokens, pad_values)
    ]
    padded_tokens = jnp.stack(padded_tokens, axis=0)
    return padded_tokens

  def __call__(
      self,
      input_strings: Sequence[str],
      total_generation_steps: int,
      rng: jt.PRNGKeyArray | None = None,
      echo: bool = False,
      return_logits: bool = False,
      end_sampling_at_eos_token: bool = True,
  ) -> SamplerOutput:
    """Samples a completion of the input string.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      total_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      rng: A Jax PRNGKey to use if sampling non-deterministically. You must
        provide if you want non-deterministic sampling.
      echo: whether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      end_sampling_at_eos_token: Whether to stop sampling for every sequence if
        the model produces an EOS token.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    if not self.deterministic_sampling and rng is None:
      raise ValueError("rng must be provided if sampling "
                       "non-deterministically.")
    if total_generation_steps < 1:
      raise ValueError("total_generation_steps must be at least 1.")

    # Create a batched array from inputs
    all_input_ids = [self.tokenize(x) for x in input_strings]
    input_lengths = jnp.asarray([len(input_ids) for input_ids in all_input_ids])
    padded_tokens = self._get_padded_tokens(all_input_ids)

    # Prefill processing stage
    sampling_state = self.prompt_processing_fn(
        self.params,
        padded_tokens,
        rng,
        input_lengths,
        total_generation_steps,
        return_logits,
        echo,
    )

    # Sampling stage
    if total_generation_steps > 1:
      sampling_state = self.sample_fn(
          self.params,
          sampling_state,
          end_sampling_at_eos_token,
      )

    # Decoding
    out_tokens = sampling_state.tokens_buffer.tolist()
    decoded_outputs = [self.vocab.DecodeIds(tokens) for tokens in out_tokens]

    if sampling_state.logits_buffer is not None:
      out_logits = sampling_state.logits_buffer.tolist()
    else:
      out_logits = []

    result = SamplerOutput(
        text=decoded_outputs,
        logits=out_logits,
        tokens=out_tokens,
    )
    return result
