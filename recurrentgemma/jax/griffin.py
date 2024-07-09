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
"""Griffin model."""

from typing import Literal, overload

from flax import linen as nn
import jax.numpy as jnp
from recurrentgemma import common
from recurrentgemma.jax import array_typing as at
from recurrentgemma.jax import layers
from recurrentgemma.jax import modules
from recurrentgemma.jax import scan


Cache = dict[str, modules.ResidualBlockCache]


class Griffin(nn.Module):
  """Griffin model - https://arxiv.org/abs/2402.19427.

  Attributes:
    config: The Griffin config.
    scan_sharding_spec: Sharding spec for running scan on sharded values.
    gradient_checkpointing: Whether to apply gradient checkpointing on every
      residual block.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  config: common.GriffinConfig
  scan_sharding_spec: scan.ShardingSpec | None = None
  gradient_checkpointing: bool = True
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  def setup(self):
    self.embedder = modules.Embedder(
        vocab_size=self.config.vocab_size,
        embed_dim=self.config.width,
        scale_by_sqrt_dim=self.config.embeddings_scale_by_sqrt_dim,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

    block_class = modules.ResidualBlock
    if self.gradient_checkpointing:
      # `return_cache` is a static argument.
      block_class = nn.remat(block_class, static_argnums=4)
    self.blocks = [
        block_class(
            name=f"blocks.{i}",
            width=self.config.width,
            mlp_expanded_width=self.config.mlp_expanded_width,
            num_heads=self.config.num_heads,
            lru_width=self.config.lru_width,
            attention_window_size=self.config.attention_window_size,
            temporal_block_type=block_type,
            scan_type=self.config.scan_type,
            final_w_init_variance_scale=2.0 / self.config.num_layers,
            scan_sharding_spec=self.scan_sharding_spec,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        for i, block_type in enumerate(self.config.block_types)
    ]
    self.final_norm = layers.RMSNorm(
        width=self.config.width,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

  @overload
  def __call__(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: Literal[False] = False,
      return_cache: Literal[False] = False,
  ) -> tuple[None, None]:
    ...

  @overload
  def __call__(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: Literal[False] = False,
      return_cache: Literal[True] = True,
  ) -> tuple[None, Cache]:
    ...

  @overload
  def __call__(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: Literal[True] = True,
      return_cache: Literal[False] = False,
  ) -> tuple[at.TokenLogits, None]:
    ...

  @overload
  def __call__(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: Literal[True] = True,
      return_cache: Literal[True] = True,
  ) -> tuple[at.TokenLogits, Cache]:
    ...

  @at.typed
  def __call__(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: bool = True,
      return_cache: bool = True,
  ) -> tuple[at.TokenLogits | None, Cache | None]:
    """Calls Griffin.

    Args:
      tokens: Sequence of input tokens.
      segment_pos: Positions of each token in the sequence.
      cache: Cache with pre-computed values for sampling.
      return_logits: Whether to compute and return the logits.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the model together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    if not return_logits and not return_cache:
      return None, None

    input_emb = self.embedder.encode(jnp.array(tokens))
    x = input_emb

    new_cache = {}
    for i, block in enumerate(self.blocks):
      layer_name = f"blocks.{i}"
      x, new_cache[layer_name] = block(
          x,
          segment_pos,
          None if cache is None else cache[layer_name],
          return_cache,
      )

    if not return_logits:
      return None, new_cache

    x = self.final_norm(x)
    logits = self.embedder.decode(x)

    c = self.config.logits_soft_cap
    if c:
      logits = jnp.tanh(logits / c) * c

    if not return_cache:
      return logits, None

    return logits, new_cache

  def init_cache(
      self,
      batch_size: int,
      dtype: at.dtype,
  ) -> Cache:
    """Initializes an empty cache for the model."""
    cache = {}
    for i, block_type in enumerate(self.config.block_types):
      cache[f"blocks.{i}"] = modules.ResidualBlock.init_cache(
          batch_size=batch_size,
          width=self.config.width,
          num_heads=self.config.num_heads,
          attention_window_size=self.config.attention_window_size,
          temporal_block_type=block_type,
          dtype=dtype,
          lru_width=self.config.lru_width,
      )
    return cache
