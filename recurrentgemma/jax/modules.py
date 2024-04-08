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
"""Griffin and Hawk"s model components."""

from typing import NamedTuple
import einops
from flax import linen as nn
import jax
import jax.numpy as jnp
from recurrentgemma import common
from recurrentgemma.jax import array_typing as at
from recurrentgemma.jax import layers
from recurrentgemma.jax import pallas


_MIN_LOGITS_VALUE = -2.3819763e38  # Set to a large negative number.
_MAX_WAVELENGTH = 10_000


@at.typed
class RecurrentBlockCache(NamedTuple):
  """The cache for a recurrent block."""

  rg_lru_state: at.RNNState
  conv1d_state: at.Conv1DState


@at.typed
class AttentionBlockCache(NamedTuple):
  """The cache for an attention block."""

  keys: at.CachedKeys
  values: at.CachedValues
  num_tokens: at.NumTokens


ResidualBlockCache = RecurrentBlockCache | AttentionBlockCache


@at.typed
def _apply_rope(
    inputs: at.Keys | at.Queries,
    positions: at.SegmentPos,
    max_wavelength: int = _MAX_WAVELENGTH,
) -> at.Keys | at.Queries:
  """Applies RoPE to the first half of inputs.

  Args:
    inputs: Queries or keys.
    positions: Positions of each token in the sequence.
    max_wavelength: The maximum wavelength used for the sin and cos.

  Returns:
    Rotated keys or queries in first half (along with original in second half).
  """
  x_rope, x = jnp.split(inputs, 2, axis=-1)
  positions = jnp.expand_dims(
      positions, [i for i in range(x.ndim) if i not in (0, 1)]
  )

  freq_exponents = 2 * jnp.arange(x_rope.shape[-1] // 2) / x_rope.shape[-1]
  timescale = max_wavelength**freq_exponents
  inv_frequencies = 1.0 / timescale

  sinusoid_imp = positions * inv_frequencies
  sin = jnp.sin(sinusoid_imp).astype(inputs.dtype)
  cos = jnp.cos(sinusoid_imp).astype(inputs.dtype)

  first_half, second_half = jnp.split(x_rope, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin

  return jnp.concatenate([first_part, second_part, x], axis=-1)


@at.typed
def _compute_causal_mask(
    q_positions: jax.Array,
    k_positions: jax.Array,
    window_size: int,
    q_segment_ids: at.QuerySegmentIds | None,
    k_segment_ids: at.KeySegmentIds | None,
) -> at.AttentionMask:
  """Computes the causal mask for local attention.

  Args:
    q_positions: Position of each query token in the sequence.
    k_positions: Position of each key token in the sequence.
    window_size: The local attention window size.
    q_segment_ids: Optional segment id for each query token.
    k_segment_ids: Optional segment id for each key token.

  Returns:
    The mask that needs to be applied to the logits of the local attention.
  """
  # Mask for attending only to the same segment.
  if q_segment_ids is not None or k_segment_ids is not None:
    assert q_segment_ids is not None and k_segment_ids is not None
    same_segment_mask = q_segment_ids[..., None] == k_segment_ids[..., None, :]
  else:
    same_segment_mask = (k_positions >= 0)[..., None, :]

  # Mask for attending only to previous tokens.
  causal_mask = q_positions[..., None] >= k_positions[..., None, :]

  # Mask for attending only to things within the window size.
  window_cond = q_positions[..., None] <= (
      k_positions[..., None, :] + window_size
  )

  mask = jnp.logical_and(causal_mask, window_cond)
  mask = jnp.logical_and(same_segment_mask, mask)
  return mask


@at.typed
def _compute_forward_pass_mask(
    segment_pos: at.SegmentPos,
    window_size: int,
) -> at.AttentionMask:
  """Compute the forward pass mask.

  Args:
    segment_pos: Position of each token in the sequence.
    window_size: The local attention window size.

  Returns:
    The mask that needs to be applied to the logits when performing a forward
    pass (e.g. prompt processing) of the local attention.
  """
  segment_ids = jnp.cumsum(segment_pos == 0, axis=-1)
  positions = jnp.arange(segment_pos.shape[-1])
  positions = jnp.repeat(positions[None], segment_pos.shape[0], axis=0)
  return _compute_causal_mask(
      positions, positions, window_size, segment_ids, segment_ids
  )


@at.typed
def _compute_cache_mask(
    num_tokens: at.NumTokens,
    window_size: int,
) -> at.AttentionMask:
  """Computes the mask when there a KV-cache is present.

  Args:
    num_tokens: The number of active tokens currently stored in the KV-cache.
    window_size: The local attention window size.

  Returns:
    The mask that needs to be applied to the logits when performing a single
    inference step with a KV-cache of the local attention.
  """
  q_positions = num_tokens[:, None]
  k_positions = jnp.arange(window_size + 1) - window_size
  k_positions = jnp.repeat(k_positions[None], q_positions.shape[0], axis=0)
  k_positions = k_positions + num_tokens[:, None]
  return _compute_causal_mask(q_positions, k_positions, window_size, None, None)


@at.typed
def _update_attention_cache(
    keys: at.Keys,
    values: at.Values,
    cache: AttentionBlockCache,
) -> AttentionBlockCache:
  """Updates the cache with the new keys and values.

  Args:
    keys: The new keys to be added to the cache.
    values: The new values to be added to the cache.
    cache: The dictionary with the cache to be updated.

  Returns:
    The updated cache dictionary.
  """
  l = keys.shape[-3]
  window_size = cache.keys.shape[-3]
  n_fill = min(window_size, l)

  new_keys = [cache.keys[:, n_fill:], keys[:, -n_fill:]]
  new_values = [cache.values[:, n_fill:], values[:, -n_fill:]]
  return AttentionBlockCache(
      keys=jnp.concatenate(new_keys, axis=-3),
      values=jnp.concatenate(new_values, axis=-3),
      num_tokens=cache.num_tokens + keys.shape[-3],
  )


@at.typed
def _attention_cache_from_prompt(
    keys: at.Keys,
    values: at.Values,
    segment_pos: at.SegmentPos,
    window_size: int,
) -> AttentionBlockCache:
  """Creates a new cache from a prompt.

  Args:
    keys: The new keys to be added to an empty cache.
    values: The new values to be added to an empty cache.
    segment_pos: Positions of each token in the sequence.
    window_size: The local attention window size.

  Returns:
    An empty initialized KV-cache updated with the given keys and values.
  """
  w = min(window_size, keys.shape[1])
  padding = [[0, 0], [window_size - w, 0], [0, 0], [0, 0]]
  return AttentionBlockCache(
      keys=jnp.pad(keys[:, -w:], padding),
      values=jnp.pad(values[:, -w:], padding),
      num_tokens=segment_pos[:, -1] + 1,
  )


class LocalAttentionBlock(nn.Module):
  """Local Multi-Head Attention (MHA) block.

  Attributes:
    width: The width of the block.
    num_heads: The number of heads for the attention mechanism.
    window_size: The local attention window size.
    final_w_init_variance_scale: The scale for the initialization of the last
      layer of the block.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width: int
  num_heads: int
  window_size: int
  final_w_init_variance_scale: float = 1.0
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  @property
  def head_dim(self) -> int:
    """The dimension of each head."""
    return self.width // self.num_heads

  @property
  def kernel_init(self) -> nn.initializers.Initializer:
    """Initialization of the kernel for the queries, keys and values projections."""
    return nn.initializers.variance_scaling(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
    )

  @property
  def out_kernel_init(self) -> nn.initializers.Initializer:
    """Initialization of the kernel for the final projection."""
    return nn.initializers.variance_scaling(
        scale=self.final_w_init_variance_scale,
        mode="fan_in",
        distribution="normal",
    )

  def setup(self):
    # Layers.
    self.q = nn.Dense(
        features=self.width,
        use_bias=False,
        kernel_init=self.kernel_init,
        name="proj_q",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.k = nn.Dense(
        features=self.head_dim,
        use_bias=False,
        kernel_init=self.kernel_init,
        name="proj_k",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.v = nn.Dense(
        features=self.head_dim,
        use_bias=False,
        kernel_init=self.kernel_init,
        name="proj_v",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.out = nn.Dense(
        features=self.width,
        use_bias=True,
        kernel_init=self.out_kernel_init,
        name="proj_final",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

  @at.typed
  def __call__(
      self,
      x: at.Activations,
      segment_pos: at.SegmentPos,
      cache: AttentionBlockCache | None = None,
  ) -> tuple[at.Activations, AttentionBlockCache]:
    """Calls the local attention block.

    Args:
      x: Sequence of input activations.
      segment_pos: Positions of each token in the sequence.
      cache: Optiona KV-cache for the block, of previous keys and values.

    Returns:
      Output of the block together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    b, t, _ = x.shape
    assert segment_pos.shape == (b, t), segment_pos.shape

    # Generate keys, values and queries.
    queries = self.q(x)
    keys = self.k(x)
    values = self.v(x)
    queries = einops.rearrange(
        queries, "... (n h) -> ... n h", n=self.num_heads
    )
    keys = einops.rearrange(keys, "... (n h) -> ... n h", n=1)
    values = einops.rearrange(values, "... (n h) -> ... n h", n=1)

    # Apply rotary embeddings.
    queries = _apply_rope(queries, segment_pos)
    keys = _apply_rope(keys, segment_pos)

    if cache is not None:
      assert t == 1, f"When cache is provided only `t=1` is supported, not {t=}"

      new_cache = _update_attention_cache(keys, values, cache)

      attn_mask = _compute_cache_mask(cache.num_tokens, self.window_size)

      keys = jnp.concatenate([cache.keys, keys], axis=-3)
      values = jnp.concatenate([cache.values, values], axis=-3)

    else:
      new_cache = _attention_cache_from_prompt(
          keys, values, segment_pos, self.window_size
      )

      attn_mask = _compute_forward_pass_mask(segment_pos, self.window_size)

    # Compute attention.
    logits = einops.einsum(queries, keys, "b t n h, b s n h -> b n t s")
    logits = logits * (self.head_dim**-0.5)
    # Expand for heads axis.
    attn_mask = jnp.expand_dims(attn_mask, axis=-3)

    masked_logits = jnp.where(attn_mask, logits, _MIN_LOGITS_VALUE)
    masked_logits = masked_logits.astype(jnp.float32)

    probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)
    encoded = einops.einsum(probs, values, "b n t s, b s n h -> b t n h")
    encoded = einops.rearrange(
        encoded, "... n h -> ... (n h)", n=self.num_heads
    )
    attn_output = self.out(encoded)

    return attn_output, new_cache

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      window_size: int,
      heads_dim: int,
      dtype: at.dtype,
  ) -> AttentionBlockCache:
    """Initializes an empty KV-cache for the block."""
    return AttentionBlockCache(
        keys=jnp.zeros((batch_size, window_size, 1, heads_dim), dtype=dtype),
        values=jnp.zeros((batch_size, window_size, 1, heads_dim), dtype=dtype),
        num_tokens=jnp.zeros([batch_size], dtype=jnp.int32),
    )


class RecurrentBlock(nn.Module):
  """Griffin and Hawk's recurrent block.

  Attributes:
    width: The width of the block.
    num_heads: The number of RG-LRU heads/blocks to use.
    lru_width: Internal dimension to be projected into for RG-LRU to operate on.
    scan_type: What kind of scan implementation to use for the RG-LRU.
    conv1d_temporal_width: The temporal width of the 1d convolution.
    final_w_init_variance_scale: The scale for the initialization of the last
      layer of the block.
    pallas_sharding_spec: Sharding spec for running Pallas on sharded values.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width: int
  num_heads: int
  lru_width: int | None = None
  scan_type: common.ScanType = common.ScanType.AUTO
  conv1d_temporal_width: int = 4
  final_w_init_variance_scale: float = 1.0
  pallas_sharding_spec: pallas.PallasShardingSpec | None = None
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  @property
  def kernel_init(self) -> nn.initializers.Initializer:
    """Initialization of the kernel for the linear x and y layers of the block."""
    return nn.initializers.variance_scaling(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
    )

  @property
  def out_kernel_init(self) -> nn.initializers.Initializer:
    """Initialization of the kernel for the last layer of the block."""
    return nn.initializers.variance_scaling(
        scale=self.final_w_init_variance_scale,
        mode="fan_in",
        distribution="normal",
    )

  def setup(self) -> None:
    lru_width = self.lru_width or self.width

    # Layers.
    self.linear_y = nn.Dense(
        features=lru_width,
        kernel_init=self.kernel_init,
        name="linear_y",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.linear_x = nn.Dense(
        features=lru_width,
        kernel_init=self.kernel_init,
        name="linear_x",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.linear_out = nn.Dense(
        features=self.width,
        kernel_init=self.out_kernel_init,
        name="linear_out",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.conv_1d = layers.Conv1D(
        width=lru_width,
        temporal_width=self.conv1d_temporal_width,
        name="conv_1d",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.lru = layers.RGLRU(
        width=lru_width,
        num_heads=self.num_heads,
        scan_type=self.scan_type,
        name="rg_lru",
        pallas_sharding_spec=self.pallas_sharding_spec,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

  @at.typed
  def __call__(
      self,
      x: at.Activations,
      segment_pos: at.SegmentPos,
      cache: RecurrentBlockCache | None = None,
  ) -> tuple[at.Activations, RecurrentBlockCache]:
    """Calls the recurrent block.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      cache: Optional cache with the previous state of the RG-LRU and Conv1D.

    Returns:
      Output of the block together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    # y branch.
    y = self.linear_y(x)
    y = jax.nn.gelu(y)

    # x branch.
    x = self.linear_x(x)
    x, conv1d_state = self.conv_1d(
        x=x,
        segment_pos=segment_pos,
        state=None if cache is None else cache.conv1d_state,
    )
    x, rg_lru_state = self.lru(
        x=x,
        segment_pos=segment_pos,
        prev_h=None if cache is None else cache.rg_lru_state,
    )

    # Join branches.
    x = x * y
    x = self.linear_out(x)

    return x, RecurrentBlockCache(
        rg_lru_state=rg_lru_state,
        conv1d_state=conv1d_state,
    )

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      lru_width: int,
      dtype: at.dtype,
      conv1d_temporal_width: int = 4,
  ) -> RecurrentBlockCache:
    """Initializes an empty RG-LRU and Conv1D cache for the block."""
    return RecurrentBlockCache(
        rg_lru_state=layers.RGLRU.init_cache(
            batch_size=batch_size,
            width=lru_width,
        ),
        conv1d_state=layers.Conv1D.init_cache(
            batch_size=batch_size,
            width=lru_width,
            dtype=dtype,
            conv1d_temporal_width=conv1d_temporal_width,
        ),
    )


class MLPBlock(nn.Module):
  """MLP block.

  Attributes:
    width: The width of the block.
    expanded_width: The width of the expansion inside the MLP block.
    final_w_init_variance_scale: The scale for the initialization of the last
      layer of the block.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width: int
  expanded_width: int
  final_w_init_variance_scale: float = 1.0
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  @property
  def out_kernel_init(self) -> nn.initializers.Initializer:
    """Initialization of the kernel for the last layer of the block."""
    return nn.initializers.variance_scaling(
        scale=self.final_w_init_variance_scale,
        mode="fan_in",
        distribution="normal",
    )

  def setup(self) -> None:
    # Layers.
    self.ffw_up = layers.Einsum(
        w_shape=(2, self.width, self.expanded_width),
        b_shape=(2, 1, 1, self.expanded_width),
        eqn="...td,cdD->c...tD",
        name="ffw_up",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.ffw_down = nn.Dense(
        name="ffw_down",
        features=self.width,
        kernel_init=self.out_kernel_init,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

  @at.typed
  def __call__(self, x: at.Activations) -> at.Activations:
    """Calls the MLP block.

    Args:
      x: Sequence of input activations.

    Returns:
      Output of the block.
    """
    out = self.ffw_up(x)
    gate_value = nn.gelu(out[0])
    activations = gate_value * out[1]

    return self.ffw_down(activations)


class ResidualBlock(nn.Module):
  """Griffin and Hawk's residual block.

  Attributes:
    width: The width of the block.
    mlp_expanded_width: The width of the expansion inside the MLP block.
    num_heads: The number of heads for the Attention or the RG-LRU.
    attention_window_size: The window size for the local attention block.
    temporal_block_type: Either "RECURRENT" or "ATTENTION", specifying the type
      of recurrent block to use.
    lru_width: The width of the RG-LRU if different from `width`.
    scan_type: What kind of scan implementation to use for the RG-LRU.
    conv1d_temporal_width: The width of the temporal convolution.
    final_w_init_variance_scale: The scale for the variance of the
      initializations of the sub blocks.
    pallas_sharding_spec: Sharding spec for running Pallas on sharded values.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width: int
  mlp_expanded_width: int
  num_heads: int
  attention_window_size: int
  temporal_block_type: common.TemporalBlockType
  lru_width: int | None = None
  scan_type: common.ScanType = common.ScanType.AUTO
  conv1d_temporal_width: int = 4
  final_w_init_variance_scale: float = 1.0
  pallas_sharding_spec: pallas.PallasShardingSpec | None = None
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  def setup(self) -> None:
    # Sub-blocks and layers.
    self.temporal_pre_norm = layers.RMSNorm(
        width=self.width,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

    match self.temporal_block_type:
      case common.TemporalBlockType.RECURRENT:
        self.recurrent_block = RecurrentBlock(
            width=self.width,
            num_heads=self.num_heads,
            lru_width=self.lru_width,
            conv1d_temporal_width=self.conv1d_temporal_width,
            scan_type=self.scan_type,
            final_w_init_variance_scale=self.final_w_init_variance_scale,
            name="recurrent_block",
            pallas_sharding_spec=self.pallas_sharding_spec,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

      case common.TemporalBlockType.ATTENTION:
        self.attention_block = LocalAttentionBlock(
            width=self.width,
            num_heads=self.num_heads,
            window_size=self.attention_window_size,
            final_w_init_variance_scale=self.final_w_init_variance_scale,
            name="attention_block",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    self.channel_pre_norm = layers.RMSNorm(
        width=self.width,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

    self.mlp = MLPBlock(
        width=self.width,
        expanded_width=self.mlp_expanded_width,
        final_w_init_variance_scale=self.final_w_init_variance_scale,
        name="mlp_block",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

  @property
  def temporal_block(self) -> nn.Module:
    """Alias for the temporal block.

    This creates a common interface while making the layer / parameter types
    easily identifiable by name in a state dictionary.
    """
    match self.temporal_block_type:
      case common.TemporalBlockType.RECURRENT:
        return self.recurrent_block
      case common.TemporalBlockType.ATTENTION:
        return self.attention_block

  @at.typed
  def __call__(
      self,
      x: at.Activations,
      segment_pos: at.SegmentPos,
      cache: ResidualBlockCache | None = None,
  ) -> tuple[at.Activations, ResidualBlockCache]:
    """Calls the residual block.

    Args:
      x: Sequence of input activations.
      segment_pos: Positions of each token in the sequence.
      cache: Optional cache for the block.

    Returns:
      Output of the block together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    raw_x = x

    inputs_normalized = self.temporal_pre_norm(raw_x)
    x, cache = self.temporal_block(inputs_normalized, segment_pos, cache)

    residual = x + raw_x

    x = self.channel_pre_norm(residual)
    x = self.mlp(x)

    x = x + residual

    return x, cache

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      width: int,
      num_heads: int,
      attention_window_size: int,
      temporal_block_type: common.TemporalBlockType,
      dtype: at.dtype,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
  ) -> ResidualBlockCache:
    """Initializes an empty cache for the block."""
    assert width % num_heads == 0
    match temporal_block_type:
      case common.TemporalBlockType.RECURRENT:
        return RecurrentBlock.init_cache(
            batch_size=batch_size,
            lru_width=lru_width or width,
            dtype=dtype,
            conv1d_temporal_width=conv1d_temporal_width,
        )
      case common.TemporalBlockType.ATTENTION:
        return LocalAttentionBlock.init_cache(
            batch_size=batch_size,
            window_size=attention_window_size,
            heads_dim=width // num_heads,
            dtype=dtype,
        )
      case _:
        raise NotImplementedError(temporal_block_type)


class Embedder(nn.Module):
  """Embedder module.

  Attributes:
    vocab_size: The size of the token vocabulary.
    embed_dim: The dimensionality of each token embedding.
    scale_by_sqrt_dim: Whether to scale the output of the block by
      `sqrt(elf.embed_dim)`.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  vocab_size: int
  embed_dim: int
  scale_by_sqrt_dim: bool
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  def setup(self):
    # Parameters.
    self.input_embedding_table = self.param(
        "input_embedding",
        nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_in",
            distribution="normal",
            in_axis=1,
            out_axis=0,
        ),
        (self.vocab_size, self.embed_dim),
        self.param_dtype,
    )

  @at.typed
  def encode(self, x: at.Tokens) -> at.Activations:
    """Encodes an input sequence of tokens."""
    x = self.input_embedding_table[(x,)]
    [x] = nn.dtypes.promote_dtype(x, dtype=self.dtype)

    if self.scale_by_sqrt_dim:
      # Cast to bfloat16 to match training.
      x = x * jnp.sqrt(self.embed_dim).astype(jnp.bfloat16)
    return x

  @at.typed
  def decode(self, x: at.Activations) -> at.TokenLogits:
    """Decodes an input sequence of activations."""
    x, embedding_table = nn.dtypes.promote_dtype(
        x, self.input_embedding_table, dtype=self.dtype,
    )
    return x @ embedding_table.T
