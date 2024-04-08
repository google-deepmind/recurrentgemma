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

import math
from typing import NamedTuple

import einops
from recurrentgemma import common
from recurrentgemma.torch import array_typing as at
from recurrentgemma.torch import layers
import torch
from torch import nn


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
    inputs: Queries or keys..
    positions: Positions of each token in the sequence.
    max_wavelength: The maximum wavelength used for the sin and cos.

  Returns:
    Rotated keys or queries in first half (along with original in second half).
  """
  batch_size, sequence_length = positions.shape
  x_rope, x = torch.chunk(inputs, 2, dim=-1)
  positions = positions.reshape(batch_size, sequence_length, 1, 1)

  freq = torch.arange(x_rope.shape[-1] // 2, device=x.device)
  freq_exponents = 2 * freq / x_rope.shape[-1]
  timescale = max_wavelength**freq_exponents
  inv_frequencies = 1.0 / timescale

  sinusoid_imp = positions * inv_frequencies
  sin = torch.sin(sinusoid_imp).type_as(inputs)
  cos = torch.cos(sinusoid_imp).type_as(inputs)

  first_half, second_half = torch.chunk(x_rope, 2, dim=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin

  return torch.concatenate([first_part, second_part, x], dim=-1)


@at.typed
def _compute_causal_mask(
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
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

  mask = torch.logical_and(causal_mask, window_cond)
  mask = torch.logical_and(same_segment_mask, mask)
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
  segment_ids = torch.cumsum(segment_pos == 0, dim=-1)
  positions = torch.arange(segment_pos.shape[-1], device=segment_pos.device)
  positions = torch.repeat_interleave(
      positions[None], segment_pos.shape[0], dim=0
  )
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
  device = num_tokens.device
  q_positions = num_tokens[:, None]
  k_positions = torch.arange(window_size + 1, device=device) - window_size
  k_positions = torch.repeat_interleave(
      k_positions[None], q_positions.shape[0], dim=0
  )
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
      keys=torch.concatenate(new_keys, axis=-3),
      values=torch.concatenate(new_values, axis=-3),
      num_tokens=cache.num_tokens + keys.shape[-3],
  )


@at.typed
def _attention_cache_from_prompt(
    keys: at.Keys,
    values: at.Values,
    segment_pos: torch.Tensor,
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
  k_padding = torch.zeros(
      (keys.shape[0], window_size - w, keys.shape[2], keys.shape[3]),
      dtype=keys.dtype,
      device=keys.device,
  )
  v_padding = torch.zeros(
      (values.shape[0], window_size - w, values.shape[2], values.shape[3]),
      dtype=values.dtype,
      device=values.device,
  )
  return AttentionBlockCache(
      keys=torch.concatenate([k_padding, keys[:, -w:]], dim=1),
      values=torch.concatenate([v_padding, values[:, -w:]], dim=1),
      num_tokens=segment_pos[:, -1] + 1,
  )


def gelu(x: torch.Tensor) -> torch.Tensor:
  """Returns the GELU activation function with the same approximation as JAX."""
  return nn.functional.gelu(x, approximate="tanh")


class LocalAttentionBlock(nn.Module):
  """Local Multi-Head Attention (MHA) block."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      window_size: int,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the local attention block.

    Args:
      width: The width of the block.
      num_heads: The number of heads for the attention mechanism.
      window_size: The local attention window size.
      final_w_init_variance_scale: The scale for the initialization of the last
        layer of the block.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialization.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.window_size = window_size
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Layers.
    self.proj_q = nn.Linear(
        in_features=self.width,
        out_features=self.width,
        bias=False,
        device=device,
        dtype=dtype,
    )
    self.proj_k = nn.Linear(
        in_features=self.width,
        out_features=self.head_dim,
        bias=False,
        device=device,
        dtype=dtype,
    )
    self.proj_v = nn.Linear(
        in_features=self.width,
        out_features=self.head_dim,
        bias=False,
        device=device,
        dtype=dtype,
    )
    self.proj_final = nn.Linear(
        in_features=self.width,
        out_features=self.width,
        bias=True,
        device=device,
        dtype=dtype,
    )

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.w_init_(self.proj_q.weight)
    self.w_init_(self.proj_k.weight)
    self.w_init_(self.proj_v.weight)
    self.out_w_init_(self.proj_final.weight)
    torch.nn.init.zeros_(self.proj_final.bias)

  @property
  def head_dim(self) -> int:
    return self.width // self.num_heads

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the queries, keys and values projections."""
    torch.nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0 / self.width))

  def out_w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the final projection."""
    std = math.sqrt(self.final_w_init_variance_scale / self.width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(
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
    queries = self.proj_q(x)
    keys = self.proj_k(x)
    values = self.proj_v(x)
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

      keys = torch.concatenate([cache.keys, keys], dim=-3)
      values = torch.concatenate([cache.values, values], dim=-3)

    else:
      new_cache = _attention_cache_from_prompt(
          keys, values, segment_pos, self.window_size
      )

      attn_mask = _compute_forward_pass_mask(segment_pos, self.window_size)

    # Compute attention.
    logits = einops.einsum(queries, keys, "b t n h, b s n h -> b n t s")
    logits = logits * (self.head_dim**-0.5)
    # Expand for heads axis.
    attn_mask = torch.unsqueeze(attn_mask, dim=1)

    masked_logits = torch.where(attn_mask, logits, _MIN_LOGITS_VALUE)
    masked_logits = masked_logits.type(torch.float32)

    probs = nn.functional.softmax(masked_logits, dim=-1).type_as(x)
    encoded = einops.einsum(probs, values, "b n t s, b s n h -> b t n h")
    encoded = einops.rearrange(
        encoded, "... n h -> ... (n h)", n=self.num_heads
    )
    attn_output = self.proj_final(encoded)

    return attn_output, new_cache

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      window_size: int,
      heads_dim: int,
      dtype: torch.dtype,
      device: str | torch.device | None = None,
  ) -> AttentionBlockCache:
    """Initializes an empty KV-cache for the block."""
    shape = (batch_size, window_size, 1, heads_dim)
    return AttentionBlockCache(
        keys=torch.zeros(shape, device=device, dtype=dtype),
        values=torch.zeros(shape, device=device, dtype=dtype),
        num_tokens=torch.zeros([batch_size], dtype=torch.int32, device=device),
    )


class RecurrentBlock(nn.Module):
  """Griffin and Hawk's recurrent block."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the recurrent block.

    Args:
      width: The width of the block.
      num_heads: The number of RG-LRU heads/blocks to use.
      lru_width: Internal dimension to be projected into for RG-LRU to operate
        on.
      conv1d_temporal_width: The temporal width of the 1d convolution.
      final_w_init_variance_scale: The scale for the initialization of the last
        layer of the block.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialization.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.lru_width = lru_width or width
    self.conv1d_temporal_width = conv1d_temporal_width
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Layers.
    self.linear_y = nn.Linear(
        in_features=self.width,
        out_features=self.lru_width,
        device=device,
        dtype=dtype,
    )
    self.linear_x = nn.Linear(
        in_features=self.width,
        out_features=self.lru_width,
        device=device,
        dtype=dtype,
    )
    self.linear_out = nn.Linear(
        in_features=self.lru_width,
        out_features=self.width,
        device=device,
        dtype=dtype,
    )
    self.conv_1d = layers.Conv1D(
        width=self.lru_width,
        temporal_width=self.conv1d_temporal_width,
        device=device,
        dtype=dtype,
    )
    self.rg_lru = layers.RGLRU(
        width=self.lru_width,
        num_heads=self.num_heads,
        device=device,
        dtype=dtype,
    )

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.w_init_(self.linear_x.weight)
    torch.nn.init.zeros_(self.linear_x.bias)
    self.w_init_(self.linear_y.weight)
    torch.nn.init.zeros_(self.linear_y.bias)
    self.out_w_init_(self.linear_out.weight)
    torch.nn.init.zeros_(self.linear_out.bias)
    self.conv_1d.reset_parameters()
    self.rg_lru.reset_parameters()

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the linear x and y layers of the block."""
    torch.nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0 / self.width))

  def out_w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the last layer of the block."""
    std = math.sqrt(self.final_w_init_variance_scale / self.lru_width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(
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
    y = gelu(y)

    # x branch.
    x = self.linear_x(x)
    x, conv1d_state = self.conv_1d(
        x=x,
        segment_pos=segment_pos,
        state=None if cache is None else cache.conv1d_state,
    )
    x, rg_lru_state = self.rg_lru(
        x=x,
        segment_pos=segment_pos,
        prev_h=None if cache is None else cache.rg_lru_state,
    )

    # Join branches.
    x = x * y
    x = self.linear_out(x)

    return x, RecurrentBlockCache(
        conv1d_state=conv1d_state,
        rg_lru_state=rg_lru_state,
    )

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      lru_width: int,
      dtype: torch.dtype,
      conv1d_temporal_width: int = 4,
      device: str | torch.device | None = None,
  ) -> RecurrentBlockCache:
    """Initializes an empty RG-LRU and Conv1D cache for the block."""
    return RecurrentBlockCache(
        rg_lru_state=layers.RGLRU.init_cache(
            batch_size=batch_size,
            width=lru_width,
            device=device,
        ),
        conv1d_state=layers.Conv1D.init_cache(
            batch_size=batch_size,
            width=lru_width,
            dtype=dtype,
            conv1d_temporal_width=conv1d_temporal_width,
            device=device,
        ),
    )


class MLPBlock(nn.Module):
  """MLP block."""

  def __init__(
      self,
      width: int,
      expanded_width: int,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the MLP block.

    Args:
      width: The width of the block.
      expanded_width: The width of the expansion inside the MLP block.
      final_w_init_variance_scale: The scale for the initialization of the last
        layer of the block.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialization.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.expanded_width = expanded_width
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Layers.
    self.ffw_up = layers.Einsum(
        w_shape=(2, self.width, self.expanded_width),
        b_shape=(2, 1, 1, self.expanded_width),
        eqn="...td,cdD->c...tD",
        device=device,
        dtype=dtype,
    )
    self.ffw_down = nn.Linear(
        in_features=self.expanded_width,
        out_features=self.width,
        device=device,
        dtype=dtype,
    )

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.ffw_up.reset_parameters()
    self.out_w_init_(self.ffw_down.weight)
    torch.nn.init.zeros_(self.ffw_down.bias)

  def out_w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the last layer of the block."""
    std = math.sqrt(self.final_w_init_variance_scale / self.expanded_width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(self, x: at.Activations) -> at.Activations:
    """Calls the MLP block.

    Args:
      x: Sequence of input activations.

    Returns:
      Output of the block.
    """
    out = self.ffw_up(x)
    gate_value = gelu(out[0])
    activations = gate_value * out[1]
    return self.ffw_down(activations)


class ResidualBlock(nn.Module):
  """Griffin and Hawk's residual block."""

  def __init__(
      self,
      width: int,
      mlp_expanded_width: int,
      num_heads: int,
      attention_window_size: int,
      temporal_block_type: common.TemporalBlockType,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the residual block.

    Args:
      width: The width of the block.
      mlp_expanded_width: The width of the expansion inside the MLP block.
      num_heads: The number of heads for the Attention or the RG-LRU.
      attention_window_size: The window size for the local attention block.
      temporal_block_type: Either "RECURRENT" or "ATTENTION", specifying the
        type of recurrent block to use.
      lru_width: The width of the RG-LRU if different from `width`.
      conv1d_temporal_width: The width of the temporal convolution.
      final_w_init_variance_scale: The scale for the variance of the
        initializations of the sub blocks.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialization.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.mlp_expanded_width = mlp_expanded_width
    self.num_heads = num_heads
    self.attention_window_size = attention_window_size
    self.temporal_block_type = temporal_block_type
    self.lru_width = lru_width
    self.conv1d_temporal_width = conv1d_temporal_width
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Sub-blocks and layers.
    self.temporal_pre_norm = layers.RMSNorm(
        width=self.width, device=device, dtype=dtype
    )

    match self.temporal_block_type:
      case common.TemporalBlockType.RECURRENT:
        self.recurrent_block = RecurrentBlock(
            width=self.width,
            num_heads=self.num_heads,
            lru_width=self.lru_width,
            conv1d_temporal_width=self.conv1d_temporal_width,
            final_w_init_variance_scale=self.final_w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

      case common.TemporalBlockType.ATTENTION:
        self.attention_block = LocalAttentionBlock(
            width=self.width,
            num_heads=self.num_heads,
            window_size=self.attention_window_size,
            final_w_init_variance_scale=self.final_w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

    self.channel_pre_norm = layers.RMSNorm(
        width=width, device=device, dtype=dtype,
    )
    self.mlp_block = MLPBlock(
        width=self.width,
        expanded_width=self.mlp_expanded_width,
        final_w_init_variance_scale=self.final_w_init_variance_scale,
        device=device,
        dtype=dtype,
    )

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.temporal_pre_norm.reset_parameters()
    self.temporal_block.reset_parameters()
    self.channel_pre_norm.reset_parameters()
    self.mlp_block.reset_parameters()

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
  def forward(
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
    x = self.mlp_block(x)

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
      dtype: torch.dtype,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
      device: str | torch.device | None = None,
  ) -> ResidualBlockCache:
    """Initializes an empty cache for the block."""
    match temporal_block_type:
      case common.TemporalBlockType.RECURRENT:
        return RecurrentBlock.init_cache(
            batch_size=batch_size,
            lru_width=lru_width or width,
            dtype=dtype,
            conv1d_temporal_width=conv1d_temporal_width,
            device=device,
        )
      case common.TemporalBlockType.ATTENTION:
        return LocalAttentionBlock.init_cache(
            batch_size=batch_size,
            window_size=attention_window_size,
            heads_dim=width // num_heads,
            dtype=dtype,
            device=device,
        )


class Embedder(nn.Module):
  """Embedder module."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      scale_by_sqrt_dim: bool,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the embedder.

    Args:
      vocab_size: The size of the token vocabulary.
      embed_dim: The dimensionality of each token embedding.
      scale_by_sqrt_dim: Whether to scale the output of the block by
        `sqrt(self.embed_dim)`
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialization.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.scale_by_sqrt_dim = scale_by_sqrt_dim

    # Parameters.
    self.input_embedding = nn.Parameter(
        torch.empty(
            [self.vocab_size, self.embed_dim], device=device, dtype=dtype
        )
    )

    # Initialization
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    torch.nn.init.normal_(
        self.input_embedding,
        mean=0.0,
        std=math.sqrt(1.0 / self.embed_dim),
    )

  @at.typed
  def encode(self, x: at.Tokens) -> at.Activations:
    """Encodes an input sequence of tokens."""
    x = self.input_embedding[(x,)]
    if self.scale_by_sqrt_dim:
      # Cast to bfloat16 to match training.
      x = x * torch.tensor(math.sqrt(self.embed_dim)).type(torch.bfloat16)
    return x

  @at.typed
  def decode(self, x: at.Activations) -> at.TokenLogits:
    """Decodes an input sequence of activations."""
    return x @ self.input_embedding.T
