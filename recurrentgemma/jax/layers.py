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
"""Base layers."""

from collections.abc import Sequence
import functools

import einops
from flax import linen as nn
import jax
import jax.numpy as jnp
from recurrentgemma import common
from recurrentgemma.jax import array_typing as at
from recurrentgemma.jax import pallas


class RMSNorm(nn.Module):
  """RMSNorm layer.

  Attributes:
    width: The number of dimensions of the input and output.
    eps: Small constant added to the square root when normalizing.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width: int
  eps: float = 1e-6
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  def setup(self):
    # Parameters.
    self.scale = self.param(
        "scale",
        nn.initializers.zeros_init(),
        (self.width,),
        self.param_dtype,
    )

  @at.typed
  def __call__(
      self, x: at.ExpandedActivations
  ) -> at.ExpandedActivations:
    """Calls the RMSNorm."""
    x, scale = nn.dtypes.promote_dtype(x, self.scale, dtype=self.dtype)

    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_x = x * jax.lax.rsqrt(var + self.eps)

    scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))

    return normed_x * (scale + 1)


class BlockDiagonalLinear(nn.Module):
  """Block-diagonal linear layer.

  Attributes:
    width: The number of dimensions of the input and output.
    num_blocks: The number of diagonal blocks in the layer.
    w_init_variance_scale: A parameters that scales the variance of the
      initialization of the weights.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width: int
  num_blocks: int
  w_init_variance_scale: float = 1.0
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  @property
  def kernel_init(self) -> nn.initializers.Initializer:
    """Initializer for the weight `w` of the layer."""
    return nn.initializers.variance_scaling(
        scale=self.w_init_variance_scale,
        mode="fan_in",
        distribution="normal",
    )

  def setup(self):
    assert self.width % self.num_blocks == 0
    block_width = self.width // self.num_blocks

    # Parameters.
    self.w = self.param(
        "w",
        self.kernel_init,
        [self.num_blocks, block_width, block_width],
        self.param_dtype,
    )
    self.b = self.param(
        "b",
        nn.initializers.zeros_init(),
        [self.num_blocks, block_width],
        self.param_dtype,
    )

  @at.typed
  def __call__(
      self, x: at.ExpandedActivations
  ) -> at.ExpandedActivations:
    """Calls the BlockDiagonalLinear."""
    x, w, b = nn.dtypes.promote_dtype(x, self.w, self.b, dtype=self.dtype)

    # Split x to blocks.
    x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

    # Linear layer over each block + bias.
    y = jnp.einsum("... h i, h i j -> ... h j", x, w) + b

    # Flatten the output.
    return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


@at.typed
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def lru_linear_scan(
    x: at.ExpandedActivations,
    h0: at.RNNState | None,
    a: at.ExpandedActivations,
    acc_dtype: at.dtype = jnp.float32,
) -> tuple[at.ExpandedActivations, at.RNNState]:
  """Computes a linear scan over the second axis of the inputs."""
  def body_fn(h_prev, current_inputs):
    x_t, a_t = current_inputs
    h_t = a_t.astype(acc_dtype) * h_prev + x_t.astype(acc_dtype)
    return h_t, h_t.astype(x.dtype)

  h_last, y = jax.lax.scan(
      body_fn,
      init=jnp.zeros(x.shape[1:], acc_dtype) if h0 is None else h0,
      xs=(x, a),
      unroll=128,
  )
  return y, h_last


@at.typed
def lru_associative_scan(
    x: at.ExpandedActivations,
    h0: at.RNNState | None,
    a: at.ExpandedActivations,
    acc_dtype: at.dtype = jnp.float32,
) -> tuple[at.ExpandedActivations, at.RNNState]:
  """Computes an associative scan over the second axis of the inputs."""

  def lru_associative_bin_op(
      element_i: tuple[jax.Array, jax.Array],
      element_j: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j

  orig_dtype = x.dtype
  x = x.astype(acc_dtype)
  a = a.astype(acc_dtype)

  # Optionally concatenate the hidden state.
  if h0 is not None:
    a = jnp.concatenate([jnp.ones_like(a[:, :1]), a], axis=1)
    x = jnp.concatenate([h0[:, None], x], axis=1)

  _, y = jax.lax.associative_scan(lru_associative_bin_op, (a, x), axis=-2)

  # Remove the first element if there was a hidden state.
  if h0 is not None:
    y = y[:, 1:]

  return y.astype(orig_dtype), y[:, -1]


@at.typed
def rnn_scan(
    x: at.ExpandedActivations,
    a: at.ExpandedActivations,
    reset: at.Reset,
    h0: at.RNNState | None,
    scan_type: common.ScanType,
    acc_dtype: at.dtype = jnp.float32,
    pallas_sharding_spec: pallas.PallasShardingSpec | None = None,
) -> tuple[at.ExpandedActivations, at.RNNState]:
  """Runs the recurrence of a linear RNN.

  Args:
    x: The input sequence.
    a: The diagonal of the recurrence matrix `A`.
    reset: Indicator of document boundaries, e.g. when to reset the hidden state
      of the RNN.
    h0: The initial hidden state.
    scan_type: Which scan implementation to use.
    acc_dtype: The data type for the accumulation.
    pallas_sharding_spec: Sharding spec for running Pallas on sharded values.

  Returns:
    The output of the linear recurrence.
  """
  assert x.ndim == 3
  assert a.shape == x.shape[-a.ndim :]
  assert a.dtype == x.dtype
  assert type(a) is type(x)
  assert h0 is None or h0.dtype == acc_dtype

  if scan_type == common.ScanType.AUTO:
    if jax.local_devices()[0].platform == "tpu":
      scan_type = common.ScanType.LINEAR_PALLAS
    else:
      scan_type = common.ScanType.LINEAR_NATIVE

  # Multiply `a` by the reset.
  a = a * (1 - reset)[..., None]

  if x.shape[1] == 1:
    # Using scan in sampling mode.
    if h0 is None:
      return x, x[:, 0].astype(acc_dtype)

    else:
      y = a.astype(acc_dtype) * h0[:, None] + x.astype(acc_dtype)
      return y.astype(x.dtype), y[:, -1]

  else:
    match scan_type:

      case common.ScanType.LINEAR_PALLAS:
        # Using a Pallas linear scan kernel.
        if acc_dtype != jnp.float32:
          raise ValueError(f"Unsupported accumulation dtype: {acc_dtype}.")

        y = pallas.sharded_lru(x, a, h0, pallas_sharding_spec)
        return y, y[:, -1].astype(acc_dtype)

      case common.ScanType.LINEAR_NATIVE:
        # Using native Jax linear scan.
        return lru_linear_scan(x, h0, a, acc_dtype)

      case common.ScanType.ASSOCIATIVE_NATIVE:
        # Using native Jax associative scan.
        return lru_associative_scan(x, h0, a, acc_dtype)

      case _:
        raise ValueError(f"Unsupported scan type: {scan_type}.")


def rnn_param_init(
    min_rad: float,
    max_rad: float,
    transform: str = "softplus",
    eps: float = 1e-8,
) -> nn.initializers.Initializer:
  """Initializes the `A` parameter of the RG-LRU uniformly on a ring."""

  def init(
      key: jax.Array,
      shape: Sequence[int],
      dtype: at.dtype = jnp.float32,
  ) -> at.RNNDiagonal:
    unif = jax.random.uniform(key, shape=shape)
    # Proportional to area in a ring.
    a_real = 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + eps)

    if transform == "softplus":
      # Inverse transform.
      return jnp.log(jnp.exp(-a_real) - 1.0).astype(dtype)
    else:
      raise NotImplementedError()

  return init


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
@at.typed
def sqrt_bound_derivative(
    x: jax.Array,
    max_gradient: float | jax.Array,
) -> jax.Array:
  """Computes a square root with a gradient clipped at `max_gradient`."""
  del max_gradient  # unused
  return jnp.sqrt(x)


@at.typed
def stable_sqrt_fwd(
    x: jax.Array,
    _: float | jax.Array
) -> tuple[jax.Array, tuple[jax.Array]]:  # pylint: disable=g-one-element-tuple
  return jnp.sqrt(x), (x,)


@at.typed
def stable_sqrt_bwd(
    max_gradient: float | jax.Array,
    res: tuple[jax.Array],  # pylint: disable=g-one-element-tuple
    g: jax.Array,
) -> tuple[jax.Array]:  # pylint: disable=g-one-element-tuple
  (x,) = res
  x_pre = jnp.maximum(x, 1 / (4 * max_gradient**2))
  return jax.vjp(jnp.sqrt, x_pre)[1](g)


sqrt_bound_derivative.defvjp(stable_sqrt_fwd, stable_sqrt_bwd)


class RGLRU(nn.Module):
  """A Real-Gated Linear Recurrent Unit (RG-LRU) layer.

  Attributes:
    width: The number of dimensions of the input and output.
    num_heads: The number of diagonal blocks in the input and A gate layers.
    scan_type: Which scan implementation to use.
    w_init_variance_scale: Initialization parameter for the BlockDiagonalLinear
      layers of the gates. See the `BlockDiagonalLinear` layer for details.
    pallas_sharding_spec: Sharding spec for running Pallas on sharded values.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width: int
  num_heads: int
  scan_type: common.ScanType = common.ScanType.AUTO
  w_init_variance_scale: float = 1.0
  pallas_sharding_spec: pallas.PallasShardingSpec | None = None
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  @property
  def a_param_init(self) -> nn.initializers.Initializer:
    """Initializer for the `A` parameter of the RG-LRU."""
    return rnn_param_init(min_rad=0.9, max_rad=0.999)

  def setup(self):
    # Parameters and layers.
    self.a_param = self.param(
        "a_param",
        self.a_param_init,
        [self.width],
        self.param_dtype,
    )
    self.input_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=self.w_init_variance_scale,
        name="input_gate",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.a_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=self.w_init_variance_scale,
        name="a_gate",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

  @at.typed
  def __call__(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      prev_h: at.RNNState | None = None,
  ) -> tuple[at.ExpandedActivations, at.RNNState]:
    """Calls the RG-LRU.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      prev_h: The previous hidden state of the RG-LRU.

    Returns:
      Output of the block together with the updated hidden state.
    """
    x, a_param = nn.dtypes.promote_dtype(x, self.a_param, dtype=self.dtype)

    bs, l, _ = x.shape
    assert segment_pos.shape == (bs, l)
    reset = segment_pos == 0

    # Gate for the input.
    gate_x = jax.nn.sigmoid(self.input_gate(x))

    # Gate for `A`.
    gate_a = jax.nn.sigmoid(self.a_gate(x))

    # Compute the parameter `A` of the recurrence.
    log_a = -8.0 * gate_a * jax.nn.softplus(a_param)
    a = jnp.exp(log_a)
    a_squared = jnp.exp(2 * log_a)

    # Gate the input.
    gated_x = x * gate_x

    # Apply gamma normalization to the input. We need to clip the derivatives of
    # `sqrt` in order to prevent NaNs during training in bfloat16.
    multiplier = sqrt_bound_derivative(1 - a_squared, 1000)
    multiplier = reset[..., None] + (1 - reset)[..., None] * multiplier
    normalized_x = gated_x * multiplier.astype(x.dtype)

    y, last_h = rnn_scan(
        x=normalized_x,
        a=a,
        reset=reset,
        h0=prev_h,
        scan_type=self.scan_type,
        pallas_sharding_spec=self.pallas_sharding_spec,
    )
    return y, last_h

  @classmethod
  def init_cache(cls, batch_size: int, width: int) -> at.RNNState:
    """Returns an empty initialized cache for the RG-LRU."""
    # RG-LRU cache always in float32.
    return jnp.zeros((batch_size, width), dtype=jnp.float32)


class Conv1D(nn.Module):
  """A 1D temporal convolution layer.

  Attributes:
    width: The number of dimensions of the input and output.
    temporal_width: The size of the temporal receptive field of the convolution.
    w_init_variance_scale: A parameter that scales the variance of the
      initialization of the weights.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width: int
  temporal_width: int
  w_init_variance_scale: float = 0.01
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  @property
  def kernel_init(self) -> nn.initializers.Initializer:
    """Initializer for the kernel of the Conv1D."""
    return nn.initializers.variance_scaling(
        scale=self.w_init_variance_scale,
        mode="fan_in",
        distribution="normal",
    )

  def setup(self):
    # Parameters.
    self.w = self.param(
        "w",
        self.kernel_init,
        [self.temporal_width, self.width],
        self.param_dtype,
    )
    self.b = self.param(
        "b",
        nn.initializers.zeros_init(),
        [self.width],
        self.param_dtype,
    )

  @at.typed
  def __call__(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      state: at.Conv1DState | None = None,
  ) -> tuple[at.ExpandedActivations, at.Conv1DState]:
    """Calls the Conv1D.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      state: The state containing the previous `self.temporal_width-1` inputs
        This is set to `None` in training mode.

    Returns:
      The output of the convolution and the updated state.
    """
    x, w, b = nn.dtypes.promote_dtype(x, self.w, self.b, dtype=self.dtype)

    if state is not None:
      # 1. Decoding mode:
      # - We have access to the previous `self.temporal_width - 1` inputs.
      # - Only a single token needs to be output.
      x = self._concatenate_with_state(x, state)
      prompt_len = self.temporal_width - 1
      output_len = 1
      state_dtype = state.dtype
    else:
      # 1. Training mode:
      # - The full sequence length need to be output.
      prompt_len = 0
      output_len = x.shape[1]
      state_dtype = x.dtype

    # 3. Perform the convolution:
    # - Initialize an accumulator for the convolution output.
    convolution_output = 0.0

    # - We cannot look back by more than the total sequence length
    #   ("valid" convolution).
    temporal_width = min(self.temporal_width, prompt_len+output_len)

    # - The convolution is implemented as a manual loop so that we can
    #   incorporate the window masking further below.
    for temporal_shift in range(temporal_width):
      start_idx, end_idx = self._convolution_window_indices(
          prompt_len=prompt_len,
          shift_back=temporal_shift,
          output_len=output_len,
      )
      x_window = x[:, start_idx:end_idx]

      if state is None:
        # - Ensure that the mask prevents accessing tokens from a different
        #   document in training mode.
        window_mask = self._compute_document_mask(
            segment_pos=segment_pos,
            start_idx=start_idx,
            end_idx=end_idx,
            max_look_ahead=temporal_shift,
        )
        x_window *= window_mask[:, :, None].astype(x.dtype)

      x_window = self._pad_window(x_window, output_len)

      # - Select w for this temporal shift, and expand on the batch and time
      #   dimensions.
      w_shift = w[self.temporal_width-temporal_shift-1][None, None, :]

      # - Accumulate the convolution result.
      convolution_output += x_window * w_shift

    # - Add the bias of the convolution.
    convolution_output += b[None, None]

    # 4. Store the new (potentially padded) state for future decoding.
    new_state = x[:, 1 - self.temporal_width:].astype(state_dtype)
    new_state = self._pad_state(new_state)

    return convolution_output, new_state

  def _concatenate_with_state(
      self,
      x: at.ExpandedActivations,
      state: at.Conv1DState,
  ) -> at.ExpandedActivations:
    """Concatenates the current input `x` with the previous state for decoding.

    Args:
      x: The current input activations (shape: [batch_size, 1, width]).
      state: State tensor storing previous inputs
        (shape: [batch_size, temporal_width - 1, width]).

    Returns:
      The concatenated input sequence
      (shape: [batch_size, temporal_width, width]).
    """
    b, num_tokens, d = x.shape
    assert state.shape == (b, self.temporal_width - 1, d)
    assert num_tokens == 1
    return jnp.concatenate([state.astype(x.dtype), x], axis=1)

  def _convolution_window_indices(
      self,
      *,
      prompt_len: int,
      shift_back: int,
      output_len: int,
  ) -> tuple[int, int]:
    """Calculates the start and end indices for the convolution window.

    Args:
      prompt_len: Length of the prompt (zero in training mode).
      shift_back: By how much the window should be shifted backwards.
      output_len: Sequence length of the output (sequence length in training
        mode, one in decoding mode).

    Returns:
      start_idx: The starting index for the convolution window.
      end_idx: The ending index for the convolution window.
    """
    start_idx = max(prompt_len - shift_back, 0)
    end_idx = prompt_len + output_len - shift_back
    return start_idx, end_idx

  def _compute_document_mask(
      self,
      *,
      segment_pos: at.SegmentPos,
      start_idx: int,
      end_idx: int,
      max_look_ahead: int,
  ) -> jax.Array:
    """Creates a mask to prevent mixing of information between documents.

    Args:
        segment_pos: Position of each token in the sequence. In particular,
          a zero indicates the start of a new document.
        start_idx: The starting index of the convolution window.
        end_idx: The ending index of the convolution window.
        max_look_ahead: How much to look ahead at most to detect a document
          boundary (depends on the convolution).

    Returns:
        An integer mask where `1` indicates a position that should be
        included in the convolution, and `0` a position that should be excluded.
    """
    batch_size = segment_pos.shape[0]
    not_a_document_boundary = (segment_pos != 0).astype(jnp.int32)
    mask = jnp.ones((batch_size, end_idx - start_idx))
    for shift in range(1, max_look_ahead + 1):
      # At each position, look ahead by `shift` tokens to see if a
      # document boundary is present there.
      mask *= not_a_document_boundary[:, start_idx+shift:end_idx+shift]
    return mask

  def _pad_window(
      self,
      window: at.ExpandedActivations,
      output_len: int,
  ) -> at.ExpandedActivations:
    """Left-pads the window if it is shorter than the output sequence length."""
    batch_size, window_len, width = window.shape
    padding_len = output_len - window_len
    padding = jnp.zeros((batch_size, padding_len, width), dtype=window.dtype)
    return jnp.concatenate([padding, window], axis=1)

  def _pad_state(
      self,
      state: at.Conv1DState,
  ) -> at.Conv1DState:
    """Left-pads the state if it is shorter than the temporal width."""
    b, state_seq_len, d = state.shape
    padding_len = self.temporal_width - state_seq_len - 1
    padding = jnp.zeros((b, padding_len, d), dtype=state.dtype)
    return jnp.concatenate([padding, state], axis=1)

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      width: int,
      dtype: at.dtype,
      conv1d_temporal_width: int = 4,
  ) -> at.Conv1DState:
    """Returns an empty initialized cache for the Conv1D."""
    shape = (batch_size, conv1d_temporal_width - 1, width)
    return jnp.zeros(shape, dtype=dtype)


class Einsum(nn.Module):
  """Einsum is a convenience module for parameterized tensor multiplication.

  Attributes:
    w_shape: The shape of the weight matrix w.
    b_shape: The shape of the bias.
    eqn: The einsum string.
    w_init_variance_scale: A parameters that scales the variance of the
      initialization of the weights.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  w_shape: Sequence[int]
  b_shape: Sequence[int]
  eqn: str
  w_init_variance_scale: float = 1.0
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32

  @property
  def kernel_init(self) -> nn.initializers.Initializer:
    """Initializer for the kernel of the Einsum."""
    return nn.initializers.variance_scaling(
        scale=self.w_init_variance_scale,
        mode="fan_in",
        distribution="normal",
        in_axis=[1],
    )

  def setup(self):
    # Parameters.
    self.w = self.param(
        "w",
        self.kernel_init,
        tuple(self.w_shape),
        self.param_dtype,
    )
    self.b = self.param(
        "b",
        nn.initializers.zeros_init(),
        tuple(self.b_shape),
        self.param_dtype,
    )

  @at.typed
  def __call__(self, x: jax.Array) -> jax.Array:
    """Calls the Einsum."""
    x, w, b = nn.dtypes.promote_dtype(x, self.w, self.b, dtype=self.dtype)
    return jnp.einsum(self.eqn, x, w) + b
