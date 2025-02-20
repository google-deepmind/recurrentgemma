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
from typing import Literal, overload

import einops
from flax import linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
from recurrentgemma import common
from recurrentgemma.jax import array_typing as at
from recurrentgemma.jax import complex_lib
from recurrentgemma.jax import scan


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
  def __call__(self, x: at.ExpandedActivations) -> at.ExpandedActivations:
    """Calls the RMSNorm."""
    x, scale = nn.dtypes.promote_dtype(x, self.scale, dtype=self.dtype)

    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_x = x * jax.lax.rsqrt(var + self.eps)

    scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))

    return normed_x * (scale + 1)


class BlockDiagonalLinear(nn.Module):
  """Block-diagonal linear layer.

  Attributes:
    width_input: The number of dimensions of the input.
    width_output: The number of dimensions of the output.
    num_blocks: The number of diagonal blocks in the layer.
    w_init_variance_scale: A parameters that scales the variance of the
      initialization of the weights.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width_input: int
  num_blocks: int
  width_output: int | None = None
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
    assert self.width_input % self.num_blocks == 0

    if self.width_output is None:
      width_output = self.width_input
    else:
      width_output = self.width_output

    assert width_output % self.num_blocks == 0

    input_size = self.width_input // self.num_blocks
    output_size = width_output // self.num_blocks

    # Parameters.
    self.w = self.param(
        "w",
        self.kernel_init,
        [self.num_blocks, input_size, output_size],
        self.param_dtype,
    )
    self.b = self.param(
        "b",
        nn.initializers.zeros_init(),
        [self.num_blocks, output_size],
        self.param_dtype,
    )

  @at.typed
  def __call__(
      self, x: jt.Float[jt.Array, "*b t e"]
  ) -> jt.Float[jt.Array, "*b t f"]:
    """Calls the BlockDiagonalLinear."""
    x, w, b = nn.dtypes.promote_dtype(x, self.w, self.b, dtype=self.dtype)

    # Split x to blocks.
    x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

    # Linear layer over each block + bias.
    y = jnp.einsum("... h i, h i j -> ... h j", x, w) + b

    # Flatten the output.
    return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


def rnn_real_param_init(
    min_rad: float,
    max_rad: float,
    transform: str = "softplus",
    eps: float = 1e-8,
) -> nn.initializers.Initializer:
  """Initializes the `A` real parameter of the RG-LRU uniformly on a ring."""

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


def rnn_imag_param_init(
    max_rad: float,
) -> nn.initializers.Initializer:
  """Initializes the `A` imag parameter of the RG-LRU uniformly on a ring."""

  def init(
      key: jax.Array,
      shape: Sequence[int],
      dtype: at.dtype = jnp.float32,
  ) -> at.RNNDiagonal:
    unif = jax.random.uniform(key, shape=shape)
    return (jnp.pi * max_rad * unif).astype(dtype)

  return init


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
@at.typed
def sqrt_bound_derivative(
    x: complex_lib.RealOrComplex,
    max_gradient: float | jax.Array,
) -> jax.Array:
  """Computes a square root with a gradient clipped at `max_gradient`."""
  del max_gradient  # unused
  return jnp.sqrt(x)


@at.typed
def stable_sqrt_fwd(
    x: jax.Array,
    _: float | jax.Array,
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
    scan_sharding_spec: Sharding spec for running scan on sharded values.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
    only_real: Whether to use only real numbers.
    min_rad: The minimum radius of the ring on which the real part of `A` is
      initialized.
  """

  width: int
  num_heads: int
  scan_type: common.ScanType = common.ScanType.AUTO
  w_init_variance_scale: float = 1.0
  scan_sharding_spec: scan.ShardingSpec | None = None
  dtype: at.dtype | None = None
  param_dtype: at.dtype = jnp.float32
  only_real: bool = True
  min_rad: float = 0.9

  @at.typed
  def merged_to_complex(
      self,
      x: jt.Float[jt.ArrayLike, "*b"],
  ) -> complex_lib.RealOrComplex:
    """Returns a (complex) array from a merged array.

    A merged array is one where the first half over the last axis represents the
    real part of a complex array, while the second part represents the
    imaginary.

    Args:
      x: The merged array.

    Returns:
      A (complex) array represented by `x`.
    """
    if self.only_real:
      return x

    assert x.shape[-1] % 2 == 0
    return self.real_imag_complex(*jnp.split(x, 2, axis=-1))

  @at.typed
  def real_imag_complex(
      self,
      real: jt.Float[jt.Array, "*b"],
      imag: jt.Float[jt.Array, "*b"],
  ) -> complex_lib.RealOrComplex:
    """Based on the settings, creates a (complex) number in the correct format.

    Args:
      real: The real part of the complex number.
      imag: The imaginary part of the complex number.

    Returns:
      The correct representation for a complex number. If `only_real=True`
      the function expects that `imag` is None and will directly return `real`.
      When using `bfloat16` or Pallas a `complex_lib.Complex` is returned,
      otherwise a native jax array with a complex type.
    """
    if self.only_real:
      assert imag is None
      return real

    if self.use_custom_complex(real.dtype):
      return complex_lib.Complex(real, imag)
    else:
      return real + 1j * imag

  def use_custom_complex(self, real_dtype: jnp.dtype) -> bool:
    return (
        real_dtype in (jnp.bfloat16, jnp.float16)
        or self.scan_type == common.ScanType.LINEAR_PALLAS
    )

  @at.typed
  def complex_to_merged(
      self,
      x: complex_lib.RealOrComplex,
  ) -> jt.Float[jt.ArrayLike, "*b"]:
    """Returns a merged array from a (complex) array.

    A merged array is one where the first half over the last axis represents the
    real part of a complex array, while the second part represents the
    imaginary.

    Args:
      x: The (complex) array.

    Returns:
      A merged array represented by `x`.
    """
    if self.only_real:
      assert not isinstance(x, complex_lib.Complex) and not jnp.iscomplexobj(x)
      return x

    else:
      return einops.rearrange([x.real, x.imag], "c ... d -> ... (c d)", c=2)

  @property
  def a_real_param_init(self) -> nn.initializers.Initializer:
    """Initializer for the real `A` parameter of the RG-LRU."""
    return rnn_real_param_init(min_rad=self.min_rad, max_rad=0.999)

  @property
  def a_imag_param_init(self) -> nn.initializers.Initializer:
    """Initializer for the imag `A` parameter of the RG-LRU."""
    return rnn_imag_param_init(max_rad=0.1)

  def setup(self):
    # Parameters and layers.
    assert self.width % 2 == 0

    if not self.only_real:
      assert self.min_rad < 0.999

    width_output = self.width if self.only_real else self.width // 2
    self.a_real_param = self.param(
        "a_param",
        self.a_real_param_init,
        [width_output],
        self.param_dtype,
    )

    if not self.only_real:
      self.a_imag_param = self.param(
          "a_imag_param",
          self.a_imag_param_init,
          [width_output],
          self.param_dtype,
      )

    self.input_gate = BlockDiagonalLinear(
        width_input=self.width,
        width_output=width_output,
        num_blocks=self.num_heads,
        w_init_variance_scale=self.w_init_variance_scale,
        name="input_gate",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )
    self.a_gate = BlockDiagonalLinear(
        width_input=self.width,
        width_output=width_output,
        num_blocks=self.num_heads,
        w_init_variance_scale=self.w_init_variance_scale,
        name="a_gate",
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

  @overload
  def __call__(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      cache: at.RNNState | None = None,
      return_cache: Literal[True] = True,
  ) -> tuple[at.ExpandedActivations, at.RNNState]:
    ...

  @overload
  def __call__(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      cache: at.RNNState | None = None,
      return_cache: Literal[False] = False,
  ) -> tuple[at.ExpandedActivations, None]:
    ...

  @at.typed
  def __call__(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      cache: at.RNNState | None = None,
      return_cache: bool = True,
  ) -> tuple[at.ExpandedActivations, at.RNNState | None]:
    """Calls the RG-LRU.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      cache: The previous hidden state of the RG-LRU.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the block together with the updated hidden state.
    """
    x, a_real_param = nn.dtypes.promote_dtype(
        x,
        self.a_real_param,
        dtype=self.dtype,
    )

    bs, l, _ = x.shape
    assert segment_pos.shape == (bs, l)

    # Gate for the input.
    gate_x = complex_lib.sigmoid(self.input_gate(x))

    # Gate for `A`.
    gate_a = complex_lib.sigmoid(self.a_gate(x))

    # Compute the parameter `A` of the recurrence.
    log_a = -8.0 * gate_a * complex_lib.softplus(a_real_param)
    if self.only_real:
      a, a_squared = complex_lib.exp(log_a), complex_lib.exp(2 * log_a)
    else:
      (a_imag_param,) = nn.dtypes.promote_dtype(
          self.a_imag_param,
          dtype=self.dtype,
      )
      log_a_imag = a_imag_param * gate_a
      log_a_complex = self.real_imag_complex(log_a, log_a_imag)
      a, a_squared = complex_lib.exp(log_a_complex), complex_lib.exp(2 * log_a)

    x = self.merged_to_complex(x)
    h0 = self.merged_to_complex(cache) if cache is not None else None

    # Gate the input.
    gated_x = x * gate_x

    reset = (segment_pos == 0).astype(a)
    # Apply gamma normalization to the input. We need to clip the derivatives of
    # `sqrt` in order to prevent NaNs during training in bfloat16.
    multiplier = sqrt_bound_derivative(1 - a_squared, 1000)
    multiplier = reset[..., None] + (1 - reset)[..., None] * multiplier
    normalized_x = gated_x * multiplier.astype(x.dtype)

    y, last_h = scan.linear_scan(
        x=normalized_x,
        a=a * (1 - reset[..., None]),
        h0=h0,
        scan_type=self.scan_type,
        sharding_spec=self.scan_sharding_spec,
        unroll=128,
    )

    y = self.complex_to_merged(y)

    if not return_cache:
      return y, None

    last_h = self.complex_to_merged(last_h)

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

  @overload
  def __call__(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      cache: at.RNNState | None = None,
      return_cache: Literal[True] = True,
  ) -> tuple[at.ExpandedActivations, at.RNNState]:
    ...

  @overload
  def __call__(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      cache: at.RNNState | None = None,
      return_cache: Literal[False] = False,
  ) -> tuple[at.ExpandedActivations, None]:
    ...

  @at.typed
  def __call__(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      cache: at.Conv1DState | None = None,
      return_cache: bool = True,
  ) -> tuple[at.ExpandedActivations, at.Conv1DState | None]:
    """Calls the Conv1D.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      cache: The state containing the previous `self.temporal_width-1` inputs
        This is set to `None` in training mode.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      The output of the convolution and the updated state.
    """
    x, w, b = nn.dtypes.promote_dtype(x, self.w, self.b, dtype=self.dtype)

    output_len = x.shape[1]

    if cache is not None:
      # 1. Decoding mode:
      # - We have access to the previous `self.temporal_width - 1` inputs.
      x = self._concatenate_with_state(x, cache)
      prompt_len = self.temporal_width - 1
      state_dtype = cache.dtype
    else:
      # 1. Training mode:
      # - The full sequence length need to be output.
      prompt_len = 0
      state_dtype = x.dtype

    # 3. Perform the convolution:
    # - Initialize an accumulator for the convolution output.
    convolution_output = 0.0

    # - We cannot look back by more than the total sequence length
    #   ("valid" convolution).
    temporal_width = min(self.temporal_width, prompt_len + output_len)

    # - The convolution is implemented as a manual loop so that we can
    #   incorporate the window masking further below.
    for temporal_shift in range(temporal_width):
      start_idx, end_idx = self._convolution_window_indices(
          prompt_len=prompt_len,
          shift_back=temporal_shift,
          output_len=output_len,
      )
      x_window = x[:, start_idx:end_idx]

      if cache is None:
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
      w_shift = w[self.temporal_width - temporal_shift - 1][None, None, :]

      # - Accumulate the convolution result.
      convolution_output += x_window * w_shift

    # - Add the bias of the convolution.
    convolution_output += b[None, None]

    if not return_cache:
      return convolution_output, None

    # 4. Store the new (potentially padded) state for future decoding.
    new_cache = x[:, 1 - self.temporal_width :].astype(state_dtype)
    new_cache = self._pad_cache(new_cache)

    return convolution_output, new_cache

  def _concatenate_with_state(
      self,
      x: at.ExpandedActivations,
      cache: at.Conv1DState,
  ) -> at.ExpandedActivations:
    """Concatenates the current input `x` with the previous cache for decoding.

    Args:
      x: The current input activations (shape: [batch_size, 1, width]).
      cache: State tensor storing previous inputs (shape: [batch_size,
        temporal_width - 1, width]).

    Returns:
      The concatenated input sequence
      (shape: [batch_size, temporal_width, width]).
    """
    b, num_tokens, d = x.shape
    assert cache.shape == (b, self.temporal_width - 1, d)
    assert num_tokens == 1
    return jnp.concatenate([cache.astype(x.dtype), x], axis=1)

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
        segment_pos: Position of each token in the sequence. In particular, a
          zero indicates the start of a new document.
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
      mask *= not_a_document_boundary[:, start_idx + shift : end_idx + shift]
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

  def _pad_cache(
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
