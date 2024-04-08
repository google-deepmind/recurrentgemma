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
import math

import einops
from recurrentgemma.torch import array_typing as at
import torch
from torch import nn


_MAX_SQRT_GRADIENT = 1000.0


class RMSNorm(nn.Module):
  """RMS Norm."""

  def __init__(
      self,
      width: int,
      eps: float = 1e-6,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the RMSNorm.

    Args:
      width: The number of dimensions of the input and output.
      eps: Small constant added to the square root when normalizing.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.eps = eps

    # Parameters.
    self.scale = nn.Parameter(torch.empty(
        [self.width], device=device, dtype=dtype
    ))

    # Initialization
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    torch.nn.init.zeros_(self.scale)

  @at.typed
  def forward(self, x: at.Activations) -> at.ExpandedActivations:
    """Calls the RMSNorm."""
    var = torch.mean(torch.square(x), axis=-1, keepdims=True)
    normed_x = x * torch.rsqrt(var + self.eps)

    scale = torch.reshape(self.scale, [1 for _ in range(x.ndim - 1)] + [-1])

    return normed_x * (scale + 1)


class BlockDiagonalLinear(nn.Module):
  """Block-diagonal linear layer."""

  def __init__(
      self,
      width: int,
      num_blocks: int,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the BlockDiagonalLinear.

    Args:
      width: The number of dimensions of the input and output.
      num_blocks: The number of diagonal blocks in the layer.
      w_init_variance_scale: A parameters that scales the variance of the
        initialization of the weights.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_blocks = num_blocks
    self.w_init_variance_scale = w_init_variance_scale
    self.block_width = self.width // self.num_blocks

    # Parameters.
    self.w = nn.Parameter(torch.empty(
        [self.num_blocks, self.block_width, self.block_width],
        device=device,
        dtype=dtype
    ))
    self.b = nn.Parameter(torch.empty(
        [self.num_blocks, self.block_width], device=device, dtype=dtype
    ))

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.w_init_(self.w)
    torch.nn.init.zeros_(self.b)

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weight `w` of the layer."""
    std = math.sqrt(self.w_init_variance_scale / self.block_width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(self, x: at.ExpandedActivations) -> at.ExpandedActivations:
    """Calls the BlockDiagonalLinear."""
    # Split x to blocks.
    x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

    # Linear layer over each block + bias.
    y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

    # Flatten the output.
    return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


@at.typed
def rnn_scan(
    x: at.ExpandedActivations,
    a: at.ExpandedActivations,
    reset: at.Reset,
    h0: at.RNNState | None,
    acc_dtype: torch.dtype = torch.float32,
) -> tuple[at.ExpandedActivations, at.RNNState]:
  """Runs the recurrence of a linear RNN.

  Args:
    x: The input sequence.
    a: The diagonal of the recurrence matrix `A`.
    reset: Indicator of document boundaries, e.g. when to reset the hidden state
      of the RNN.
    h0: The initial hidden state.
    acc_dtype: The data type for the accumulation.

  Returns:
    The output of the linear recurrence.
  """
  assert x.ndim == 3
  assert a.shape == x.shape[-a.ndim :]
  assert a.dtype == x.dtype
  assert type(a) is type(x)
  assert h0 is None or h0.dtype == acc_dtype

  # Multiply `a` by the reset.
  a = a * ~reset[..., None]

  if x.shape[1] == 1:
    # Using scan in sampling mode.
    if h0 is None:
      return x, x[:, 0].type(acc_dtype)

    else:
      y = a.type(acc_dtype) * h0[:, None] + x.type(acc_dtype)
      return y.type(x.dtype), y[:, -1]

  else:
    # Using scan in linear mode.
    if h0 is not None:
      h_t = h0
    else:
      h_t = torch.zeros(x[:, 0].shape, dtype=acc_dtype, device=x.device)

    y = torch.zeros_like(x)
    for t in range(x.shape[1]):
      h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype)
      y[:, t] = h_t.type(x.dtype)

  return y, h_t


def rnn_param_init(
    tensor: torch.Tensor,
    min_rad: float,
    max_rad: float,
    transform: str = "softplus",
    eps: float = 1e-8,
) -> torch.Tensor:
  """Initializes the `A` parameter of the RG-LRU uniformly on a ring."""
  with torch.no_grad():
    # Proportional to area in a ring.
    # 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + 1e-8)
    tensor.uniform_(min_rad ** 2 + eps, max_rad ** 2 + eps)
    tensor.log_().mul_(0.5)

    if transform == "softplus":
      # Inverse transform.
      # jnp.log(jnp.exp(-a_real) - 1.0).astype(dtype)
      return tensor.neg_().exp_().sub_(1.0).log_()
    else:
      raise NotImplementedError()


class SqrtBoundDerivative(torch.autograd.Function):
  """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

  @staticmethod
  def forward(ctx, x: torch.Tensor) -> torch.Tensor:
    """The forward pass, which is a normal `sqrt`."""
    ctx.save_for_backward(x)
    return torch.sqrt(x)

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
    """The backward pass, which clips the `sqrt` gradient."""
    (x,) = ctx.saved_tensors
    clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
    return grad_output / torch.sqrt(clipped_x_times_4)


class RGLRU(nn.Module):
  """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the RG-LRU.

    Args:
      width: The number of dimensions of the input and output.
      num_heads: The number of diagonal blocks in the input and A gate layers.
      w_init_variance_scale: Initialization parameter for the
        BlockDiagonalLinear layers of the gates. See the `BlockDiagonalLinear`
        layer for details.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.w_init_variance_scale = w_init_variance_scale

    # Parameters and layers.
    self.a_param = nn.Parameter(torch.empty(
        [self.width], device=device, dtype=dtype
    ))
    self.input_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=w_init_variance_scale,
        device=device,
        dtype=dtype,
    )
    self.a_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=self.w_init_variance_scale,
        device=device,
        dtype=dtype,
    )

    # Initialization
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.input_gate.reset_parameters()
    self.a_gate.reset_parameters()
    self.a_param_init(self.a_param)

  def a_param_init(self, w: torch.Tensor) -> torch.Tensor:
    """Initializes the `A` parameter of the RG-LRU."""
    return rnn_param_init(w, min_rad=0.9, max_rad=0.999)

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

    bs, l, _ = x.shape
    assert segment_pos.shape == (bs, l)
    reset = segment_pos == 0

    # Gates for x and a.
    gate_x = torch.sigmoid(self.input_gate(x))
    gate_a = torch.sigmoid(self.a_gate(x))

    # Compute the parameter `A` of the recurrence.
    log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
    a = torch.exp(log_a)
    a_square = torch.exp(2 * log_a)

    # Gate the input.
    gated_x = x * gate_x

    # Apply gamma normalization to the input. We need to clip the derivatives of
    # `sqrt` in order to prevent NaNs during training in bfloat16.
    multiplier = SqrtBoundDerivative.apply(1 - a_square)
    multiplier = reset[..., None] + ~reset[..., None] * multiplier
    normalized_x = gated_x * multiplier.type(x.dtype)

    y, last_h = rnn_scan(
        x=normalized_x,
        a=a,
        reset=reset,
        h0=prev_h,
    )
    return y, last_h

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      width: int,
      device: str | torch.device | None = None,
  ) -> at.RNNState:
    """Returns an empty initialized cache for the RG-LRU."""
    # RG-LRU cache always in float32.
    return torch.zeros((batch_size, width), dtype=torch.float32, device=device)


class Conv1D(nn.Module):
  """A 1D temporal convolution layer."""

  def __init__(
      self,
      width: int,
      temporal_width: int,
      w_init_variance_scale: float = 0.01,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the Conv1D.

    Args:
      width: The number of features for both inputs and outputs.
      temporal_width: The size of the temporal receptive field of the
        convolution. In other words, how much back in time the convolution can
        look to produce an output.
      w_init_variance_scale: A parameter that scales the variance of the
        initialization of the weights.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.temporal_width = temporal_width
    self.w_init_variance_scale = w_init_variance_scale

    # Parameters.
    self.w = nn.Parameter(torch.empty(
        [self.temporal_width, self.width], device=device, dtype=dtype
    ))
    self.b = nn.Parameter(torch.empty([width], device=device, dtype=dtype))

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.w_init_(self.w)
    torch.nn.init.zeros_(self.b)

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weight matrix `w` of the Conv1D."""
    std = math.sqrt(self.w_init_variance_scale / self.temporal_width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(
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

      if state is None:
        # - Ensure that the mask prevents accessing tokens from a different
        #   document in training mode.
        window_mask = self._compute_document_mask(
            segment_pos=segment_pos,
            start_idx=start_idx,
            end_idx=end_idx,
            max_look_ahead=temporal_shift,
        )
        x_window *= window_mask[:, :, None].type(x.dtype).to(device=x.device)

      x_window = self._pad_window(x_window, output_len)

      # - Select w for this temporal shift, and expand on the batch and time
      #   dimensions.
      w = self.w[self.temporal_width - temporal_shift - 1][None, None, :]

      # - Accumulate the convolution result.
      convolution_output += x_window * w

    # - Add the bias of the convolution.
    convolution_output += self.b[None, None]

    # 4. Store the new (potentially padded) state for future decoding.
    new_state = x[:, 1 - self.temporal_width :].type(state_dtype)
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
      state: State tensor storing previous inputs (shape: [batch_size,
        temporal_width - 1, width]).

    Returns:
      The concatenated input sequence
      (shape: [batch_size, temporal_width, width]).
    """
    b, num_tokens, d = x.shape
    assert state.shape == (b, self.temporal_width - 1, d)
    assert num_tokens == 1
    return torch.concatenate([state.type(x.dtype), x], dim=1)

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
  ) -> torch.Tensor:
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
    not_a_document_boundary = (segment_pos != 0).type(torch.int32)
    mask = torch.ones(
        (batch_size, end_idx - start_idx),
        device=segment_pos.device,
    )
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
    padding = torch.zeros(
        (batch_size, padding_len, width),
        dtype=window.dtype,
        device=window.device,
    )
    return torch.concatenate([padding, window], dim=1)

  def _pad_state(
      self,
      state: at.Conv1DState,
  ) -> at.Conv1DState:
    """Left-pads the state if it is shorter than the temporal width."""
    b, state_seq_len, d = state.shape
    padding_len = self.temporal_width - state_seq_len - 1
    padding = torch.zeros(
        (b, padding_len, d),
        dtype=state.dtype,
        device=state.device,
    )
    return torch.concatenate([padding, state], dim=1)

  @classmethod
  def init_cache(
      cls,
      *,
      batch_size: int,
      width: int,
      dtype: torch.dtype,
      conv1d_temporal_width: int = 4,
      device: str | torch.device | None = None,
  ) -> at.Conv1DState:
    """Returns an empty initialized cache for the Conv1D."""
    shape = (batch_size, conv1d_temporal_width - 1, width)
    return torch.zeros(shape, dtype=dtype, device=device)


class Einsum(nn.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      w_shape: Sequence[int],
      b_shape: Sequence[int],
      eqn: str,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the Einsum.

    Args:
      w_shape: The shape of the weight matrix w.
      b_shape: The shape of the bias.
      eqn: The einsum string.
      w_init_variance_scale: A parameters that scales the variance of the
        initialization of the weights.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.w_shape = tuple(w_shape)
    self.b_shape = tuple(b_shape)
    self.eqn = eqn
    self.w_init_variance_scale = w_init_variance_scale

    # Parameters.
    self.w = nn.Parameter(torch.empty(self.w_shape, device=device, dtype=dtype))
    self.b = nn.Parameter(torch.empty(self.b_shape, device=device, dtype=dtype))

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.w_init_(self.w)
    torch.nn.init.zeros_(self.b)

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weight matrix `w` of the Einsum."""
    std = math.sqrt(self.w_init_variance_scale / self.w_shape[1])
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    """Calls the Einsum."""
    return torch.einsum(self.eqn, x, self.w) + self.b
