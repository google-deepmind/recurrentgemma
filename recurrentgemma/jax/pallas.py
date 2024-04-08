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
"""Pallas implementation of the scan operation for the RG-LRU."""
import functools
import math
from typing import NamedTuple

import jax
from jax._src.lax.control_flow import for_loop
import jax.experimental.pallas as pl
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from recurrentgemma.jax import array_typing as at


class PallasShardingSpec(NamedTuple):
  """The sharding spec for running a Pallas kernel with sharded values."""
  mesh: jax.sharding.Mesh
  batch_sharding_axis: str | None
  activations_sharding_axis: str | None

  @property
  def activations_sharding_spec(self) -> jax.sharding.PartitionSpec:
    return jax.sharding.PartitionSpec(
        self.batch_sharding_axis, None, self.activations_sharding_axis
    )

  @property
  def rnn_state_sharding_spec(self) -> jax.sharding.PartitionSpec:
    return jax.sharding.PartitionSpec(
        self.batch_sharding_axis, self.activations_sharding_axis
    )


class PallasKernelSpec(NamedTuple):
  """The grid and size specs for the Pallas kernel."""

  batch_grid_size: int
  seq_grid_size: int
  dim_grid_size: int
  batch_tile_size: int
  seq_tile_size: int
  dim_tile_size: int
  singleton_tile_size: int = 128

  @property
  def grid(self) -> tuple[int, int, int]:
    return (self.batch_grid_size, self.dim_grid_size, self.seq_grid_size)


def carry_dtype(dtype: at.dtype) -> at.dtype:
  if dtype in (jnp.bfloat16, jnp.float16):
    return jnp.float32
  else:
    return dtype


def reverse_block_spec(spec: pl.BlockSpec, num_seq_blocks: int) -> pl.BlockSpec:
  """Reverses the order of accessing sequence axis tiles."""
  return pl.BlockSpec(
      lambda b, d, s: spec.index_map(b, d, num_seq_blocks - 1 - s),
      spec.block_shape,
  )


def to_blocks(
    x: jax.Array | None,
    s: int,
) -> jax.Array | None:
  """Reshapes `x` such that it's last dim is equal to `s` by adding an extra axis."""
  if x is None:
    return x
  return jnp.reshape(x, (*x.shape[:-1], x.shape[-1] // s, s))


def from_blocks(x: jax.Array) -> jax.Array:
  """Reverse the effect of `to_blocks`."""
  return jnp.reshape(x, (*x.shape[:-2], -1))


def compute_tile_size(
    dim_size: int,
    max_block_size: int,
    min_block_size: int,
) -> int:
  """Computes the correct Pallas block size."""
  block_size = max_block_size
  while dim_size % block_size != 0 and block_size >= min_block_size:
    block_size = block_size // 2
  if block_size < min_block_size:
    raise ValueError(
        "Could not find a block size that is a power of 2, which "
        f"divides the last dimension {dim_size} and is between "
        f"{min_block_size} and {max_block_size}."
    )
  return block_size


def compute_pallas_kernel_spec(
    x: jax.Array,
    max_seq_block_size: int,
    min_seq_block_size: int,
    singleton_tile_size: int = 128,
) -> PallasKernelSpec:
  """Returns the correct Pallas grid."""
  # Special logic based on the devices
  device = jax.devices()[0]
  if device.platform != "tpu":
    raise ValueError(f"Only TPU devices are supported, got {device.platform=}.")

  batch_size, seq_len, dim = x.shape

  if dim % singleton_tile_size != 0:
    raise ValueError(
        f"{dim=} has to be divisible by {singleton_tile_size=} for pallas scan."
    )

  batch_tile_size = 1
  seq_tile_size = compute_tile_size(
      seq_len,
      max_seq_block_size,
      min_seq_block_size,
  )

  # Find the best tile size
  dim_total_tile_size = dim // singleton_tile_size
  dim_tile_size = min(8, dim_total_tile_size)
  dim_grid_size = math.ceil(dim_total_tile_size / dim_tile_size)

  return PallasKernelSpec(
      batch_grid_size=batch_size // batch_tile_size,
      seq_grid_size=seq_len // seq_tile_size,
      dim_grid_size=dim_grid_size,
      batch_tile_size=batch_tile_size,
      seq_tile_size=seq_tile_size,
      dim_tile_size=dim_tile_size,
      singleton_tile_size=singleton_tile_size,
  )


def make_block_specs(
    kernel_spec: PallasKernelSpec,
) -> tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec]:
  """Returns the block specs for each variable."""
  x_shape = (
      kernel_spec.batch_tile_size,
      kernel_spec.seq_tile_size,
      kernel_spec.dim_tile_size,
      kernel_spec.singleton_tile_size,
  )
  x_spec = pl.BlockSpec(lambda b, d, s: (b, s, d, 0), x_shape)
  # Same as x, but without the sequence dimension
  h_shape = x_shape[:1] + x_shape[2:]
  h_spec = pl.BlockSpec(lambda b, d, s: (0, 0, 0), h_shape)
  hs_spec = pl.BlockSpec(lambda b, d, s: (s, b, d, 0), (1, *h_shape))
  h0_spec = pl.BlockSpec(lambda b, d, s: (b, d, 0), h_shape)

  return x_spec, h_spec, hs_spec, h0_spec


def make_block_shape_and_dtypes(
    x: jax.Array,
    kernel_spec: PallasKernelSpec,
) -> tuple[
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
]:
  """Returns the correct shape and dtype for each variable."""
  s = kernel_spec.singleton_tile_size
  shape = (x.shape[0], x.shape[1], x.shape[2] // s, s)
  x_shape = jax.ShapeDtypeStruct(shape=shape, dtype=x.dtype)
  dtype = carry_dtype(x.dtype)
  h_size = (
      kernel_spec.batch_tile_size,
      kernel_spec.dim_tile_size,
      kernel_spec.singleton_tile_size,
  )
  h_shape = jax.ShapeDtypeStruct(shape=h_size, dtype=dtype)
  hs_size = (kernel_spec.seq_grid_size, x.shape[0], *shape[2:])
  hs_shape = jax.ShapeDtypeStruct(shape=hs_size, dtype=dtype)
  h0_shape = jax.ShapeDtypeStruct(shape=shape[:1] + shape[2:], dtype=dtype)
  return x_shape, h_shape, hs_shape, h0_shape


def initialize_carry(
    h_carry_ref: jax.Array,
    h_init_ref: jax.Array | None,
):
  """Initializes the accumulator and product carries."""

  def init_h0():
    if h_init_ref is None:
      h_carry_ref[:] = jnp.zeros_like(h_carry_ref)
    else:
      h_carry_ref[:] = h_init_ref[:].astype(h_carry_ref.dtype)

  # Initialize to zeros only the first sequence block
  jax.lax.cond(pl.program_id(2) == 0, init_h0, lambda: None)


def forward_kernel_body(
    t: int,
    h_carry_ref: jax.Array,
    dtype: at.dtype,
    x_ref: jax.Array,
    a_ref: jax.Array,
    y_ref: jax.Array,
):
  """Computes the forward LRU loop."""
  # RNN
  h_t = h_carry_ref[:]
  a_t = a_ref[:, t].astype(dtype)
  x_t = x_ref[:, t].astype(dtype)
  h_next = a_t * h_t + x_t
  # Store
  h_carry_ref[:] = h_next
  y_ref[:, t] = h_next.astype(y_ref.dtype)


def backward_kernel_body(
    i: int,
    dh_carry_ref: jax.Array,
    dtype: at.dtype,
    dy_ref: jax.Array,
    a_ref: jax.Array,
    dx_ref: jax.Array,
):
  """Computes the backprop LRU loop."""
  seq_len = a_ref.shape[1]
  t = seq_len - 1 - i

  # RNN backprop
  dh_t = dh_carry_ref[:]
  a_t = a_ref[:, t].astype(dtype)
  dy_prev = dy_ref[:, t - 1].astype(dtype)
  dh_prev = a_t * dh_t + dy_prev
  # Store
  dx_ref[:, t] = dh_t.astype(dx_ref.dtype)
  dh_carry_ref[:] = dh_prev


def lru_forward_kernel(
    x_ref: jax.Array,
    a_ref: jax.Array,
    h_init_ref: jax.Array | None,
    y_ref: jax.Array,
    h_carry_ref: jax.Array,
):
  """Forward pass of the LRU reccurence."""
  # The sequence length for this Pallas block execution
  seq_len = x_ref.shape[1]
  # The hidden is always in jnp.float32 or higher precision
  dtype = carry_dtype(x_ref.dtype)
  # Initialize carries
  initialize_carry(h_carry_ref, h_init_ref)
  # Create concrete (non-reference) carry values for the loop
  h_carry = h_carry_ref[:]
  # Execute loop
  h_n = for_loop.for_loop(  # pytype: disable=wrong-arg-types
      seq_len,
      functools.partial(
          forward_kernel_body,
          dtype=dtype,
          x_ref=x_ref,
          a_ref=a_ref,
          y_ref=y_ref,
      ),
      h_carry,
  )
  # Store the last values of the loop in the carry references
  h_carry_ref[:] = h_n


def lru_backward_kernel(
    dy_ref: jax.Array,
    y_ref: jax.Array,
    a_ref: jax.Array,
    dx_ref: jax.Array,
    dh_carry_ref: jax.Array,
):
  """Backward pass of the LRU reccurence."""
  # The sequence length for this Pallas block execution
  seq_len = y_ref.shape[1]
  # The hidden is always in jnp.float32 or higher precision
  dtype = carry_dtype(y_ref.dtype)
  # Initialize carries
  initialize_carry(dh_carry_ref, None)
  # Create concerete (non-reference) carry values for the loop
  h_carry = dy_ref[:, seq_len - 1].astype(dtype) + dh_carry_ref[:].astype(dtype)
  # Execute loop
  dh_0 = for_loop.for_loop(
      seq_len - 1,
      functools.partial(
          backward_kernel_body,
          dtype=dtype,
          dy_ref=dy_ref,
          a_ref=a_ref,
          dx_ref=dx_ref,
      ),
      h_carry,
  )
  # Update the last input derivative
  dx_ref[:, 0] = dh_0.astype(dx_ref.dtype)
  # Store the last values of the loop in the carry references
  a_0 = a_ref[:, 0].astype(dtype)
  dh_carry_ref[:] = a_0 * dh_0


@at.typed
def _lru_with_h0(
    x: at.ExpandedActivations,
    a: at.ExpandedActivations,
    h0: at.RNNState | None,
    kernel_spec: PallasKernelSpec,
) -> at.ExpandedActivations:
  """Runs the LRU scan using a Pallas kernel."""
  # Make specs and shapes
  specs = make_block_specs(kernel_spec)
  shapes = make_block_shape_and_dtypes(x, kernel_spec)

  args = (x, a, h0)
  args = [to_blocks(arg, kernel_spec.singleton_tile_size) for arg in args]
  in_specs = [specs[0], specs[0], (None if h0 is None else specs[-1])]

  outputs = pl.pallas_call(  # pytype: disable=wrong-arg-types
      lru_forward_kernel,
      in_specs=in_specs,
      out_shape=shapes[:2],
      out_specs=specs[:2],
      grid=kernel_spec.grid,
  )(*args)
  outputs = [from_blocks(arg) for arg in outputs]
  y, _ = outputs
  return y


@functools.partial(jax.custom_vjp, nondiff_argnums=[3])
@at.typed
def _lru(
    x: at.ExpandedActivations,
    a: at.ExpandedActivations,
    h0: at.RNNState | None,
    kernel_spec: PallasKernelSpec,
) -> at.ExpandedActivations:
  """Runs the LRU forward pass without residuals for backprop."""
  return _lru_with_h0(x, a, h0, kernel_spec)


@at.typed
def _lru_fwd(
    x: at.ExpandedActivations,
    a: at.ExpandedActivations,
    h0: at.RNNState | None,
    kernel_spec: PallasKernelSpec,
) -> tuple[
    at.ExpandedActivations,
    tuple[at.ExpandedActivations, at.ExpandedActivations, at.RNNState, bool],
]:
  """Runs the LRU forward pass."""
  has_h0 = h0 is not None

  y = _lru_with_h0(x, a, h0, kernel_spec)
  h0 = h0 if has_h0 else jnp.zeros_like(y[:, -1])

  return y, (a, y, h0, has_h0)


@at.typed
def _lru_bwd_with_dh0(
    dy: at.ExpandedActivations,
    a: at.ExpandedActivations,
    y: at.ExpandedActivations,
    kernel_spec: PallasKernelSpec,
    has_h0: bool,
) -> at.ExpandedActivations:
  """Runs the LRU backward pass."""
  # Make specs and shapes
  specs = make_block_specs(kernel_spec)
  shapes = make_block_shape_and_dtypes(dy, kernel_spec)

  # Extract relevant specs and shapes
  x_spec, h_spec, _, h0_spec = specs
  x_shape, h_shape, _, h0_shape = shapes
  dh_spec = h0_spec if has_h0 else h_spec
  dh_shape = h0_shape if has_h0 else h_shape

  # Used for correcting the specs and shapes
  reverse_seq = functools.partial(
      reverse_block_spec, num_seq_blocks=kernel_spec.seq_grid_size
  )

  dy, y, a = [
      to_blocks(arg, kernel_spec.singleton_tile_size) for arg in (dy, y, a)
  ]

  in_specs = [x_spec, x_spec, x_spec]
  out_specs = [x_spec, dh_spec]
  out_shapes = [x_shape, dh_shape]

  dx, _ = pl.pallas_call(  # pytype: disable=wrong-arg-types
      lru_backward_kernel,
      out_shape=out_shapes,
      in_specs=list(map(reverse_seq, in_specs)),
      out_specs=list(map(reverse_seq, out_specs)),
      grid=kernel_spec.grid,
  )(dy, y, a)
  return from_blocks(dx)


@at.typed
def _lru_bwd(
    kernel_spec: PallasKernelSpec,
    res: tuple[
        at.ExpandedActivations, at.ExpandedActivations, at.RNNState, bool
    ],
    dy: at.ExpandedActivations,
) -> tuple[at.ExpandedActivations, at.ExpandedActivations, at.RNNState | None]:
  """Runs the LRU backward pass."""
  # Unpack
  a, y, h0, has_h0 = res

  dx = _lru_bwd_with_dh0(dy, a, y, kernel_spec, has_h0)

  # Compute da
  h_prev_first = h0[:, None].astype(y.dtype)
  h_prev = jnp.concatenate([h_prev_first, y[:, :-1]], axis=1)
  da = dx * h_prev

  # Compute dh0
  dh0 = (dx[:, 0] * a[:, 0]).astype(h0.dtype)

  return dx, da, (dh0 if has_h0 else None)


_lru.defvjp(_lru_fwd, _lru_bwd)


def pad_array_to_divisible(
    x: jax.Array,
    divisor: int,
    axis: int,
) -> jax.Array:
  """Pads the variable `x` to have size along `axis` divisible by `divisor`."""
  if x.shape[axis] % divisor == 0:
    return x
  n = divisor - x.shape[axis] % divisor
  pad_shape = list(x.shape)
  pad_shape[axis] = n
  zeros = jnp.zeros(pad_shape, dtype=x.dtype)
  return jnp.concatenate([x, zeros], axis=axis)


@at.typed
@functools.partial(jax.jit, static_argnums=[3, 4, 5, 6])
@jax.named_call
def lru(
    x: at.ExpandedActivations,
    a: at.ExpandedActivations,
    h0: at.RNNState | None = None,
    max_seq_block_size: int = 256,
    min_seq_block_size: int = 64,
    pad_seq_to_min_block_size: bool = True,
    pad_last_dim_to_128: bool = True,
) -> at.ExpandedActivations:
  """Runs the LRU scan using a Pallas kernel."""
  *_, t, d = x.shape

  if pad_seq_to_min_block_size:
    f = functools.partial(
        pad_array_to_divisible, divisor=min_seq_block_size, axis=-2
    )
    x, a = f(x), f(a)

  if pad_last_dim_to_128:
    f = functools.partial(pad_array_to_divisible, divisor=128, axis=-1)
    x, a = f(x), f(a)
    if h0 is not None:
      h0 = f(h0)

  kernel_spec = compute_pallas_kernel_spec(
      x=x,
      max_seq_block_size=max_seq_block_size,
      min_seq_block_size=min_seq_block_size,
  )

  result = _lru(x, a, h0, kernel_spec)
  return result[..., :t, :d]


def sharded_lru(
    x: at.ExpandedActivations,
    a: at.ExpandedActivations,
    h0: at.RNNState | None = None,
    pallas_sharding_spec: PallasShardingSpec | None = None,
    max_seq_block_size: int = 256,
    min_seq_block_size: int = 64,
    pad_seq_to_min_block_size: bool = True,
    pad_last_dim_to_128: bool = True,
) -> at.ExpandedActivations:
  """Applies a complex LRU with `shard_map`."""
  if pallas_sharding_spec is None:
    return lru(
        x=x,
        a=a,
        h0=h0,
        max_seq_block_size=max_seq_block_size,
        min_seq_block_size=min_seq_block_size,
        pad_seq_to_min_block_size=pad_seq_to_min_block_size,
        pad_last_dim_to_128=pad_last_dim_to_128,
    )

  args = (x, a, h0)
  args_specs = (
      pallas_sharding_spec.activations_sharding_spec,
      pallas_sharding_spec.activations_sharding_spec,
      pallas_sharding_spec.rnn_state_sharding_spec,
  )

  if h0 is None:
    args = args[:2]
    args_specs = args_specs[:2]

  f = shard_map(
      functools.partial(
          lru,
          max_seq_block_size=max_seq_block_size,
          min_seq_block_size=min_seq_block_size,
          pad_seq_to_min_block_size=pad_seq_to_min_block_size,
          pad_last_dim_to_128=pad_last_dim_to_128,
      ),
      mesh=pallas_sharding_spec.mesh,
      in_specs=args_specs,
      out_specs=pallas_sharding_spec.activations_sharding_spec,
      check_rep=False,
  )
  return f(*args)
