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
"""Pallas implementation of the scan operation for a LRU."""
import functools
import math
from typing import Literal, NamedTuple, TypeVar, overload

import jax
from jax._src.lax.control_flow import for_loop
import jax.experimental.pallas as pl
import jax.numpy as jnp
from recurrentgemma.jax import complex_lib


T = TypeVar("T")
Spec = TypeVar("Spec", complex_lib.Complex, pl.BlockSpec, None)

LruPallasResiduals = tuple[
    complex_lib.RealOrComplex,  # a
    complex_lib.RealOrComplex,  # y
    complex_lib.RealOrComplex | None,  # h0
    bool,  # has_h0
]


def get_acc_dtype(
    x: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None,
    acc_float_dtype: jnp.dtype = jnp.float32,
) -> jnp.dtype:
  """Returns the accumulation dtype for the given inputs."""
  if h0 is not None:
    assert acc_float_dtype == h0.dtype
    return acc_float_dtype
  elif isinstance(x, complex_lib.Complex) or not jnp.iscomplexobj(x):
    return acc_float_dtype
  else:
    return (x.real.astype(acc_float_dtype) * 1j).dtype


def sequence_shard_index(
    seq_axis: str | None = None,
    seq_axis_index_groups: list[list[int]] | None = None,
) -> jax.Array:
  """Returns the correct sequence shard index for this device."""
  if seq_axis is None:
    return jnp.zeros([], dtype=jnp.int32)

  raw_axis_index = jax.lax.axis_index(seq_axis)
  if seq_axis_index_groups is None:
    return raw_axis_index

  index = [list(range(len(group))) for group in seq_axis_index_groups]
  seq_index = jnp.asarray(seq_axis_index_groups).flatten()
  index = jnp.asarray(index).flatten()
  return jnp.sum(index * (seq_index == raw_axis_index))


def multi_shard_correction(
    *,
    y: complex_lib.RealOrComplex,
    a_prod: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None,
    reverse: bool,
    h_last: complex_lib.RealOrComplex | None = None,
    a_prod_last: complex_lib.RealOrComplex | None = None,
    acc_float_dtype: jnp.dtype = jnp.float32,
    seq_axis: str | None = None,
    seq_axis_index_groups: list[list[int]] | None = None,
    shift_a_prod: bool = False,
    sync_h_last: bool = True,
) -> tuple[
    complex_lib.RealOrComplex,
    complex_lib.RealOrComplex,
    complex_lib.RealOrComplex,
]:
  """This codes corrects the result `y` from a single shard.

  The unrolled RNN computation can be written as:
  ```
  h_per_shard[shard_idx+1, local_t] = (
      prod(a_per_shard[shard_idx+1, :local_t]) * h_per_shard[shard_idx, -1]
      + some_function(a_per_shard[shard_idx+1, :], x_per_shard[shard_idx+1,:])
  )
  ```
  Since each shard has so far performed the computation using `local_h0=0`
  instead of `h_per_shard[shard_idx, -1]`, this function corrrects the result by
  adding the missing term
  `prod(a_per_shard[shard_idx+1, :local_t]) * h_per_shard[shard_idx, -1]`.

  Args:
    y: The LRU non-corrected accumulated result for this shard.
    a_prod: The cumulative product of `A` values for this shard.
    h0: The initial hidden state for the first shard.
    reverse: Whether the RNN is run in reverse.
    h_last: The optional last hidden state for this shard.
    a_prod_last: The optional last a product of `A` values for this shard.
    acc_float_dtype: The dtype to use for the accumulation.
    seq_axis: The name of the sequence axis.
    seq_axis_index_groups: The index groups for the sequence axis.
    shift_a_prod: Whether to shift the a_prod values by 1, which is needed for
      the backward pass.
    sync_h_last: Whether to sync the last hidden state across shards.

  Returns:
    The corrected result `y`, `h_last` and the corrected initial state `h0`.
  """
  num_seq_shards = get_num_seq_shards(seq_axis, seq_axis_index_groups)
  shard_index = sequence_shard_index(seq_axis)
  last_shard = 0 if reverse else (num_seq_shards - 1)
  acc_dtype = get_acc_dtype(y, h0, acc_float_dtype)

  index_field = list(range(num_seq_shards))
  if reverse:
    index_field = index_field[::-1]

  last_index = 0 if reverse else y.shape[1] - 1
  if h_last is None:
    h_last = y[:, last_index].astype(acc_dtype)
  if a_prod_last is None:
    a_prod_last = a_prod[:, last_index].astype(acc_dtype)

  if h0 is None:
    h0 = complex_lib.zeros_like(h_last)
  else:
    h0 = h0.astype(h_last.dtype)

  if num_seq_shards == 1:
    return y, h0  # pytype: disable=bad-return-type

  # Last hidden state and last product of `a` for every shard.
  h_last_gathered, a_prod_all = jax.lax.all_gather(
      (h_last, a_prod_last),
      seq_axis,
      axis_index_groups=seq_axis_index_groups,
  )

  # `h0_all` contains the uncorrected initial state for every shard.
  if reverse:
    h0_uncorrected = list(h_last_gathered[1:]) + [h0]
  else:
    h0_uncorrected = [h0] + list(h_last_gathered[:-1])

  # This discards the last shard `a_prod_all`, since it's not needed
  a_prod_all = list(a_prod_all)

  # This value iteratively computes the corrected `h0` for every shard.
  h0_shards = h0_uncorrected[index_field[0]]
  # This value stores the corrected `h0` for the current shard.
  h0_corrected = h0_shards

  for i in range(1, num_seq_shards):
    idx, idx_next = index_field[i - 1], index_field[i]
    h0_shards = a_prod_all[idx] * h0_shards + h0_uncorrected[idx_next]

    # Update only if this is the index for the current shard.
    cond = (shard_index == index_field[i]).astype(h0_shards.dtype)
    h0_corrected = cond * h0_shards + (1 - cond) * h0_corrected

  # This is needed for the backward pass.
  if shift_a_prod:
    ones_like_a = complex_lib.ones_like(a_prod[:, :1])
    if reverse:
      a_prod = [a_prod[:, 1:], ones_like_a]
    else:
      a_prod = [ones_like_a, a_prod[:, :-1]]
    a_prod: complex_lib.RealOrComplex = complex_lib.concatenate(a_prod, axis=1)

  # Updates and corrects `y`.
  y_corrected = y + h0_corrected[:, None].astype(a_prod.dtype) * a_prod

  # Updates and corrects `h_last`.
  idx = index_field[-1]
  h_last_corrected = a_prod_all[idx] * h0_shards + h_last_gathered[idx]

  if not sync_h_last:
    # In this case we zero out the `h_last` on all shards but the last.
    cond = (shard_index == last_shard).astype(h_last_corrected.dtype)
    h_last_corrected = cond * h_last_corrected

  return y_corrected, h_last_corrected, h0_corrected


class ShardingSpec(NamedTuple):
  """The sharding spec for running a Pallas kernel with sharded values."""

  mesh: jax.sharding.Mesh | None = None
  batch_axis_name: str | tuple[str, ...] | None = None
  sequence_axis_name: str | tuple[str, ...] | None = None
  activations_axis_name: str | tuple[str, ...] | None = None
  sequence_axis_index_groups: list[list[int]] | None = None

  @property
  def activations_sharding_spec(self) -> jax.sharding.PartitionSpec:
    return jax.sharding.PartitionSpec(
        self.batch_axis_name,
        self.sequence_axis_name,
        self.activations_axis_name,
    )

  @property
  def activations_sharding(self) -> jax.sharding.NamedSharding:
    return jax.sharding.NamedSharding(
        mesh=self.mesh,
        spec=self.activations_sharding_spec,
    )

  @property
  def rnn_state_sharding_spec(self) -> jax.sharding.PartitionSpec:
    return jax.sharding.PartitionSpec(
        self.batch_axis_name,
        self.activations_axis_name,
    )

  @property
  def rnn_state_sharding(self) -> jax.sharding.NamedSharding:
    return jax.sharding.NamedSharding(
        mesh=self.mesh,
        spec=self.rnn_state_sharding_spec,
    )


def get_num_seq_shards(
    seq_axis: str | None = None,
    seq_axis_index_groups: list[list[int]] | None = None,
) -> int:
  if seq_axis is None:
    return 1
  else:
    return jax.lax.psum(1, seq_axis, axis_index_groups=seq_axis_index_groups)


def pad_array_to_divisible(
    x: complex_lib.RealOrComplex,
    divisor: int,
    axis: int,
    pad_on_back: bool = True,
    value: float = 0.0,
) -> complex_lib.RealOrComplex:
  """Pads the variable `x` to have size along `axis` divisible by `divisor`."""
  if x.shape[axis] % divisor == 0:
    return x
  n = divisor - x.shape[axis] % divisor
  pad_shape = list(x.shape)
  pad_shape[axis] = n
  fill = jnp.full(pad_shape, value, dtype=x.dtype)
  if isinstance(x, complex_lib.Complex):
    fill = complex_lib.Complex(fill, jnp.zeros_like(fill))

  to_concat = [x, fill] if pad_on_back else [fill, x]
  return complex_lib.concatenate(to_concat, axis=axis)


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


def carry_dtype(dtype: jnp.dtype) -> jnp.dtype:
  if dtype in (jnp.bfloat16, jnp.float16):
    return jnp.float32
  else:
    return dtype


def maybe_wrap_in_complex(v: T, do_wrap: bool) -> T | complex_lib.Complex:
  if not do_wrap:
    return v
  return complex_lib.Complex(v, v)


def reverse_block_spec(spec: Spec, num_seq_blocks: int) -> Spec:
  """Reverses the order of accessing sequence axis tiles."""
  if spec is None:
    return None

  elif isinstance(spec, complex_lib.Complex):
    return complex_lib.Complex(
        real=reverse_block_spec(spec.real, num_seq_blocks),  # pytype: disable=wrong-arg-types
        imag=reverse_block_spec(spec.imag, num_seq_blocks),  # pytype: disable=wrong-arg-types
    )

  else:
    assert isinstance(spec, pl.BlockSpec)  # For typing...
    return pl.BlockSpec(
        block_shape=spec.block_shape,
        index_map=lambda b, d, s: spec.index_map(b, d, num_seq_blocks - 1 - s),
    )


def to_blocks(
    x: complex_lib.RealOrComplex | None,
    s: int,
) -> complex_lib.RealOrComplex | None:
  """Reshapes `x` such that it's last dim is equal to `s` by adding an extra axis."""
  if x is None:
    return None
  return complex_lib.reshape(x, (*x.shape[:-1], x.shape[-1] // s, s))


def from_blocks(
    x: complex_lib.RealOrComplex,
) -> complex_lib.RealOrComplex:
  """Reverse the effect of `to_blocks`."""
  return complex_lib.reshape(x, (*x.shape[:-2], -1))


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
    x: complex_lib.RealOrComplex,
    max_seq_block_size: int,
    min_seq_block_size: int,
    singleton_tile_size: int = 128,
) -> PallasKernelSpec:
  """Retrurns the correct Pallas grid."""
  batch_size, seq_len, dim = x.shape

  # Special logic based on the devices
  device = jax.devices()[0]

  if dim % singleton_tile_size != 0:
    raise ValueError(
        f"{dim=} has to be divisible by {singleton_tile_size=} for pallas scan."
    )

  if device.platform == "gpu":
    # For GPUs we always use the full sequence length
    max_seq_block_size = seq_len

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
  x_spec = pl.BlockSpec(
      block_shape=x_shape,
      index_map=lambda b, d, s: (b, s, d, 0),
  )
  # Same as x, but without the sequence dimension
  h_shape = x_shape[:1] + x_shape[2:]
  # If we don't return h_last, this can be instead:
  # h_spec = pl.BlockSpec(lambda b, d, s: (0, 0, 0), h_shape)
  h_spec = pl.BlockSpec(
      block_shape=h_shape,
      index_map=lambda b, d, s: (b, d, 0),
  )
  hs_spec = pl.BlockSpec(
      block_shape=(1, *h_shape),
      index_map=lambda b, d, s: (s, b, d, 0),
  )
  h0_spec = pl.BlockSpec(
      block_shape=h_shape,
      index_map=lambda b, d, s: (b, d, 0),
  )

  return x_spec, h_spec, hs_spec, h0_spec


def make_block_shape_and_dtypes(
    x: complex_lib.RealOrComplex,
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
  # If we don't return h_last, this can be instead:
  # h_size = (
  #     kernel_spec.batch_tile_size,
  #     kernel_spec.dim_tile_size,
  #     kernel_spec.singleton_tile_size,
  # )
  # h_shape = jax.ShapeDtypeStruct(shape=h_size, dtype=dtype)
  h_shape = jax.ShapeDtypeStruct(shape=shape[:1] + shape[2:], dtype=dtype)
  hs_size = (kernel_spec.seq_grid_size, x.shape[0], *shape[2:])
  hs_shape = jax.ShapeDtypeStruct(shape=hs_size, dtype=dtype)
  h0_shape = jax.ShapeDtypeStruct(shape=shape[:1] + shape[2:], dtype=dtype)
  return x_shape, h_shape, hs_shape, h0_shape


def initialize_carry(
    h_carry_ref: complex_lib.RealOrComplex,
    a_prod_carry_ref: complex_lib.RealOrComplex | None,
    h_init_ref: complex_lib.RealOrComplex | None,
):
  """Initializes the accumulator and product carries."""

  def init_h0():
    if h_init_ref is None:
      h_carry_ref[:] = complex_lib.zeros_like(h_carry_ref)
    else:
      h_carry_ref[:] = h_init_ref[:].astype(h_carry_ref.dtype)

    if a_prod_carry_ref is not None:
      a_prod_carry_ref[:] = complex_lib.ones_like(a_prod_carry_ref)

  # Initialize to zeros only the first sequence block
  jax.lax.cond(pl.program_id(2) == 0, init_h0, lambda: None)


def linear_rnn_loop_body(
    i: int,
    h_and_a_prod_carry_refs: tuple[
        complex_lib.RealOrComplex,
        complex_lib.RealOrComplex | None
    ],
    x_ref: complex_lib.RealOrComplex,
    a_ref: complex_lib.RealOrComplex,
    y_ref: complex_lib.RealOrComplex,
    a_prod_ref: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    backprop: bool = False,
) -> None:
  """Evaluates a single step of a linear RNN loop."""
  # Compute the correct indices.
  seq_len = x_ref.shape[1]

  if backprop:
    a_idx = i if reverse else (seq_len - 1 - i)
    x_idx = a_idx + (1 if reverse else -1)
  else:
    a_idx = (seq_len - 2 - i) if reverse else (i + 1)
    x_idx = a_idx

  h_carry_ref, a_prod_carry_ref = h_and_a_prod_carry_refs

  # RNN
  h_t = h_carry_ref[:]
  a_t = a_ref[:, a_idx].astype(h_t.dtype)
  x_t = x_ref[:, x_idx].astype(h_t.dtype)
  h_next = a_t * h_t + x_t

  # Store
  h_carry_ref[:] = h_next
  y_ref[:, x_idx] = h_next.astype(y_ref.dtype)

  if a_prod_ref is not None:
    assert a_prod_carry_ref is not None
    # Product
    a_prod_t = a_prod_carry_ref[:]
    a_prod_next = a_prod_t * a_t
    # Store
    a_prod_carry_ref[:] = a_prod_next
    a_prod_ref[:, a_idx] = a_prod_next.astype(a_prod_ref.dtype)


def linear_rnn_pallas_kernel(
    x_ref: complex_lib.RealOrComplex,
    a_ref: complex_lib.RealOrComplex,
    h_init_ref: complex_lib.RealOrComplex | None,
    y_ref: complex_lib.RealOrComplex,
    h_carry_ref: complex_lib.RealOrComplex,
    a_prod_ref: complex_lib.RealOrComplex | None = None,
    a_prod_carry_ref: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    backprop: bool = False,
):
  """A Pallas kernel for computing a linear RNN."""
  if (a_prod_ref is None) != (a_prod_carry_ref is None):
    raise ValueError(
        "a_prod_ref and a_prod_carry_ref have to be both None or both not None."
    )

  # Useful indices
  seq_len = x_ref.shape[1]
  last_idx = 0 if reverse else seq_len - 1
  first_idx = (seq_len - 1) - last_idx

  # Initialize carries
  initialize_carry(h_carry_ref, a_prod_carry_ref, h_init_ref)

  # Compute outside of the for loop the first step, since it contains special
  # logic for the backprop case.
  if backprop:
    idx, a_0 = last_idx, jnp.ones([], dtype=h_carry_ref.dtype)
  else:
    idx, a_0 = first_idx, a_ref[:, first_idx].astype(h_carry_ref.dtype)

  h_carry = a_0 * h_carry_ref[:] + x_ref[:, idx].astype(h_carry_ref.dtype)
  y_ref[:, idx] = h_carry.astype(y_ref.dtype)

  a_prod_carry = None
  if a_prod_carry_ref is not None:
    a_prod_carry = a_prod_carry_ref[:] * a_0
    if not backprop:
      assert a_prod_ref is not None
      a_prod_ref[:, idx] = a_prod_carry.astype(a_prod_ref.dtype)

  # Execute loop
  (h_n, a_prod_n) = for_loop.for_loop(  # pytype: disable=wrong-arg-types
      seq_len - 1,
      functools.partial(
          linear_rnn_loop_body,
          x_ref=x_ref,
          a_ref=a_ref,
          y_ref=y_ref,
          a_prod_ref=a_prod_ref,
          reverse=reverse,
          backprop=backprop,
      ),
      (h_carry, a_prod_carry),
  )

  if backprop:
    a_n = a_ref[:, first_idx].astype(h_carry.dtype)
    h_n = h_n * a_n

    if a_prod_ref is not None:
      a_prod_n = a_prod_n * a_n
      a_prod_ref[:, first_idx] = a_prod_n.astype(a_prod_ref.dtype)

  # Store the last values of the loop in the carry references
  h_carry_ref[:] = h_n
  if a_prod_ref is not None:
    a_prod_carry_ref[:] = a_prod_n


@overload
def linear_rnn_pallas_call(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None,
    reverse: bool,
    kernel_spec: PallasKernelSpec,
    compute_a_prod: Literal[True],
    backprop: bool,
) -> tuple[
    complex_lib.RealOrComplex,
    complex_lib.RealOrComplex,
    complex_lib.RealOrComplex,
    complex_lib.RealOrComplex
]:
  ...


@overload
def linear_rnn_pallas_call(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None,
    reverse: bool,
    kernel_spec: PallasKernelSpec,
    compute_a_prod: Literal[False],
    backprop: bool,
) -> tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex]:
  ...


def linear_rnn_pallas_call(
    x,
    a,
    h0,
    reverse,
    kernel_spec,
    compute_a_prod,
    backprop,
):
  """A convience wrapper for the linear_rnn_pallas_kernel."""
  # Compute the output shapes.
  x_shape, h_shape, _, _ = make_block_shape_and_dtypes(x, kernel_spec)
  if compute_a_prod:
    out_shapes = [x_shape, h_shape, x_shape, h_shape]
  else:
    out_shapes = [x_shape, h_shape]

  # Make input and output specs.
  x_spec, h_spec, *_ = make_block_specs(kernel_spec)
  in_specs = [x_spec, x_spec, (None if h0 is None else h_spec)]
  if compute_a_prod:
    out_specs = [x_spec, h_spec, x_spec, h_spec]
  else:
    out_specs = [x_spec, h_spec]

  if reverse != backprop:
    reverse_seq = functools.partial(
        reverse_block_spec, num_seq_blocks=kernel_spec.seq_grid_size
    )
    in_specs = list(map(reverse_seq, in_specs))
    out_specs = list(map(reverse_seq, out_specs))

  # Wrap the specs if we are using our custom Complex class.
  use_custom_complex = isinstance(x, complex_lib.Complex)
  wrap = functools.partial(maybe_wrap_in_complex, do_wrap=use_custom_complex)
  in_specs, out_specs, out_shapes = jax.tree.map(
      wrap, (in_specs, out_specs, out_shapes)
  )

  # Prepare the arguments in correct format.
  args = (x, a, h0)
  args = [to_blocks(arg, kernel_spec.singleton_tile_size) for arg in args]

  # Execute the Pallas call.
  outputs = pl.pallas_call(
      functools.partial(
          linear_rnn_pallas_kernel,
          reverse=reverse,
          backprop=backprop,
      ),
      out_shape=out_shapes,
      in_specs=in_specs,
      out_specs=out_specs,
      grid=kernel_spec.grid,
  )(*args)

  # Return arguments in correct format.
  return tuple([from_blocks(out) for out in outputs])


def linear_rnn_shard_corrected_pallas_call(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None,
    reverse: bool,
    backprop: bool,
    kernel_spec: PallasKernelSpec,
    seq_axis: str | None = None,
    seq_axis_index_groups: list[list[int]] | None = None,
) -> tuple[
    complex_lib.RealOrComplex,
    complex_lib.RealOrComplex,
    complex_lib.RealOrComplex | None
]:
  """A call to the linear RNN Pallas kernel with shard correction."""
  num_seq_shards = get_num_seq_shards(seq_axis, seq_axis_index_groups)

  if num_seq_shards == 1:
    y, h_last = linear_rnn_pallas_call(
        x=x,
        a=a,
        h0=h0,
        reverse=reverse,
        kernel_spec=kernel_spec,
        compute_a_prod=False,
        backprop=backprop,
    )
    return y, h_last, h0  # pytype: disable=bad-return-type

  else:
    y, h_last, a_prod, a_prod_last = linear_rnn_pallas_call(
        x=x,
        a=a,
        h0=None,
        reverse=reverse,
        kernel_spec=kernel_spec,
        compute_a_prod=True,
        backprop=backprop,
    )

    return multi_shard_correction(  # pytype: disable=bad-return-type
        y=y,
        a_prod=a_prod,
        h0=h0,
        reverse=(not reverse) if backprop else reverse,
        h_last=h_last,
        a_prod_last=a_prod_last,
        acc_float_dtype=carry_dtype(y.dtype),
        seq_axis=seq_axis,
        seq_axis_index_groups=seq_axis_index_groups,
        shift_a_prod=backprop,
        sync_h_last=not backprop,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=[3, 4, 5, 6])
def _lru(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None,
    reverse: bool,
    kernel_spec: PallasKernelSpec,
    seq_axis: str | None = None,
    seq_axis_index_groups: list[list[int]] | None = None,
) -> tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex]:
  """Runs the RNN forward pass without residuals for backprop."""
  y, h_last, _ = linear_rnn_shard_corrected_pallas_call(
      x=x,
      a=a,
      h0=h0,
      reverse=reverse,
      backprop=False,
      kernel_spec=kernel_spec,
      seq_axis=seq_axis,
      seq_axis_index_groups=seq_axis_index_groups,
  )
  return y, h_last  # pytype: disable=bad-return-type


def _lru_fwd(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None,
    reverse: bool,
    kernel_spec: PallasKernelSpec,
    seq_axis: str | None,
    seq_axis_index_groups: list[list[int]] | None = None,
) -> tuple[complex_lib.RealOrComplex, LruPallasResiduals]:
  """Runs the RNN forward pass and corrects for any sequence sharding."""
  y, h_last, h0_corrected = linear_rnn_shard_corrected_pallas_call(
      x=x,
      a=a,
      h0=h0,
      reverse=reverse,
      backprop=False,
      kernel_spec=kernel_spec,
      seq_axis=seq_axis,
      seq_axis_index_groups=seq_axis_index_groups,
  )
  return (y, h_last), (y, a, h0_corrected, h0 is not None)


def _lru_bwd(
    reverse: bool,
    kernel_spec: PallasKernelSpec,
    seq_axis: str | None,
    seq_axis_index_groups: list[list[int]] | None,
    res: LruPallasResiduals,
    dy_and_dh_last: tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex],
) -> tuple[
    complex_lib.RealOrComplex,
    complex_lib.RealOrComplex,
    complex_lib.RealOrComplex | None,
]:
  """Runs the RNN backward pass and corrects for any sequence sharding."""
  dy, dh_last = dy_and_dh_last
  num_seq_shards = get_num_seq_shards(seq_axis, seq_axis_index_groups)

  if num_seq_shards > 1:
    # This is needed because of how Jax propagates gradients in `shard_map`.
    dh_last = jax.lax.psum(
        dh_last,
        seq_axis,
        axis_index_groups=seq_axis_index_groups,
    )

  # Unpack
  y, a, h0, has_h0 = res

  # Conjugate for our custom Complex class.
  if isinstance(a, complex_lib.Complex):
    a = complex_lib.conjugate(a)

  dx, dh0, _ = linear_rnn_shard_corrected_pallas_call(
      x=dy,
      a=a,
      h0=dh_last,
      reverse=reverse,
      backprop=True,
      kernel_spec=kernel_spec,
      seq_axis=seq_axis,
      seq_axis_index_groups=seq_axis_index_groups,
  )

  # Compute the outputs shifted by 1.
  if h0 is None:
    h0 = complex_lib.zeros_like(y[:, 0])
  h0 = h0[:, None].astype(y.dtype)

  y_shifted = [y[:, 1:], h0] if reverse else [h0, y[:, :-1]]
  y_shifted = complex_lib.concatenate(y_shifted, axis=1)
  if isinstance(dx, complex_lib.Complex):
    y_shifted = complex_lib.conjugate(y_shifted)

  da = dx * y_shifted

  return dx, da, (dh0 if has_h0 else None)  # pytype: disable=bad-return-type


_lru.defvjp(_lru_fwd, _lru_bwd)


def pallas_lru(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    seq_axis: str | None = None,
    seq_axis_index_groups: list[list[int]] | None = None,
    max_seq_block_size: int = 256,
    min_seq_block_size: int = 16,
    pad_seq_to_min_block_size: bool = True,
    pad_last_dim_to_128: bool = False,
) -> tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex]:
  """Runs the LRU scan using a Pallas kernel."""
  *_, t, d = x.shape
  if pad_seq_to_min_block_size:
    x = pad_array_to_divisible(
        x,
        divisor=min_seq_block_size,
        axis=-2,
        pad_on_back=not reverse,
        value=0.0,
    )
    a = pad_array_to_divisible(
        a,
        divisor=min_seq_block_size,
        axis=-2,
        pad_on_back=not reverse,
        value=1.0,
    )

  if pad_last_dim_to_128:
    x, a, h0 = jax.tree.map(
        functools.partial(pad_array_to_divisible, divisor=128, axis=-1),
        (x, a, h0),
    )

  native_complex = False
  if not isinstance(x, complex_lib.Complex) and jnp.iscomplexobj(x):
    native_complex = True
    x, a, h0 = jax.tree.map(complex_lib.to_custom_complex, x, a, h0)

  kernel_spec = compute_pallas_kernel_spec(
      x=x,
      max_seq_block_size=max_seq_block_size,
      min_seq_block_size=min_seq_block_size,
  )

  y, h_last = _lru(
      x=x,
      a=a,
      h0=h0,
      reverse=reverse,
      kernel_spec=kernel_spec,
      seq_axis=seq_axis,
      seq_axis_index_groups=seq_axis_index_groups,
  )

  if reverse:
    y = y[..., -t:, :d]
  else:
    y = y[..., :t, :d]

  if native_complex:
    assert isinstance(y, complex_lib.Complex)
    assert isinstance(h_last, complex_lib.Complex)
    return y.to_numpy(), h_last.to_numpy()
  else:
    return y, h_last


def lru_pallas_scan(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    seq_axis: str | None = None,
    seq_axis_index_groups: list[list[int]] | None = None,
    max_seq_block_size: int = 256,
    min_seq_block_size: int = 16,
    pad_seq_to_min_block_size: bool = True,
    pad_last_dim_to_128: bool = False,
) -> tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex]:
  """Runs the LRU scan using a Pallas kernel."""
  *_, t, d = x.shape
  if pad_seq_to_min_block_size:
    x = pad_array_to_divisible(
        x,
        divisor=min_seq_block_size,
        axis=-2,
        pad_on_back=not reverse,
        value=0.0,
    )
    a = pad_array_to_divisible(
        a,
        divisor=min_seq_block_size,
        axis=-2,
        pad_on_back=not reverse,
        value=1.0,
    )

  if pad_last_dim_to_128:
    x, a, h0 = jax.tree.map(
        functools.partial(pad_array_to_divisible, divisor=128, axis=-1),
        (x, a, h0),
    )

  native_complex = False
  if not isinstance(x, complex_lib.Complex) and jnp.iscomplexobj(x):
    native_complex = True
    x, a, h0 = jax.tree.map(complex_lib.to_custom_complex, (x, a, h0))

  kernel_spec = compute_pallas_kernel_spec(
      x=x,
      max_seq_block_size=max_seq_block_size,
      min_seq_block_size=min_seq_block_size,
  )

  y, h_last = _lru(
      x=x,
      a=a,
      h0=h0,
      reverse=reverse,
      kernel_spec=kernel_spec,
      seq_axis=seq_axis,
      seq_axis_index_groups=seq_axis_index_groups,
  )

  if reverse:
    y = y[..., -t:, :d]
  else:
    y = y[..., :t, :d]

  if native_complex:
    assert isinstance(y, complex_lib.Complex)
    assert isinstance(h_last, complex_lib.Complex)
    return y.to_numpy(), h_last.to_numpy()
  else:
    return y, h_last
