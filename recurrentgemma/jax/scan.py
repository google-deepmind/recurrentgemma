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
"""All scan implementations."""

import functools
from typing import Literal, overload

from absl import logging
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
from recurrentgemma import common
from recurrentgemma.jax import complex_lib
from recurrentgemma.jax import pallas


ShardingSpec = pallas.ShardingSpec
lru_pallas_scan = pallas.lru_pallas_scan


def resolve_scan_type(scan_type: common.ScanType) -> common.ScanType:
  """Resolves the scan type if its AUTO."""
  match scan_type:
    case common.ScanType.AUTO:
      if jax.local_devices()[0].platform == "tpu":
        return common.ScanType.LINEAR_PALLAS
      else:
        return common.ScanType.LINEAR_NATIVE
    case _:
      return scan_type


@overload
def lru_linear_scan(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    return_a_prod: Literal[False] = False,
    acc_float_dtype: jnp.dtype = jnp.float32,
    unroll: int = 1,
) -> tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex]:
  ...


@overload
def lru_linear_scan(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    return_a_prod: Literal[True] = True,
    acc_float_dtype: jnp.dtype = jnp.float32,
    unroll: int = 1,
) -> tuple[
    tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex],
    tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex],
]:
  ...


def lru_linear_scan(
    x,
    a,
    h0=None,
    reverse=False,
    return_a_prod=False,
    acc_float_dtype=jnp.float32,
    unroll=1,
):
  """Computes a linear scan over the second axis of the inputs."""
  acc_dtype = pallas.get_acc_dtype(x, h0, acc_float_dtype)

  def body_fn(carry, current_inputs):
    h_prev, a_prev = carry
    x_t, a_t = current_inputs
    h_t = a_t.astype(acc_dtype) * h_prev + x_t.astype(acc_dtype)
    h_out = h_t.astype(x.dtype)

    if return_a_prod:
      assert a_prev is not None
      a_t = a_t.astype(acc_dtype) * a_prev
      a_out = a_t.astype(x.dtype)
    else:
      a_t, a_out = None, None

    return (h_t, a_t), (h_out, a_out)

  h0 = complex_lib.zeros_like(x[:, 0], acc_dtype) if h0 is None else h0
  a0 = complex_lib.ones_like(h0) if return_a_prod else None

  scan_fn = jax.vmap(
      lambda init, xs: jax.lax.scan(
          body_fn,
          init=init,
          xs=xs,
          unroll=unroll,
          reverse=reverse,
      ),
      in_axes=0,
      out_axes=0,
  )
  (h_last, a_prod_last), (y, a_prod) = scan_fn((h0, a0), (x, a))

  if return_a_prod:
    return (y, h_last), (a_prod, a_prod_last)
  else:
    return (y, h_last)


@overload
def lru_associative_scan(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    acc_float_dtype: jnp.dtype = jnp.float32,
    return_a_prod: Literal[False] = False,
) -> tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex]:
  ...


@overload
def lru_associative_scan(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    acc_float_dtype: jnp.dtype = jnp.float32,
    return_a_prod: Literal[True] = True,
) -> tuple[
    tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex],
    tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex],
]:
  ...


def lru_associative_scan(
    x,
    a,
    h0=None,
    reverse=False,
    acc_float_dtype=jnp.float32,
    return_a_prod=False,
):
  """Computes an associative scan over the second axis of the inputs."""
  acc_dtype = pallas.get_acc_dtype(x, h0, acc_float_dtype)

  def lru_associative_bin_op(
      element_i: tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex],
      element_j: tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex],
  ) -> tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex]:
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j

  orig_dtype = x.dtype
  x = x.astype(acc_dtype)
  a = a.astype(acc_dtype)

  # Optionally concatenate the hidden state.
  if h0 is not None:
    if reverse:
      a = complex_lib.concatenate([a, complex_lib.ones_like(a[:, :1])], axis=1)
      x = complex_lib.concatenate([x, h0[:, None]], axis=1)
    else:
      a = complex_lib.concatenate([complex_lib.ones_like(a[:, :1]), a], axis=1)
      x = complex_lib.concatenate([h0[:, None], x], axis=1)

  a_prod, y = jax.lax.associative_scan(
      lru_associative_bin_op,
      (a, x),
      axis=x.ndim - 2,
      reverse=reverse,
  )

  # Remove the first element if there was a hidden state.
  if h0 is not None:
    y = y[:, :-1] if reverse else y[:, 1:]
    a_prod = a_prod[:, :-1] if reverse else a_prod[:, 1:]

  last_index = 0 if reverse else -1
  h_last = y[:, last_index]
  a_prod_last = a_prod[:, last_index]

  y_out = y.astype(orig_dtype)
  a_prod_out = a_prod.astype(orig_dtype)

  if return_a_prod:
    return (y_out, h_last), (a_prod_out, a_prod_last)
  else:
    return (y_out, h_last)


def single_shard_rnn_scan(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    scan_type: common.ScanType = common.ScanType.AUTO,
    acc_float_dtype: jnp.dtype = jnp.float32,
    seq_axis: str | None = None,
    seq_axis_index_groups: list[list[int]] | None = None,
    unroll: int = 1,
    pallas_max_seq_block_size: int = 256,
    pallas_min_seq_block_size: int = 16,
    pallas_pad_seq_to_min_block_size: bool = True,
    pallas_pad_last_dim_to_128: bool = False,
) -> tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex]:
  """Runs the recurrence of a linear RNN on a single (sequence) shard.

  Args:
    x: The input sequence.
    a: The diagonal of the recurrence matrix `A`.
    h0: The initial hidden state.
    reverse: Whether to run the scan in reverse.
    scan_type: Which scan implementation to use.
    acc_float_dtype: The data type for the accumulation.
    seq_axis: The sequence axis name (for sharding).
    seq_axis_index_groups: The sequence axis index groups if any.
    unroll: The unroll parameter for the linear native scan.
    pallas_max_seq_block_size: The maximum sequence block size for the linear
      Pallas scan.
    pallas_min_seq_block_size: The minimum sequence block size for the linear
      Pallas scan.
    pallas_pad_seq_to_min_block_size: Whether to pad the sequence to the minimum
      block size for the linear Pallas scan.
    pallas_pad_last_dim_to_128: Whether to pad the last dimension to 128 for the
      linear Pallas scan.

  Returns:
    The output of the linear recurrence together with the final hidden state.
  """
  assert x.ndim == 3
  assert a.shape == x.shape[-a.ndim :]
  assert a.dtype == x.dtype
  assert type(a) is type(x)

  if seq_axis is None:
    num_seq_shards = 1
  else:
    num_seq_shards = jax.lax.psum(
        1,
        seq_axis,
        axis_index_groups=seq_axis_index_groups,
    )

  match resolve_scan_type(scan_type):

    case common.ScanType.LINEAR_PALLAS:
      # Using a Pallas linear scan kernel.
      if acc_float_dtype != jnp.float32:
        raise ValueError(f"Unsupported accumulation dtype: {acc_float_dtype}.")

      # The multi-shard correction is already handled inside `pallas_lru`.
      return pallas.lru_pallas_scan(  # pytype: disable=bad-return-type
          x=x,
          a=a,
          h0=h0,
          reverse=reverse,
          seq_axis=seq_axis,
          seq_axis_index_groups=seq_axis_index_groups,
          min_seq_block_size=pallas_min_seq_block_size,
          max_seq_block_size=pallas_max_seq_block_size,
          pad_seq_to_min_block_size=pallas_pad_seq_to_min_block_size,
          pad_last_dim_to_128=pallas_pad_last_dim_to_128,
      )

    case common.ScanType.LINEAR_NATIVE:
      # Using native Jax linear scan.
      if num_seq_shards == 1:
        return lru_linear_scan(  # pytype: disable=bad-return-type
            x=x,
            a=a,
            h0=h0,
            reverse=reverse,
            acc_float_dtype=acc_float_dtype,
            unroll=unroll,
        )
      else:
        (y, h_last), (a_prod, a_prod_last) = lru_linear_scan(
            x=x,
            a=a,
            h0=None,
            reverse=reverse,
            return_a_prod=True,
            acc_float_dtype=acc_float_dtype,
            unroll=unroll,
        )
        y, h_last, _ = pallas.multi_shard_correction(
            y=y,
            a_prod=a_prod,
            h0=h0,
            reverse=reverse,
            h_last=h_last,
            a_prod_last=a_prod_last,
            acc_float_dtype=acc_float_dtype,
            seq_axis=seq_axis,
            seq_axis_index_groups=seq_axis_index_groups,
        )
        return y, h_last  # pytype: disable=bad-return-type

    case common.ScanType.ASSOCIATIVE_NATIVE:
      # Using native Jax associative scan.
      if num_seq_shards == 1:
        return lru_associative_scan(  # pytype: disable=bad-return-type
            x=x,
            a=a,
            h0=h0,
            reverse=reverse,
            acc_float_dtype=acc_float_dtype,
        )

      else:
        (y, h_last), (a_prod, a_prod_last) = lru_associative_scan(
            x=x,
            a=a,
            h0=None,
            reverse=reverse,
            acc_float_dtype=acc_float_dtype,
            return_a_prod=True,
        )
        y, h_last, _ = pallas.multi_shard_correction(
            y=y,
            a_prod=a_prod,
            h0=h0,
            reverse=reverse,
            h_last=h_last,
            a_prod_last=a_prod_last,
            acc_float_dtype=acc_float_dtype,
            seq_axis=seq_axis,
            seq_axis_index_groups=seq_axis_index_groups,
        )
        return y, h_last  # pytype: disable=bad-return-type

    case _:
      raise ValueError(f"Unsupported scan type: {scan_type}.")


def linear_scan(
    x: complex_lib.RealOrComplex,
    a: complex_lib.RealOrComplex,
    h0: complex_lib.RealOrComplex | None = None,
    reverse: bool = False,
    scan_type: common.ScanType = common.ScanType.AUTO,
    acc_float_dtype: jnp.dtype = jnp.float32,
    sharding_spec: pallas.ShardingSpec | None = None,
    unroll: int = 1,
    pallas_max_seq_block_size: int = 256,
    pallas_min_seq_block_size: int = 16,
    pallas_pad_seq_to_min_block_size: bool = True,
    pallas_pad_last_dim_to_128: bool = False,
) -> tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex]:
  """Runs the recurrence of a linear RNN on a single (sequence) shard.

  Args:
    x: The input sequence.
    a: The diagonal of the recurrence matrix `A`.
    h0: The initial hidden state.
    reverse: Whether to run the scan in reverse.
    scan_type: Which scan implementation to use.
    acc_float_dtype: The precision data type for the accumulation.
    sharding_spec: Sharding spec for running Pallas on sharded values.
    unroll: The unroll parameter for the linear native scan.
    pallas_max_seq_block_size: The maximum sequence block size for the linear
      Pallas scan.
    pallas_min_seq_block_size: The minimum sequence block size for the linear
      Pallas scan.
    pallas_pad_seq_to_min_block_size: Whether to pad the sequence to the minimum
      block size for the linear Pallas scan.
    pallas_pad_last_dim_to_128: Whether to pad the last dimension to 128 for the
      linear Pallas scan.

  Returns:
    The output of the linear recurrence together the final hiddenstate.
  """
  last_index = 0 if reverse else -1
  acc_dtype = pallas.get_acc_dtype(x, h0, acc_float_dtype)
  scan_type = resolve_scan_type(scan_type)

  if x.shape[1] == 1:
    assert a.shape[1] == 1

    logging.info("Running RNN in sampling mode.")

    if h0 is None:
      return x, x[:, 0].astype(acc_dtype)

    else:
      y = a.astype(acc_dtype) * h0[:, None] + x.astype(acc_dtype)
      return y.astype(x.dtype), y[:, last_index]

  elif sharding_spec is None:

    logging.info("Running RNN scan on a single shard in mode %s.", scan_type)
    return single_shard_rnn_scan(  # pytype: disable=bad-return-type
        x=x,
        a=a,
        h0=h0,
        reverse=reverse,
        scan_type=scan_type,
        acc_float_dtype=acc_float_dtype,
        unroll=unroll,
        pallas_max_seq_block_size=pallas_max_seq_block_size,
        pallas_min_seq_block_size=pallas_min_seq_block_size,
        pallas_pad_seq_to_min_block_size=pallas_pad_seq_to_min_block_size,
        pallas_pad_last_dim_to_128=pallas_pad_last_dim_to_128,
    )

  elif sharding_spec.mesh is None:

    logging.info("Running RNN scan under `pmap` in mode %s.", scan_type)
    return single_shard_rnn_scan(  # pytype: disable=bad-return-type
        x=x,
        a=a,
        h0=h0,
        reverse=reverse,
        scan_type=scan_type,
        acc_float_dtype=acc_float_dtype,
        unroll=unroll,
        seq_axis=sharding_spec.sequence_axis_name,
        seq_axis_index_groups=sharding_spec.sequence_axis_index_groups,
        pallas_max_seq_block_size=pallas_max_seq_block_size,
        pallas_min_seq_block_size=pallas_min_seq_block_size,
        pallas_pad_seq_to_min_block_size=pallas_pad_seq_to_min_block_size,
        pallas_pad_last_dim_to_128=pallas_pad_last_dim_to_128,
    )

  else:
    logging.info("Running RNN scan under `pjit` in mode %s.", scan_type)

    f = shard_map.shard_map(
        functools.partial(
            single_shard_rnn_scan,
            reverse=reverse,
            scan_type=scan_type,
            acc_float_dtype=acc_float_dtype,
            unroll=unroll,
            seq_axis=sharding_spec.sequence_axis_name,
            seq_axis_index_groups=sharding_spec.sequence_axis_index_groups,
            pallas_max_seq_block_size=pallas_max_seq_block_size,
            pallas_min_seq_block_size=pallas_min_seq_block_size,
            pallas_pad_seq_to_min_block_size=pallas_pad_seq_to_min_block_size,
            pallas_pad_last_dim_to_128=pallas_pad_last_dim_to_128,
        ),
        mesh=sharding_spec.mesh,
        in_specs=(
            sharding_spec.activations_sharding_spec,
            sharding_spec.activations_sharding_spec,
            sharding_spec.rnn_state_sharding_spec,
        ),
        out_specs=(
            sharding_spec.activations_sharding_spec,
            sharding_spec.rnn_state_sharding_spec,
        ),
        check_rep=False,
    )
    return f(x, a, h0)
