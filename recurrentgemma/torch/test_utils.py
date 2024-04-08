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
"""Tests for base layers."""

from typing import Any
import jax
import jax.numpy as jnp

from recurrentgemma import conversion

import torch


def generate_input(
    rng: jax.Array,
    input_shape: tuple[int, ...] | list[int],
    dtype: str,
    vocab_size: int,
) -> jax.Array:
  if dtype == "int32":
    assert vocab_size is not None
    # Tokens
    return jax.random.randint(rng, input_shape, minval=0, maxval=vocab_size)
  else:
    return jax.random.normal(rng, input_shape).astype(dtype)


def compare_jax_to_torch(
    jax_outputs: Any,
    torch_outputs: Any,
    tols: dict[str, float] | None = None,
) -> None:
  """Compares numerically Jax values to PyTorch values."""
  tols = tols or {}
  jax_leaves = jax.tree_util.tree_leaves(jax_outputs)
  torch_leaves = jax.tree_util.tree_leaves(
      torch_outputs,
      is_leaf=lambda x: isinstance(x, torch.Tensor),
  )
  assert len(jax_outputs) == len(torch_outputs)

  for jax_array, torch_array in zip(jax_leaves, torch_leaves):
    converted_array = conversion.jax_array_to_torch_tensor(jax_array)
    torch.testing.assert_close(torch_array, converted_array, **tols)


def numerically_compare_modules(
    jax_module,
    torch_module,
    uses_segment_pos: bool,
    has_cache: bool,
    input_shape: tuple[int, ...] | list[int],
    dtype: str,
    seed: int,
    num_unroll_steps: int = 2,
    vocab_size: int | None = None,
    tols: dict[str, float] | None = None,
) -> None:
  """Compares numerically Jax and PyTorch modules."""
  x_rng, y_rng, init_rng = jax.random.split(jax.random.PRNGKey(seed), 3)
  x = generate_input(x_rng, input_shape, dtype, vocab_size)
  segment_pos = jnp.repeat(jnp.arange(input_shape[1] // 2), 2, axis=0)[None]
  jax_args = [x, segment_pos] if uses_segment_pos else [x]
  torch_args = [conversion.jax_array_to_torch_tensor(xi) for xi in jax_args]

  params = jax_module.init(init_rng, *jax_args)["params"]
  torch_params = conversion.flax_params_to_pytorch_state_dict(params)
  torch_module.load_state_dict(torch_params)

  # Forward pass
  jax_outputs = jax_module.apply(dict(params=params), *jax_args)
  torch_outputs = torch_module(*torch_args)
  compare_jax_to_torch(jax_outputs, torch_outputs, tols)

  if not has_cache or num_unroll_steps == 0:
    return

  _, jax_cache = jax_outputs
  _, torch_cache = torch_outputs
  compare_jax_to_torch(jax_cache, torch_cache, tols)

  # Sampling
  segment_pos = jax_args[1][:, -1:] + 1
  y_shape = [input_shape[0], num_unroll_steps, *input_shape[2:]]
  y = generate_input(y_rng, y_shape, dtype, vocab_size)

  for i in range(num_unroll_steps):
    jax_args = [y[:, i : i + 1], segment_pos + i]
    torch_args = [conversion.jax_array_to_torch_tensor(xi) for xi in jax_args]

    jax_output, jax_cache = jax_module.apply(
        dict(params=params), *jax_args, jax_cache
    )
    torch_output, torch_cache = torch_module(*torch_args, torch_cache)

    compare_jax_to_torch(jax_output, torch_output, tols)
    compare_jax_to_torch(jax_cache, torch_cache, tols)
