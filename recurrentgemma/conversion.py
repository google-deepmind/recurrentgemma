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
"""Utilities for conversion between Flax and PyTorch."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch


def jax_array_to_torch_tensor(x: jax.Array) -> torch.Tensor:
  """Converts a JAX array to a PyTorch Tensor."""
  if x.dtype == jnp.bfloat16:
    # We cannot directly convert bf16 to numpy, so we temporary cast to float32.
    x = x.astype(np.float32)
    dtype = torch.bfloat16
  else:
    dtype = getattr(torch, str(x.dtype))

  return torch.tensor(np.array(x), dtype=dtype)


def torch_tensor_to_jax_array(x: torch.Tensor) -> jax.Array:
  """Converts a PyTorch Tensor to a JAX array."""
  if x.dtype == torch.bfloat16:
    x = x.astype(jnp.float32)
    dtype = jnp.bfloat16
  else:
    dtype = str(x.dtype).split(".")[1]

  return jnp.asarray(x.numpy(), dtype=dtype)


def flatten_nested_dict(
    nested_dict: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, jax.Array]:
  """Recursively flattens a nested dictionary."""
  flat_dict = {}
  for key, value in nested_dict.items():
    prefix_and_key = prefix + key
    if isinstance(value, dict):
      flat_dict.update(flatten_nested_dict(value, prefix_and_key + "."))
    else:
      flat_dict[prefix_and_key] = value

  return flat_dict


def flax_params_to_pytorch_state_dict(
    params: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
  """Converts a Flax params dict to a PyTorch state dict."""
  torch_state = {}
  for key, value in flatten_nested_dict(params).items():
    # Map parameter names from Flax nn.Linear to PyTorch nn.Linear.
    key = key.replace("kernel", "weight")

    # Convert to PyTorch Tensor.
    value = jax_array_to_torch_tensor(value)

    if key.endswith("weight") and value.ndim == 2 and "conv1d" not in key:
      # Different axis convention for Linear layers.
      value = value.T

    torch_state[key] = value

  return torch_state


def pytorch_state_dict_to_flax_params(
    state_dict: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
  """Converts a PyTorch state dict to a Flax params dict."""
  flax_params = dict(params=dict())

  for key, value in state_dict.items():
    # Map parameter names from PyTorch nn.Linear to Flax nn.Linear.
    key = key.replace("weight", "kernel")

    # Convert to JAX array (handling potential dtype differences)
    value = torch_tensor_to_jax_array(value)

    if key.endswith("kernel") and value.ndim == 2 and "conv1d" not in key:
      # Reverse the transposition for Linear layers.
      value = value.T

    # Restore nested structure (assumes the structure is known beforehand)
    path = key.split(".")
    assert path[0] == "params" and path[1] in (
        "blocks",
        "embedder",
        "final_norm",
    )

    current_dict = flax_params[path[0]]
    if path[1] == "blocks":
      next_dict_key = f"{path[1]}.{path[2]}"
      i = 3
    else:
      next_dict_key = path[1]
      i = 2

    for path_i in path[i:]:
      if next_dict_key not in current_dict:
        current_dict[next_dict_key] = {}
      current_dict = current_dict[next_dict_key]
      next_dict_key = path_i

    current_dict[next_dict_key] = value

  return flax_params
