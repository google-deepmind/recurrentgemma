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
"""Utility functions for loading + saving parameters from a checkpoint."""

from collections.abc import Mapping
from typing import Any
import jax
from jax.experimental import mesh_utils
import orbax.checkpoint
from recurrentgemma.jax import array_typing as at


def save_parameters(checkpoint_path: str, params: at.Params):
  ckpt = {"params": params}
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  orbax_checkpointer.save(checkpoint_path, ckpt,)


def load_parameters(
    checkpoint_path: str,
    sharding: str | Mapping[str, Any],
) -> at.Params:
  """A helper function for loading parameters from a checkpoint.

  Args:
    checkpoint_path: The path to the orbax checkpoint to load.
    sharding: One of three options:
      1. "single_device" - Loads all parameters on a single device, determined
        by `jax.local_devices()[0]`.
      2. "replicated" - Loads all parameters replicated on all devices.
      3. A user specified PyTree of `jax.sharding.Sharding`, which specifies
        the sharding of each individual parameter.

  Returns:
    The loaded parameters, sharded as specified by `sharding`.
  """

  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  structure = checkpointer.metadata(checkpoint_path)

  if isinstance(sharding, str):
    if sharding == "single_device":
      sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    elif sharding == "replicated":
      devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
      sharding = jax.sharding.PositionalSharding(devices).replicate(
          axis=0, keepdims=True
      )

    # Make a sharding tree for all parameters
    sharding_tree = jax.tree_util.tree_map(lambda x: sharding, structure)

  else:
    sharding_tree = sharding

  restore_args = jax.tree_util.tree_map(
      lambda x, s: orbax.checkpoint.ArrayRestoreArgs(
          restore_type=jax.Array,
          sharding=s,
      ),
      structure, sharding_tree,
  )

  # Load
  return checkpointer.restore(checkpoint_path, restore_args=restore_args)
