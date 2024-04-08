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
"""Tests for full Griffin model."""

from absl.testing import absltest
from absl.testing import parameterized

from recurrentgemma import common
from recurrentgemma.jax import griffin as jax_griffin
from recurrentgemma.torch import griffin
from recurrentgemma.torch import test_utils

import torch


class GriffinTest(parameterized.TestCase):

  @parameterized.product(
      vocab_size=[128],
      width=[256],
      mlp_expanded_width=[512],
      num_heads=[8],
      scale_by_sqrt_dim=[True],
      attention_window_size=[16],
      seq_len=[64],
      dtype=["float32"],
      seed=[93282131],
  )
  def test_numerically_to_jax(
      self,
      vocab_size: int,
      width: int,
      mlp_expanded_width: int,
      num_heads: int,
      attention_window_size: int,
      scale_by_sqrt_dim: bool,
      seq_len: int,
      dtype: str,
      seed: int,
      num_unroll_steps: int = 2,
  ):
    config = common.GriffinConfig(
        vocab_size=vocab_size,
        width=width,
        mlp_expanded_width=mlp_expanded_width,
        num_heads=num_heads,
        attention_window_size=attention_window_size,
        scan_type=common.ScanType.LINEAR_NATIVE,
        embeddings_scale_by_sqrt_dim=scale_by_sqrt_dim,
        block_types=(
            common.TemporalBlockType.RECURRENT,
            common.TemporalBlockType.ATTENTION,
        ),
    )
    test_utils.numerically_compare_modules(
        jax_module=jax_griffin.Griffin(
            config=config,
            param_dtype=dtype,
        ),
        torch_module=griffin.Griffin(
            config=config,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=True,
        has_cache=True,
        input_shape=[1, seq_len],
        dtype="int32",
        seed=seed,
        num_unroll_steps=num_unroll_steps,
        vocab_size=vocab_size,
    )


if __name__ == "__main__":
  absltest.main()
