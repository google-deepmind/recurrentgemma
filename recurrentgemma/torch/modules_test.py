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
"""Tests for the recurrent block."""

from absl.testing import absltest
from absl.testing import parameterized

from recurrentgemma import common
from recurrentgemma.jax import griffin as jax_griffin
from recurrentgemma.jax import modules as jax_modules
from recurrentgemma.torch import griffin
from recurrentgemma.torch import modules
from recurrentgemma.torch import test_utils

import torch


class EmbedderTest(parameterized.TestCase):

  @parameterized.product(
      vocab_size=[128, 256],
      embed_dim=[512, 1024],
      scale_by_sqrt_dim=[False, True],
      dtype=["float32"],
      seed=[981273821],
  )
  def test_numerically_to_jax(
      self,
      vocab_size: int,
      embed_dim: int,
      scale_by_sqrt_dim: bool,
      dtype: str,
      seed: int,
  ):
    # Create a Griffin model without any blocks.
    config = common.GriffinConfig(
        vocab_size=vocab_size,
        width=embed_dim,
        mlp_expanded_width=0,
        num_heads=0,
        embeddings_scale_by_sqrt_dim=scale_by_sqrt_dim,
        block_types=tuple(),
        logits_soft_cap=None,
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
        input_shape=[1, 16],
        dtype="int32",
        vocab_size=vocab_size,
        seed=seed,
        num_unroll_steps=0,
    )


class LocalAttentionTest(parameterized.TestCase):

  @parameterized.product(
      width=[128, 1024],
      num_heads=[1, 8],
      window_size=[8, 16],
      seq_len=[64],
      dtype=["float32"],
      seed=[12413166],
  )
  def test_numerically_to_jax(
      self,
      width: int,
      num_heads: int,
      window_size: int,
      seq_len: int,
      dtype: str,
      seed: int,
      num_unroll_steps: int = 2,
  ):
    test_utils.numerically_compare_modules(
        jax_module=jax_modules.LocalAttentionBlock(
            width=width,
            num_heads=num_heads,
            window_size=window_size,
            param_dtype=dtype,
        ),
        torch_module=modules.LocalAttentionBlock(
            width=width,
            num_heads=num_heads,
            window_size=window_size,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=True,
        has_cache=True,
        input_shape=[1, seq_len, width],
        dtype=dtype,
        seed=seed,
        num_unroll_steps=num_unroll_steps,
    )


class RecurrentBlockTest(parameterized.TestCase):

  @parameterized.product(
      width=[128, 1024],
      num_heads=[1, 8],
      lru_width=[128, 256],
      seq_len=[64],
      dtype=["float32"],
      seed=[12413166],
  )
  def test_numerically_to_jax(
      self,
      width: int,
      num_heads: int,
      lru_width: int,
      seq_len: int,
      dtype: str,
      seed: int,
      num_unroll_steps: int = 2,
  ):
    test_utils.numerically_compare_modules(
        jax_module=jax_modules.RecurrentBlock(
            width=width,
            num_heads=num_heads,
            lru_width=lru_width,
            scan_type=common.ScanType.LINEAR_NATIVE,
            param_dtype=dtype,
        ),
        torch_module=modules.RecurrentBlock(
            width=width,
            num_heads=num_heads,
            lru_width=lru_width,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=True,
        has_cache=True,
        input_shape=[1, seq_len, width],
        dtype=dtype,
        seed=seed,
        num_unroll_steps=num_unroll_steps,
    )


class MLPBlockTest(parameterized.TestCase):

  @parameterized.product(
      width=[128, 1024],
      expanded_width=[256, 2048],
      dtype=["float32"],
      seed=[12413166],
  )
  def test_numerically_to_jax(
      self,
      width: int,
      expanded_width: int,
      dtype: str,
      seed: int,
      num_unroll_steps: int = 2,
  ):
    test_utils.numerically_compare_modules(
        jax_module=jax_modules.MLPBlock(
            width=width,
            expanded_width=expanded_width,
            param_dtype=dtype,
        ),
        torch_module=modules.MLPBlock(
            width=width,
            expanded_width=expanded_width,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=False,
        has_cache=False,
        input_shape=[1, 2, width],
        dtype=dtype,
        seed=seed,
        num_unroll_steps=num_unroll_steps,
    )


class ResidualBlockTest(parameterized.TestCase):

  @parameterized.product(
      width=[128],
      mlp_expanded_width=[512],
      lru_width=[256],
      num_heads=[8],
      attention_window_size=[32],
      temporal_block_type=[
          common.TemporalBlockType.RECURRENT,
          common.TemporalBlockType.ATTENTION,
      ],
      seq_len=[128],
      dtype=["float32"],
      seed=[12413166],
  )
  def test_numerically_to_jax(
      self,
      width: int,
      mlp_expanded_width: int,
      lru_width: int,
      num_heads: int,
      attention_window_size: int,
      temporal_block_type: common.TemporalBlockType,
      seq_len: int,
      dtype: str,
      seed: int,
      num_unroll_steps: int = 2,
  ):
    test_utils.numerically_compare_modules(
        jax_module=jax_modules.ResidualBlock(
            width=width,
            mlp_expanded_width=mlp_expanded_width,
            lru_width=lru_width,
            num_heads=num_heads,
            attention_window_size=attention_window_size,
            temporal_block_type=temporal_block_type,
            scan_type=common.ScanType.LINEAR_NATIVE,
            param_dtype=dtype,
        ),
        torch_module=modules.ResidualBlock(
            width=width,
            mlp_expanded_width=mlp_expanded_width,
            lru_width=lru_width,
            num_heads=num_heads,
            attention_window_size=attention_window_size,
            temporal_block_type=temporal_block_type,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=True,
        has_cache=True,
        input_shape=[1, seq_len, width],
        dtype=dtype,
        seed=seed,
        num_unroll_steps=num_unroll_steps,
    )


if __name__ == "__main__":
  absltest.main()
