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

from absl.testing import absltest
from absl.testing import parameterized

from recurrentgemma import common
from recurrentgemma.jax import layers as jax_layers
from recurrentgemma.torch import layers
from recurrentgemma.torch import test_utils

import torch


class RMSNormTest(parameterized.TestCase):

  @parameterized.product(
      width=[128, 1024],
      eps=[1e-6, 1e-3],
      dtype=["float32", "bfloat16"],
      seed=[9817321],
  )
  def test_numerically_to_jax(
      self,
      width: int,
      eps: float,
      dtype: str,
      seed: int,
  ):
    test_utils.numerically_compare_modules(
        jax_module=jax_layers.RMSNorm(
            width=width,
            eps=eps,
            param_dtype=dtype,
        ),
        torch_module=layers.RMSNorm(
            width=width,
            eps=eps,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=False,
        has_cache=False,
        input_shape=[1, 2, width],
        dtype=dtype,
        seed=seed,
    )


class BlockDiagonalLinearTest(parameterized.TestCase):

  @parameterized.product(
      width=[128, 1024],
      num_blocks=[1, 16],
      dtype=["float32", "bfloat16"],
      seed=[9018323],
  )
  def test_numerically_to_jax(
      self,
      width: int,
      num_blocks: int,
      dtype: str,
      seed: int,
  ):
    test_utils.numerically_compare_modules(
        jax_module=jax_layers.BlockDiagonalLinear(
            width=width,
            num_blocks=num_blocks,
            param_dtype=dtype,
        ),
        torch_module=layers.BlockDiagonalLinear(
            width=width,
            num_blocks=num_blocks,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=False,
        has_cache=False,
        input_shape=[1, 2, width],
        dtype=dtype,
        seed=seed,
    )


class RGLRUTest(parameterized.TestCase):

  @parameterized.product(
      width=[128, 1024],
      num_heads=[1, 16],
      seq_len=[32, 128],
      dtype=["float32"],
      seed=[9018323],
  )
  def test_numerically_to_jax(
      self,
      width: int,
      num_heads: int,
      seq_len: int,
      dtype: str,
      seed: int,
      num_unroll_steps: int = 2,
  ):
    test_utils.numerically_compare_modules(
        jax_module=jax_layers.RGLRU(
            width=width,
            num_heads=num_heads,
            scan_type=common.ScanType.LINEAR_NATIVE,
            param_dtype=dtype,
        ),
        torch_module=layers.RGLRU(
            width=width,
            num_heads=num_heads,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=True,
        has_cache=True,
        input_shape=[1, seq_len, width],
        dtype=dtype,
        seed=seed,
        tols=dict(rtol=1e-2, atol=3e-2) if dtype == "bfloat16" else None,
        num_unroll_steps=num_unroll_steps,
    )


class Conv1DTest(parameterized.TestCase):

  @parameterized.product(
      width=[128, 1024],
      temporal_width=[4, 8],
      seq_len=[32],
      dtype=["float32", "bfloat16"],
      seed=[9018323],
  )
  def test_numerically_to_jax(
      self,
      width: int,
      temporal_width: int,
      seq_len: int,
      dtype: str,
      seed: int,
      num_unroll_steps: int = 2,
  ):
    test_utils.numerically_compare_modules(
        jax_module=jax_layers.Conv1D(
            width=width,
            temporal_width=temporal_width,
            param_dtype=dtype,
        ),
        torch_module=layers.Conv1D(
            width=width,
            temporal_width=temporal_width,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=True,
        has_cache=True,
        input_shape=[1, seq_len, width],
        dtype=dtype,
        seed=seed,
        tols=dict(rtol=1e-2, atol=3e-2) if dtype == "bfloat16" else None,
        num_unroll_steps=num_unroll_steps,
    )


class EinsumTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          inputs_shape=(1, 16),
          w_shape=(3, 2, 16, 3),
          b_shape=(3, 1, 1, 3),
          eqn="td,sndh->stnh",
          dtype="float32",
          seed=21987321,
      ),
      dict(
          inputs_shape=(1, 16),
          w_shape=(3, 2, 16, 3),
          b_shape=(3, 1, 1, 3),
          eqn="td,sndh->stnh",
          dtype="bfloat16",
          seed=21987321,
      ),
      dict(
          inputs_shape=(1, 16, 128),
          w_shape=(2, 128, 256),
          b_shape=(2, 1, 1, 256),
          eqn="...td,cdD->c...tD",
          dtype="float32",
          seed=1239084,
      ),
      dict(
          inputs_shape=(1, 16, 128),
          w_shape=(2, 128, 256),
          b_shape=(2, 1, 1, 256),
          eqn="...td,cdD->c...tD",
          dtype="bfloat16",
          seed=1239084,
      ),
  )
  def test_numerically_to_jax(
      self,
      inputs_shape: tuple[int, ...],
      w_shape: tuple[int, ...],
      b_shape: tuple[int, ...],
      eqn: str,
      dtype: str,
      seed: int,
  ):
    test_utils.numerically_compare_modules(
        jax_module=jax_layers.Einsum(
            w_shape=w_shape,
            b_shape=b_shape,
            eqn=eqn,
            param_dtype=dtype,
        ),
        torch_module=layers.Einsum(
            w_shape=w_shape,
            b_shape=b_shape,
            eqn=eqn,
            dtype=getattr(torch, dtype),
        ),
        uses_segment_pos=False,
        has_cache=False,
        input_shape=inputs_shape,
        dtype=dtype,
        seed=seed,
    )


if __name__ == "__main__":
  absltest.main()
