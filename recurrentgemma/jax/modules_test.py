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
"""Tests for the module components used in Griffin and Hawk."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from recurrentgemma import common
from recurrentgemma.jax import modules


class LocalAttentionTest(parameterized.TestCase):

  @parameterized.parameters([1, 8])
  def test_local_attention_output_shapes(
      self,
      seq_len: int,
      seed: int = 12319843,
  ):
    # Given.
    key = jax.random.PRNGKey(seed)
    batch_size, width = 1, 8
    num_heads = 2
    window_size = 16
    head_dim = width // num_heads

    x = jax.random.normal(key, shape=(batch_size, seq_len, width))
    segment_pos = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))

    block = modules.LocalAttentionBlock(
        width=width,
        num_heads=num_heads,
        window_size=window_size,
    )

    if seq_len == 1:
      # Sampling mode.
      cache = modules.LocalAttentionBlock.init_cache(
          batch_size=batch_size,
          window_size=window_size,
          heads_dim=head_dim,
          dtype=jnp.float32,
      )

    else:
      # Forward pass mode.
      cache = None

    # When.
    (out, cache), _ = block.init_with_output(key, x, segment_pos, cache)

    # Then.
    self.assertEqual(out.shape, (batch_size, seq_len, width))
    self.assertEqual(cache.keys.shape, (batch_size, window_size, 1, head_dim))
    self.assertEqual(cache.values.shape, (batch_size, window_size, 1, head_dim))

  def test_local_attention_updates_cache_correctly(self, seed: int = 874321):
    key = jax.random.PRNGKey(seed)
    batch_size, seq_len, width = 1, 1, 4
    num_heads = 1
    window_size = 8
    segment_pos = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))

    x = jax.random.normal(key, shape=(batch_size, seq_len, width))
    block = modules.LocalAttentionBlock(
        width=width,
        num_heads=num_heads,
        window_size=window_size,
    )
    cache = modules.LocalAttentionBlock.init_cache(
        batch_size=batch_size,
        window_size=window_size,
        heads_dim=width // num_heads,
        dtype=jnp.float32,
    )

    # Produce a new cache.
    (_, new_cache), _ = block.init_with_output(key, x, segment_pos, cache)

    # Check except the last index, the cache is zero
    np.testing.assert_array_almost_equal(new_cache.keys[:, :-1], 0.0)
    np.testing.assert_array_almost_equal(new_cache.values[:, :-1], 0.0)
    np.testing.assert_array_almost_equal(new_cache.num_tokens, 1)

    # And check that the last index is not zero
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_almost_equal,
        new_cache.keys[:, -1],
        0.0,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_almost_equal,
        new_cache.values[:, -1],
        0.0,
    )


class RecurrentBlockTest(parameterized.TestCase):

  @parameterized.parameters([1, 8])
  def test_recurrent_block_output_shapes(
      self,
      seq_len: int,
      seed: int = 1208743,
  ):
    # given
    key = jax.random.PRNGKey(seed)
    batch_size, width = 32, 4
    lru_width = 10
    num_heads = 2

    x = jax.random.normal(key, shape=(batch_size, seq_len, width))
    segment_pos = jnp.tile(jnp.arange(seq_len), (batch_size, 1))

    block = modules.RecurrentBlock(
        width=width,
        lru_width=lru_width,
        num_heads=num_heads,
        conv1d_temporal_width=4,
    )

    if seq_len == 1:
      # Sampling mode.
      cache = modules.ResidualBlock.init_cache(
          batch_size=batch_size,
          width=width,
          num_heads=num_heads,
          attention_window_size=2048,
          temporal_block_type=common.TemporalBlockType.RECURRENT,
          lru_width=lru_width,
          dtype=jnp.float32,
      )
    else:
      # Forward pass mode.
      cache = None

    params = block.init(key, x, segment_pos, cache)

    # when
    out, _ = block.apply(params, x, segment_pos, cache)

    # then
    self.assertEqual(out.shape, (batch_size, seq_len, width))


if __name__ == "__main__":
  absltest.main()
