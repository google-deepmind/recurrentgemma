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
"""Tests for Griffin layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from recurrentgemma import common
from recurrentgemma.jax import layers


class LayersTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          inputs_shape=(1, 4),
          w_shape=(3, 2, 4, 3),
          b_shape=(3, 1, 1, 3),
          eqn='TD,SNDH->STNH',
          expected_shape=(3, 1, 2, 3),
      ),
      dict(
          inputs_shape=(1, 2, 4),
          w_shape=(2, 4, 8),
          b_shape=(1, 8),
          eqn='ANH,NHD->AD',
          expected_shape=(1, 8),
      ),
  )
  def test_einsum(
      self,
      inputs_shape: tuple[int, int],
      w_shape: tuple[int, int, int],
      b_shape: tuple[int, int],
      eqn: str,
      expected_shape: tuple[int, int],
  ):
    einsum = layers.Einsum(
        w_shape=w_shape,
        b_shape=b_shape,
        eqn=eqn,
    )
    output = einsum.apply(
        {'params': {'w': jnp.ones(w_shape), 'b': jnp.ones(b_shape)}},
        jnp.ones(inputs_shape),
    )
    self.assertEqual(output.shape, expected_shape)

  @parameterized.parameters(dict(x=[0.1, 0.2], expected=[0.6324429, 1.2648858]))
  def test_rmsnorm(self, x: float, expected: float):
    x = jnp.array([x])
    rmsnorm = layers.RMSNorm(width=2)
    params = rmsnorm.init(jax.random.PRNGKey(0), x)
    output = rmsnorm.apply(params, x)
    np.testing.assert_array_equal(output, jnp.array([expected]))

  @parameterized.product(
      seq_len=[1, 4, 8],
      dtype=['bfloat16', 'float32'],
      scan_type=[
          common.ScanType.LINEAR_NATIVE,
          common.ScanType.ASSOCIATIVE_NATIVE,
      ],
  )
  def test_scan(
      self,
      seq_len: int,
      dtype: str,
      scan_type: common.ScanType,
  ):
    # Given.
    key = jax.random.PRNGKey(0)
    x_key, a_key, h_key = jax.random.split(key, 3)
    b, d = 2, 8

    x = jax.random.normal(x_key, shape=(b, seq_len, d), dtype=dtype)
    a = jax.random.normal(a_key, shape=(b, seq_len, d), dtype=dtype)
    h0 = jax.random.normal(h_key, shape=(b, d), dtype=jnp.float32)

    reset = jnp.zeros((b, seq_len), dtype=jnp.bool_)

    # when
    y, h_next = layers.rnn_scan(
        x,
        a,
        reset,
        h0,
        scan_type=scan_type,
    )

    # then
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(y.dtype, x.dtype)

    self.assertEqual(h_next.shape, h0.shape)
    self.assertEqual(h_next.dtype, h0.dtype)


if __name__ == '__main__':
  absltest.main()
