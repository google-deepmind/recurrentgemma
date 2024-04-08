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
"""Tests for Pallas scan kernel."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from recurrentgemma.jax import pallas


def _lru_reference(
    x: jax.Array,
    a: jax.Array,
    h0: jax.Array | None,
) -> jax.Array:

  @jax.vmap
  def batched_scan(x_, h0_, a_):
    def body_fn(h_prev, current_inputs):
      acc_dtype = h_prev.dtype
      x_t, a_t = current_inputs
      h_next = a_t.astype(acc_dtype) * h_prev + x_t.astype(acc_dtype)
      return h_next, h_next.astype(x_t.dtype)

    _, hi = jax.lax.scan(
        body_fn,
        init=h0_,
        xs=(x_, a_),
    )
    return hi

  if h0 is None:
    h0 = jnp.zeros_like(x[:, 0], dtype=jnp.float32)

  return batched_scan(x, h0, a)


def _convert_outputs_to_float32(func):
  def wrapped(*args, **kwargs):
    outputs = func(*args, **kwargs)
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), outputs)

  return wrapped


@jax.jit
@_convert_outputs_to_float32
def _lru_reference_vjp(
    x: jax.Array,
    a: jax.Array,
    h0: jax.Array | None,
    dh: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  v, vjp = jax.vjp(_lru_reference, x, a, h0)
  return v, vjp(dh)


@jax.jit
@_convert_outputs_to_float32
def _lru_vjp(
    x: jax.Array,
    a: jax.Array,
    h0: jax.Array | None,
    dh: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  v, vjp = jax.vjp(pallas.lru, x, a, h0)
  return v, vjp(dh)


def _generate_inputs(
    shape: tuple[int, int, int],
    have_h0: bool,
    dtype: str,
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array]:
  rng = jax.random.PRNGKey(seed)
  a_key, x_key, h_key, dh_key = jax.random.split(rng, 4)
  x = jax.random.uniform(x_key, shape).astype(dtype)
  log_a = jax.random.uniform(a_key, shape).astype(dtype)
  dy = jax.random.uniform(dh_key, shape).astype(dtype)

  if have_h0:
    h0 = jax.random.uniform(h_key, (shape[0], shape[2]), dtype=jnp.float32)
  else:
    h0 = None

  return x, jnp.exp(-log_a), h0, dy


@absltest.skipThisClass('Pallas requires TPU. Run manually if TPU available.')
class PallasScanTest(parameterized.TestCase):
  """Tests for nested rematerialised scan."""

  @parameterized.product(
      shape=[
          (1, 512, 128),
          (2, 512, 1024 * 4),
      ],
      have_h0=[True, False],
      dtype=['float32', 'bfloat16'],
      seed=[981732821, 123921876],
  )
  def test_lru(
      self,
      shape: tuple[int, int, int],
      have_h0: bool,
      dtype: str,
      seed: int,
  ):
    if dtype == 'bfloat16':
      tols = dict(atol=1e-2, rtol=1e-2)
    elif dtype == 'float32':
      tols = dict(atol=1e-9, rtol=1e-9)
    else:
      raise NotImplementedError(f'Unsupported dtype: {dtype}.')

    x, a, h0, dh = _generate_inputs(shape, have_h0, dtype, seed)

    y_ref, df_ref = _lru_reference_vjp(x, a, h0, dh)
    y, df = _lru_vjp(x, a, h0, dh)

    self.assertEqual(len(df), len(df_ref))

    np.testing.assert_allclose(y, y_ref, **tols)

    for df_i, df_ref_i in zip(df[:-1], df_ref[:-1]):
      np.testing.assert_allclose(df_i, df_ref_i, **tols)

    if have_h0:
      np.testing.assert_allclose(df[-1], df_ref[-1], **tols)
    else:
      assert df_ref[-1] is None and df[-1] is None


if __name__ == '__main__':
  absltest.main()
