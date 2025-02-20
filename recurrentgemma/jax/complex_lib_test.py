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
"""Tests for the complex_lib module."""

import operator
import types
from typing import Any, Final, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import einops
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from recurrentgemma.jax import complex_lib


_BINARY_OPS: Final[list[str]] = ['add', 'sub', 'mul', 'truediv']
_BINARY_OPS_REVERSED: Final[list[str]] = ['add', 'sub', 'mul']

_REAL_PART = np.array(
    [[0.36971679, 0.07883132], [0.60356846, 0.48390578]],
)
_IMAG_PART = np.array(
    [[0.12918229, 0.10327365], [0.36421535, 0.67493899]],
)

_TEST_ARRAY: Final[np.ndarray] = _REAL_PART + 1j * _IMAG_PART


class TestForwardCase(NamedTuple):
  module: types.ModuleType
  op_name: str
  args: list[Any]
  kwargs: dict[str, Any]


_FORWARD_OPS_UNDER_TEST: Final[list[TestForwardCase]] = [
    TestForwardCase(
        module=jnp,
        op_name='broadcast_to',
        args=[_TEST_ARRAY],
        kwargs=dict(shape=(3, *_TEST_ARRAY.shape)),
    ),
    TestForwardCase(
        module=jnp,
        op_name='concatenate',
        args=[[_TEST_ARRAY, _TEST_ARRAY + 4]],
        kwargs={},
    ),
    TestForwardCase(
        module=jnp,
        op_name='reshape',
        args=[_TEST_ARRAY],
        kwargs=dict(
            shape=(
                _TEST_ARRAY.shape[0] // 2,
                -1,
            ),  # ATTENTION: the array 1st dim should be divisible by 2
        ),
    ),
    TestForwardCase(
        module=jnp,
        op_name='stack',
        args=[[_TEST_ARRAY, _TEST_ARRAY - 2]],
        kwargs={},
    ),
    TestForwardCase(
        module=jnp,
        op_name='flip',
        args=[_TEST_ARRAY],
        kwargs=dict(axis=1),
    ),
    TestForwardCase(
        module=jnp,
        op_name='tile',
        args=[_TEST_ARRAY],
        kwargs=dict(reps=3),
    ),
    TestForwardCase(
        module=lax,
        op_name='slice_in_dim',
        args=[_TEST_ARRAY],
        kwargs=dict(
            stride=1,
            axis=1,
            start_index=0,
            limit_index=None,
        ),
    ),
    TestForwardCase(
        module=lax,
        op_name='pad',
        args=[_TEST_ARRAY, np.complex64(0.0)],
        kwargs=dict(
            padding_config=[(1, 1, 0), (1, 1, 0)],
        ),
    ),
    TestForwardCase(
        module=lax,
        op_name='add',
        args=[_TEST_ARRAY, _TEST_ARRAY - 4],
        kwargs=dict(),
    ),
    TestForwardCase(
        module=einops,
        op_name='rearrange',
        args=[_TEST_ARRAY],
        kwargs=dict(
            pattern='b (l d) -> b l d',
            l=2,  #  ATTENITON! The last dim should be divisible by 2
        ),
    ),
    TestForwardCase(
        module=einops,
        op_name='repeat',
        args=[_TEST_ARRAY],
        kwargs=dict(
            pattern='b d -> r b d',
            r=2,
        ),
    ),
    TestForwardCase(
        module=jnp,
        op_name='zeros_like',
        args=[_TEST_ARRAY],
        kwargs=dict(),
    ),
    TestForwardCase(
        module=jnp,
        op_name='ones_like',
        args=[_TEST_ARRAY],
        kwargs=dict(),
    ),
    TestForwardCase(
        module=jnp,
        op_name='exp',
        args=[_TEST_ARRAY],
        kwargs=dict(),
    ),
    TestForwardCase(
        module=jnp,
        op_name='log',
        args=[_TEST_ARRAY],
        kwargs=dict(),
    ),
    TestForwardCase(
        module=jnp,
        op_name='conjugate',
        args=[_TEST_ARRAY],
        kwargs=dict(),
    ),
]


def _get_testing_arrays(
    count: int,
    shape: tuple[int, ...] = (8, 8),
) -> list[jax.Array]:
  rngs = jax.random.split(jax.random.PRNGKey(42), count)
  return [jax.random.normal(rng, shape=shape) for rng in rngs]


class ComplexTest(parameterized.TestCase):

  @parameterized.parameters(_BINARY_OPS)
  def test_complex_wrapper_with_complex_wrapper_ops(self, op_name: str) -> None:
    op = getattr(operator, op_name)
    x, y, a, b = _get_testing_arrays(count=4)
    custom_complex_one = complex_lib.Complex(x, y)
    custom_complex_two = complex_lib.Complex(a, b)
    xy_complex = x + 1j * y
    ab_complex = a + 1j * b

    res = op(custom_complex_one, custom_complex_two)
    res_expected = op(xy_complex, ab_complex)

    assert jnp.allclose(res.real, res_expected.real)
    assert jnp.allclose(res.imag, res_expected.imag)

  @parameterized.parameters(_BINARY_OPS)
  def test_complex_wrapper_with_jax_array_ops(self, op_name: str) -> None:
    op = getattr(operator, op_name)
    x, y, a = _get_testing_arrays(count=3)

    xy_custom_complex = complex_lib.Complex(x, y)
    xy_complex = x + 1j * y

    res_expected = op(xy_complex, a)
    res = op(xy_custom_complex, a)

    assert jnp.allclose(res.real, res_expected.real, atol=1e-5)
    assert jnp.allclose(res.imag, res_expected.imag, atol=1e-5)

  @parameterized.parameters(_BINARY_OPS_REVERSED)
  def test_complex_wrapper_with_jax_array_ops_commutes(self, op_name: str):
    """Tests to make sure that the reverse operations are implemented."""
    op = getattr(operator, op_name)
    x, y, a = _get_testing_arrays(count=3)

    xy_custom_complex = complex_lib.Complex(x, y)
    xy_complex = x + 1j * y

    res_expected = op(a, xy_complex)
    res = op(a, xy_custom_complex)

    assert jnp.allclose(res.real, res_expected.real, atol=1e-5)
    assert jnp.allclose(res.imag, res_expected.imag, atol=1e-5)

  def test_change_dtype(self) -> None:
    rng = jax.random.PRNGKey(42)
    rng_x, rng_y = jax.random.split(rng, 2)

    x = jax.random.normal(rng_x, (16, 16), dtype=jnp.float32)
    y = jax.random.normal(rng_y, (16, 16), dtype=jnp.float32)

    xy_complex = complex_lib.Complex(
        real=x.astype(jnp.bfloat16),
        imag=y.astype(jnp.bfloat16),
    )
    assert xy_complex.dtype == jnp.bfloat16

  @parameterized.parameters(_BINARY_OPS)
  def test_complex_wrapper_ops_with_jax_complex_exception(
      self,
      op_name: str,
  ) -> None:
    op = getattr(operator, op_name)
    x, y = _get_testing_arrays(count=2)

    xy_custom_complex = complex_lib.Complex(x, y)
    xy = x + 1j * y
    with self.assertRaises(ValueError):
      op(xy_custom_complex, xy)

  @parameterized.parameters(_BINARY_OPS)
  def test_complex_wrapper_ops_with_different_dtype_exception(
      self,
      op_name: str,
  ) -> None:
    op = getattr(operator, op_name)
    x, y, a, b = _get_testing_arrays(count=4)

    xy_custom_complex = complex_lib.Complex(
        x.astype(jnp.bfloat16),
        y.astype(jnp.bfloat16),
    )
    ab_custom_complex = complex_lib.Complex(
        a.astype(jnp.float32),
        b.astype(jnp.float32),
    )

    with self.assertRaises(ValueError):
      op(xy_custom_complex, ab_custom_complex)

  @parameterized.parameters(_FORWARD_OPS_UNDER_TEST)
  def test_treat_methods(
      self,
      module: types.ModuleType,
      op_name: str,
      args: list[Any],
      kwargs: dict[str, Any],
  ) -> None:
    complex_lib_op = getattr(complex_lib, op_name)
    op_ref = getattr(module, op_name)

    def convert_to_complex_wrapper(x: Any) -> Any:
      if isinstance(x, jnp.ndarray) and jnp.iscomplexobj(x):
        return complex_lib.Complex(x.real, x.imag)
      return x

    complex_lib_args = jax.tree_util.tree_map(convert_to_complex_wrapper, args)
    res = complex_lib_op(*complex_lib_args, **kwargs)
    res_expected = op_ref(*args, **kwargs)
    assert jnp.allclose(res.real, res_expected.real)
    assert jnp.allclose(res.imag, res_expected.imag)

  def test_einsum(self) -> None:
    op_name = 'einsum'
    complex_lib_op = getattr(complex_lib, op_name)
    op_ref = getattr(jnp, op_name)

    def to_complex_wrapper(x: Any) -> Any:
      if isinstance(x, jnp.ndarray) and jnp.iscomplexobj(x):
        return complex_lib.Complex(x.real, x.imag)
      return x

    arg_str = 'ab,bc->ac'
    args = [_TEST_ARRAY, _TEST_ARRAY + 1.0 + 1j]
    complex_lib_all_args = [
        [to_complex_wrapper(args[0]), args[1]],
        [args[0], to_complex_wrapper(args[1])],
        [to_complex_wrapper(args[0]), to_complex_wrapper(args[1])],
    ]

    res_expected = op_ref(arg_str, *args)

    for comlex_lib_args in complex_lib_all_args:
      res = complex_lib_op(arg_str, *comlex_lib_args)
      assert jnp.allclose(res.real, res_expected.real)
      assert jnp.allclose(res.imag, res_expected.imag)


if __name__ == '__main__':
  absltest.main()
