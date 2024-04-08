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
import jax
import jax.numpy as jnp
from recurrentgemma import common
from recurrentgemma import conversion
from recurrentgemma.jax import griffin as griffin_jax


BATCH_SIZE = 4
SEQ_LEN = 8
VOCAB_SIZE = 6
WIDTH = 16
MLP_EXPANDED_WIDTH = 6
NUM_HEADS = 2
LRU_WIDTH = 8
NUM_SAMPLING_STEPS = 3
BLOCK_TYPES = (
    common.TemporalBlockType.RECURRENT,
    common.TemporalBlockType.ATTENTION,
)


class ConversionTest(absltest.TestCase):

  def test_conversion_back_and_forth(self, seed: int = 1287312):
    # given a model
    config = common.GriffinConfig(
        vocab_size=VOCAB_SIZE,
        width=WIDTH,
        mlp_expanded_width=MLP_EXPANDED_WIDTH,
        num_heads=NUM_HEADS,
        lru_width=LRU_WIDTH,
        block_types=BLOCK_TYPES,
    )
    model = griffin_jax.Griffin(config)

    rng = jax.random.PRNGKey(seed=seed)
    tokens = jax.random.randint(
        rng,
        shape=[BATCH_SIZE, SEQ_LEN],
        minval=0,
        maxval=VOCAB_SIZE,
    )
    segment_pos = jnp.repeat(jnp.arange(SEQ_LEN)[None], BATCH_SIZE, axis=0)

    # where
    jax_params = model.init(rng, tokens, segment_pos=segment_pos)
    torch_params = conversion.flax_params_to_pytorch_state_dict(jax_params)
    back_params = conversion.pytorch_state_dict_to_flax_params(torch_params)

    # then
    self.assertEqual(
        jax.tree_util.tree_structure(jax_params),
        jax.tree_util.tree_structure(back_params)
    )
    for leaf_jax, leaf_back in zip(
        jax.tree_util.tree_leaves(jax_params),
        jax.tree_util.tree_leaves(back_params)
    ):
      self.assertEqual(leaf_jax.shape, leaf_back.shape)
      self.assertEqual(leaf_jax.dtype, leaf_back.dtype)
      self.assertEqual(jnp.linalg.norm(leaf_jax - leaf_back), 0.0)


if __name__ == '__main__':
  absltest.main()
