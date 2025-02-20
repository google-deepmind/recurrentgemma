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
"""Short example how to run a model."""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import orbax.checkpoint
from recurrentgemma import jax as recurrentgemma


_DEBUG_MODE = flags.DEFINE_bool("debug", False, "Debug mode.")

_SAVE_TENSORS = flags.DEFINE_string("save_path", None, "Path to save tensors.")

_KEY = flags.DEFINE_integer("key", 1241312, "Key to use for randomization.")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _DEBUG_MODE.value:
    logging.info("Running debug mode.")
    config = recurrentgemma.GriffinConfig(
        vocab_size=100,
        width=128,
        mlp_expanded_width=3 * 128,
        lru_width=256,
        num_heads=2,
        block_types=(
            recurrentgemma.TemporalBlockType.RECURRENT,
            recurrentgemma.TemporalBlockType.ATTENTION,
        ),
        embeddings_scale_by_sqrt_dim=True,
        attention_window_size=2048,
        logits_soft_cap=30.0,
    )
  else:
    logging.info("Running RecurrentGemma 2B.")
    config = recurrentgemma.GriffinConfig.from_preset(
        vocab_size=256_000,
        preset=recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1,
    )

  model = recurrentgemma.Griffin(config)

  batch_size = 4
  sequence_length = 8 * 1024
  rng = jax.random.PRNGKey(_KEY.value)
  rng, key = jax.random.split(rng)
  tokens = jax.random.randint(
      key,
      shape=[batch_size, sequence_length],
      minval=0,
      maxval=config.vocab_size,
  )
  pos = jnp.repeat(jnp.arange(sequence_length)[None], batch_size, axis=0)

  # Forward pass/prompt processing
  logging.info("Initialise model and prefill.")
  params = model.init(rng, tokens, segment_pos=pos)
  apply_func = jax.jit(model.apply)
  logits, cache = apply_func(params, tokens, segment_pos=pos)
  prefill_logits = logits

  pos = pos[:, -1:] + 1
  # Sampling tokens one by one
  logging.info("Decoding.")
  sampled_tokens = []
  for i in range(128):
    rng, key = jax.random.split(rng)
    next_token = jax.random.categorical(key, logits[:, -1])
    sampled_tokens.append(next_token)
    logits, cache = apply_func(
        params,
        next_token[:, None],
        segment_pos=pos + i,
        cache=cache,
    )

  sampled_tokens = jnp.stack(sampled_tokens, axis=1)
  logging.info("Sampled tokens.")
  logging.debug(sampled_tokens)

  if _SAVE_TENSORS.value:
    # Bundle everything together.
    ckpt = {
        "model": params,
        "config": config,
        "inputs_tokens": tokens,
        "inputs_pos": pos,
        "sampled_tokens": sampled_tokens,
        "prefill_logits": prefill_logits,
    }
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(_SAVE_TENSORS.value, ckpt)


if __name__ == "__main__":
  app.run(main)
