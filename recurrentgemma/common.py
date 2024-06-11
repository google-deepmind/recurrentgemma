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
"""Utilities for both the Jax and Pytorch modules."""

import enum
import itertools
from typing import Any, NamedTuple, Mapping


@enum.unique
class TemporalBlockType(enum.Enum):
  """Type of temporal mixing to use in a residual block."""

  ATTENTION = enum.auto()
  RECURRENT = enum.auto()


@enum.unique
class ScanType(enum.Enum):
  """Which Jax implementation to use for the scan in the RG-LRU in Jax.

  On TPUs Pallas is faster, hence when using `AUTO` the code will pick Pallas
  automatically if you are running on a TPU device and otherwise will fallback
  to the NATIVE Jax for loop.
  """

  AUTO = enum.auto()
  LINEAR_NATIVE = enum.auto()
  ASSOCIATIVE_NATIVE = enum.auto()
  LINEAR_PALLAS = enum.auto()


@enum.unique
class Preset(enum.Enum):
  """All default preset variants."""

  GRIFFIN_PAPER_7B = enum.auto()
  HAWK_PAPER_7B = enum.auto()
  RECURRENT_GEMMA_2B_V1 = enum.auto()
  RECURRENT_GEMMA_9B_V1 = enum.auto()

  @property
  def config_dict(self) -> dict[str, Any]:
    griffin_pattern = itertools.cycle([
        TemporalBlockType.RECURRENT,
        TemporalBlockType.RECURRENT,
        TemporalBlockType.ATTENTION,
    ])

    match self:

      case Preset.GRIFFIN_PAPER_7B:
        return dict(
            width=4096,
            mlp_expanded_width=3 * 4096,
            num_heads=32,
            lru_width=5632,
            block_types=tuple(itertools.islice(griffin_pattern, 32)),
            embeddings_scale_by_sqrt_dim=False,
            attention_window_size=1024,
            logits_soft_cap=0.0,
            scan_type=ScanType.AUTO,
        )

      case Preset.HAWK_PAPER_7B:
        return dict(
            width=4096,
            mlp_expanded_width=3 * 4096,
            num_heads=32,
            lru_width=5632,
            block_types=(TemporalBlockType.RECURRENT,) * 32,
            embeddings_scale_by_sqrt_dim=False,
            attention_window_size=1024,
            logits_soft_cap=0.0,
            scan_type=ScanType.AUTO,
        )

      case Preset.RECURRENT_GEMMA_2B_V1:
        return dict(
            width=2560,
            mlp_expanded_width=3 * 2560,
            num_heads=10,
            lru_width=2560,
            block_types=tuple(itertools.islice(griffin_pattern, 26)),
            embeddings_scale_by_sqrt_dim=True,
            attention_window_size=2048,
            logits_soft_cap=30.0,
            scan_type=ScanType.AUTO,
        )

      case Preset.RECURRENT_GEMMA_9B_V1:
        return dict(
            width=4096,
            mlp_expanded_width=3 * 4096,
            num_heads=16,
            lru_width=4096,
            block_types=tuple(itertools.islice(griffin_pattern, 38)),
            embeddings_scale_by_sqrt_dim=True,
            attention_window_size=2048,
            logits_soft_cap=30.0,
            scan_type=ScanType.AUTO,
        )


class GriffinConfig(NamedTuple):
  """Griffin config - https://arxiv.org/abs/2402.19427.

  Attributes:
    vocab_size: The number of tokens in the vocabulary.
    width: The dimenonality of the model, e.g. the dimensonality of the
      embeddings and the output of each layer.
    mlp_expanded_width: The width of the hidden layer in the MLP block.
    num_heads: The number of heads for the attention block and the number of
      heads/blocks for the block-diagonal layers used in the RG-LRU gates. This
      number must divide `width` and `lru_width`.
    block_types: A sequence containing the type of the residual blocks in the
      architecture, specifying each block in order if it should use a recurrent
      or an attention sub-block for the temporal-mixing.
    embeddings_scale_by_sqrt_dim: Whether to scale the output of the embeddings
      by `sqrt(width)`.
    attention_window_size: The size of the attention window used in the
      attention block.
    logits_soft_cap: This will cap the values of the final logits to not exceed
      this cap in absolute value by applying a `tanh`. If `0` no capping is
      applied.
    lru_width: The width of the RG-LRU if different from `width`.
    scan_type: If running Flax, this specifies which implementation to use for
      the scan in the RG-LRU.
  """

  vocab_size: int
  width: int
  mlp_expanded_width: int
  num_heads: int
  block_types: tuple[TemporalBlockType, ...]
  embeddings_scale_by_sqrt_dim: bool
  attention_window_size: int
  logits_soft_cap: float
  lru_width: int | None = None
  scan_type: ScanType = ScanType.AUTO

  @property
  def max_cache_length(self) -> int:
    """The maximum length of the cache used for the model."""
    return self.attention_window_size

  @property
  def num_layers(self) -> int:
    """The number of layers of the model."""
    return len(self.block_types)

  @classmethod
  def from_preset(
      cls,
      preset: Preset,
      vocab_size: int = 256_000,
      max_sequence_length: int | None = None,
  ) -> "GriffinConfig":
    """Creates a `GriffinConfig` for a given preset."""
    cls_kwargs = preset.config_dict
    if max_sequence_length is not None:
      w = min(cls_kwargs["attention_window_size"], max_sequence_length)
      cls_kwargs["attention_window_size"] = w

    return cls(vocab_size=vocab_size, **cls_kwargs)

  @classmethod
  def _from_parameter_kwargs(
      cls,
      kwargs: dict[str, int | tuple[TemporalBlockType, ...]],
      preset: Preset | None = None,
      embeddings_scale_by_sqrt_dim: bool | None = None,
      attention_window_size: int | None = None,
      logits_soft_cap: float | None = None,
      scan_type: ScanType = ScanType.AUTO,
      max_sequence_length: int | None = None,
  ):
    """Creates a `GriffinConfig` from kwargs inferred from parameters."""
    if preset is not None:
      # Verify that the kwargs match the preset
      default_kwargs = preset.config_dict
      for key, value in kwargs.items():
        if key != "vocab_size" and value != default_kwargs[key]:
          raise ValueError(
              "The parameters provided does not seem to match the preset "
              f"{preset} provided, because the value for {key} is {value}, "
              "which is not equal to the preset value of "
              f"{default_kwargs[key]}."
          )

    else:
      default_kwargs = {}

    special_kwargs = dict(
        embeddings_scale_by_sqrt_dim=embeddings_scale_by_sqrt_dim,
        attention_window_size=attention_window_size,
        logits_soft_cap=logits_soft_cap,
        scan_type=scan_type,
    )

    cls_kwargs = dict(**kwargs)
    for key, value in special_kwargs.items():
      cls_kwargs[key] = value if value is not None else default_kwargs.get(key)

    if max_sequence_length is not None:
      w = min(cls_kwargs["attention_window_size"], max_sequence_length)
      cls_kwargs["attention_window_size"] = w

    return cls(**cls_kwargs)

  @classmethod
  def from_flax_params_or_variables(
      cls,
      flax_params_or_variables: Mapping[str, Any],
      preset: Preset | None = None,
      embeddings_scale_by_sqrt_dim: bool | None = None,
      attention_window_size: int | None = None,
      logits_soft_cap: float | None = None,
      scan_type: ScanType = ScanType.AUTO,
      max_sequence_length: int | None = None,
  ) -> "GriffinConfig":
    """Creates a `GriffinConfig` from Flax parameters.

    Args:
      flax_params_or_variables: The Flax parameters or variables (a dict
        containing a key 'params' corresponding to the actual parameters) to use
        to reconstruct the config.
      preset: The expected preset from which the parameters have been derived.
        If this is set values for hyper parameters that can't be inferred from
        the parameter, such as `embeddings_scale_by_sqrt_dim`,
        `attention_window_size`, `logits_soft_cap`, `scan_type`, will be taken
        from the preset, unless the corresponding argument to this method is
        set, in which case that will take precedence.
      embeddings_scale_by_sqrt_dim: Whether to scale the output of the
        embeddings by `sqrt(width)`. If this is `None` it is taken from the
        `preset` hypers. This argument or `preset` must be set.
      attention_window_size: The size of the attention window used in the
        attention block. If this is `None` it is taken from the `preset` hypers.
        This argument or `preset` must be set.
      logits_soft_cap: This will cap the values of the final logits to not
        exceed this cap in absolute value by applying a `tanh`. If this is
        `None` it is taken from the `preset` hypers. This argument or `preset`
        must be set.
      scan_type: If running Flax, this specifies which implementation to use for
        the scan in the RG-LRU. If this is `None` it is taken from the `preset`
        hypers. This argument or `preset` must be set.
      max_sequence_length: The maximum sequence length this models is intended
        to process. If this is lower than 2048, the `attention_window_size` will
        best to this value instead, in order to not create a cache that is
        larger than necessary.

    Returns:
      The reconstructed `GriffinConfig`.
    """
    if "params" in flax_params_or_variables:
      params = flax_params_or_variables["params"]
    else:
      params = flax_params_or_variables

    vocab_size, width = params["embedder"]["input_embedding"].shape
    mlp_exp_width = params["blocks.0"]["mlp_block"]["ffw_up"]["w"].shape[-1]

    # Defaults
    lru_width = None
    num_heads = None

    block_types = []
    i = 0
    while f"blocks.{i}" in params:
      block_params = params[f"blocks.{i}"]
      if "recurrent_block" in block_params:
        block_types.append(TemporalBlockType.RECURRENT)

        rg_lru = block_params["recurrent_block"]["rg_lru"]
        num_heads, head_dim, _ = rg_lru["a_gate"]["w"].shape
        lru_width = num_heads * head_dim

      elif "attention_block" in block_params:
        block_types.append(TemporalBlockType.ATTENTION)

        k_proj = block_params["attention_block"]["proj_k"]
        heads_dim = k_proj["kernel"].shape[1]
        num_heads = width // heads_dim

      else:
        raise ValueError(
            f"Can't recongnize the type of blocks.{i} with keys"
            f"{block_params.keys()}."
        )

      i += 1

    return cls._from_parameter_kwargs(
        kwargs=dict(
            vocab_size=vocab_size,
            width=width,
            mlp_expanded_width=mlp_exp_width,
            num_heads=num_heads,
            lru_width=lru_width,
            block_types=tuple(block_types),
        ),
        preset=preset,
        embeddings_scale_by_sqrt_dim=embeddings_scale_by_sqrt_dim,
        attention_window_size=attention_window_size,
        logits_soft_cap=logits_soft_cap,
        scan_type=scan_type,
        max_sequence_length=max_sequence_length,
    )

  @classmethod
  def from_torch_params(
      cls,
      params: dict[str, Any],
      preset: Preset | None = None,
      embeddings_scale_by_sqrt_dim: bool | None = None,
      attention_window_size: int | None = None,
      logits_soft_cap: float | None = None,
      scan_type: ScanType | None = None,
      max_sequence_length: int | None = None,
  ) -> "GriffinConfig":
    """Creates a `GriffinConfig` from Pytorch parameters.

    Args:
      params: The Pytorch parameters to use to reconstruct the config.
      preset: The expected preset from which the parameters have been derived.
        If this is set values for hyper parameters that can't be inferred from
        the parameter, such as `embeddings_scale_by_sqrt_dim`,
        `attention_window_size`, `logits_soft_cap`, `scan_type, will be taken
        from the preset, unless the corresponding argument to this method is
        set, in which case that will take precedence.
      embeddings_scale_by_sqrt_dim: Whether to scale the output of the
        embeddings by `sqrt(width)`. If this is `None` it is taken from the
        `preset` hypers. This argument or `preset` must be set.
      attention_window_size: The size of the attention window used in the
        attention block. If this is `None` it is taken from the `preset` hypers.
        This argument or `preset` must be set.
      logits_soft_cap: This will cap the values of the final logits to not
        exceed this cap in absolute value by applying a `tanh`. If this is
        `None` it is taken from the `preset` hypers. This argument or `preset`
        must be set.
      scan_type: If running Flax, this specifies which implementation to use for
        the scan in the RG-LRU. If this is `None` it is taken from the `preset`
        hypers. This argument or `preset` must be set.
      max_sequence_length: The maximum sequence length this models is intended
        to process. If this is lower than 2048, the `attention_window_size` will
        be set to this value instead, in order to not create a cache that is
        larger than necessary.

    Returns:
      The reconstructed `GriffinConfig`.
    """

    vocab_size, width = params["embedder.input_embedding"].shape
    mlp_exp_width = params["blocks.0.mlp_block.ffw_up.w"].shape[-1]

    # Defaults
    lru_width = None
    num_heads = None

    block_types = []
    i = 0

    while f"blocks.{i}.channel_pre_norm.scale" in params:
      if f"blocks.{i}.recurrent_block.rg_lru.a_gate.w" in params:
        block_types.append(TemporalBlockType.RECURRENT)

        w = params[f"blocks.{i}.recurrent_block.rg_lru.a_gate.w"]
        num_heads, head_dim, _ = w.shape
        lru_width = num_heads * head_dim

      elif f"blocks.{i}.attention_block.proj_k.weight" in params:
        block_types.append(TemporalBlockType.ATTENTION)

        heads_dim = params[f"blocks.{i}.attention_block.proj_k.weight"].shape[1]
        num_heads = width // heads_dim

      else:
        raise ValueError(f"Can't recongnize the type of blocks.{i}.")

      i += 1

    return cls._from_parameter_kwargs(
        kwargs=dict(
            vocab_size=vocab_size,
            width=width,
            mlp_expanded_width=mlp_exp_width,
            num_heads=num_heads,
            lru_width=lru_width,
            block_types=tuple(block_types),
        ),
        preset=preset,
        embeddings_scale_by_sqrt_dim=embeddings_scale_by_sqrt_dim,
        attention_window_size=attention_window_size,
        logits_soft_cap=logits_soft_cap,
        scan_type=scan_type,
        max_sequence_length=max_sequence_length,
    )


def apply_it_formatter(input_string: str) -> str:
  return f"<start_of_turn>user\n{input_string}<end_of_turn>\n<start_of_turn>model\n"
