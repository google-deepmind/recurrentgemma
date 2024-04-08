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
"""Common types used by layers and modules."""

from typing import Any, Callable, Mapping, TypeVar

import jax.numpy as jnp
import jaxtyping as jt
import typeguard


F = TypeVar("F", bound=Callable)


def typed(function: F) -> F:
  return jt.jaxtyped(function, typechecker=typeguard.typechecked)


# Notation:
# b = batch size
# t = number of tokens
# d = model dimension
# e = expanded model dimension in the Recurrent Block
# w = conv1d window size
# v = vocab size
# n = number of heads for multi-query attention.
# s = number of keys/values
# h = head dimension

# General
Activations = jt.Float[jt.Array, "*b t d"]
SegmentPos = jt.Integer[jt.Array, "*b t"]
Tokens = jt.Integer[jt.Array, "*b t"]
TokenLogits = jt.Float[jt.Array, "*b t v"]
Params = Mapping[str, Any]
dtype = str | type(jnp.float64)

# Attention block
Queries = jt.Float[jt.Array, "*b t n h"]
Keys = jt.Float[jt.Array, "*b t 1 h"]
Values = jt.Float[jt.Array, "*b t 1 h"]
QuerySegmentIds = jt.Integer[jt.Array, "*b t"]
KeySegmentIds = jt.Integer[jt.Array, "*b t"]
CachedKeys = jt.Float[jt.Array, "*b s 1 h"]
CachedValues = jt.Float[jt.Array, "*b s 1 h"]
NumTokens = jt.Integer[jt.Array, "*b"]
AttentionMask = jt.Bool[jt.Array, "*b t s"]

# Recurrent block
ExpandedActivations = jt.Float[jt.Array, "*b t e"]
RNNDiagonal = jt.Float[jt.Array, "e"]
RNNState = jt.Float[jt.Array, "*b e"]
Conv1DState = jt.Float[jt.Array, "*b w e"]
Reset = jt.Bool[jt.Array, "*b t"]
