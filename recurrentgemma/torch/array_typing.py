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

from typing import Callable, TypeVar, Mapping, Any

import jaxtyping as jt
import torch


F = TypeVar("F", bound=Callable)


def typed(function: F) -> F:
  # We comment out this, since it breaks torch.compile
  # return jt.jaxtyped(function, typechecker=typeguard.typechecked)
  return function

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
Activations = jt.Float[torch.Tensor, "*b t d"]
SegmentPos = jt.Integer[torch.Tensor, "*b t"]
Tokens = jt.Integer[torch.Tensor, "*b t"]
TokenLogits = jt.Float[torch.Tensor, "*b t v"]
Params = Mapping[str, Any]

# Attention block
Queries = jt.Float[torch.Tensor, "*b t n h"]
Keys = jt.Float[torch.Tensor, "*b t 1 h"]
Values = jt.Float[torch.Tensor, "*b t 1 h"]
QuerySegmentIds = jt.Integer[torch.Tensor, "*b t"]
KeySegmentIds = jt.Integer[torch.Tensor, "*b t"]
CachedKeys = jt.Float[torch.Tensor, "*b s 1 h"]
CachedValues = jt.Float[torch.Tensor, "*b s 1 h"]
NumTokens = jt.Integer[torch.Tensor, "*b"]
AttentionMask = jt.Bool[torch.Tensor, "*b t s"]

# Recurrent block
ExpandedActivations = jt.Float[torch.Tensor, "*b t e"]
RNNDiagonal = jt.Float[torch.Tensor, "e"]
RNNState = jt.Float[torch.Tensor, "*b e"]
Conv1DState = jt.Float[torch.Tensor, "*b w e"]
Reset = jt.Bool[torch.Tensor, "*b t"]
