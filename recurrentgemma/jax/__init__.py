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
"""Space-Gemma Jax public API."""

from recurrentgemma import common
from recurrentgemma.jax import griffin
from recurrentgemma.jax import layers
from recurrentgemma.jax import modules
from recurrentgemma.jax import pallas
from recurrentgemma.jax import sampler
from recurrentgemma.jax import utils


ScanType = common.ScanType
TemporalBlockType = common.TemporalBlockType
GriffinConfig = common.GriffinConfig
PallasShardingSpec = pallas.PallasShardingSpec
sharded_lru = pallas.sharded_lru
rnn_scan = layers.rnn_scan
BlockDiagonalLinear = layers.BlockDiagonalLinear
RGLRU = layers.RGLRU
Conv1D = layers.Conv1D
RecurrentBlockCache = modules.RecurrentBlockCache
RecurrentBlock = modules.RecurrentBlock
AttentionBlockCache = modules.AttentionBlockCache
LocalAttentionBlock = modules.LocalAttentionBlock
ResidualBlockCache = modules.ResidualBlockCache
ResidualBlock = modules.ResidualBlock
Griffin = griffin.Griffin
Sampler = sampler.Sampler
load_parameters = utils.load_parameters


__all__ = (
    "ScanType",
    "TemporalBlockType",
    "PallasShardingSpec",
    "sharded_lru",
    "rnn_scan",
    "BlockDiagonalLinear",
    "RGLRU",
    "Conv1D",
    "RecurrentBlockCache",
    "RecurrentBlock",
    "AttentionBlockCache",
    "LocalAttentionBlock",
    "ResidualBlockCache",
    "ResidualBlock",
    "GriffinConfig",
    "Griffin",
    "Sampler",
    "load_parameters",
)

# Prevents from accessing anything except the exported symbols
try:
  del jax, common  # pylint: disable=undefined-variable
except NameError:
  pass
