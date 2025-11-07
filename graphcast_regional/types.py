# Copyright 2024 Regional Weather Prediction Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Type definitions for Regional Weather Prediction System."""

from typing import Any, Dict, Mapping, Tuple, Union

import jax.numpy as jnp
import numpy as np


# Array types - can be numpy or JAX arrays
ArrayLike = Union[np.ndarray, jnp.ndarray, Any]

# Tree of arrays (nested structures)
ArrayLikeTree = Union[ArrayLike, Mapping[str, Any], Tuple[Any, ...]]

# Model parameters (Haiku params structure)
Params = Mapping[str, Mapping[str, ArrayLike]]

# Optimizer state
OptState = Any

# Coordinate arrays
Coordinates = Tuple[np.ndarray, np.ndarray]  # (lat, lon)

# Node indices
NodeIndices = np.ndarray  # 1D array of integer indices

# Edge indices
EdgeIndices = Tuple[np.ndarray, np.ndarray]  # (senders, receivers)

# Feature arrays
NodeFeatures = ArrayLike  # Shape: [num_nodes, feature_dim]
EdgeFeatures = ArrayLike  # Shape: [num_edges, feature_dim]

# Data dimensions
DataDimensions = Dict[str, int]  # e.g., {"time": 100, "lat": 201, "lon": 281}

# Pressure levels (hPa)
PRESSURE_LEVELS = np.array([100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000])

# HPA variable names
HPA_VARIABLES = ["DPT", "GPH", "TEM", "U", "V"]

# Number of input channels: 5 HPA vars × 11 levels × 2 timesteps + 1 precip × 2 timesteps
NUM_INPUT_CHANNELS = len(HPA_VARIABLES) * len(PRESSURE_LEVELS) * 2 + 2  # = 112

# Node type names for typed graph
UPSTREAM_NODE_TYPE = "upstream_nodes"
DOWNSTREAM_NODE_TYPE = "downstream_nodes"

# Edge type names for typed graph
DOWNSTREAM_INTRA_EDGE_TYPE = "downstream_intra"
UPSTREAM_TO_DOWNSTREAM_EDGE_TYPE = "upstream_to_downstream"
