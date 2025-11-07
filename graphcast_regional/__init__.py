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
"""Regional Weather Prediction System using Graph Neural Networks."""

from graphcast_regional.config import ModelConfig, RegionConfig, TrainingConfig
from graphcast_regional.types import (
    ArrayLike,
    ArrayLikeTree,
    Params,
    OptState,
)

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "RegionConfig",
    "TrainingConfig",
    "ArrayLike",
    "ArrayLikeTree",
    "Params",
    "OptState",
]
