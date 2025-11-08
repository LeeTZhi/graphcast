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
"""Regional GNN model for weather prediction."""

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import xarray as xr

from graphcast import deep_typed_graph_net
from graphcast import typed_graph
from graphcast_regional.config import ModelConfig, RegionConfig
from graphcast_regional import types


def prepare_input_features(
    inputs: xr.Dataset,
    upstream_indices: jnp.ndarray,
    downstream_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Prepare input features for all nodes.
    
    Stacks HPA variables across levels and timesteps, plus precipitation,
    to create feature vector per node. Supports variable number of timesteps.
    
    Args:
        inputs: xarray Dataset with dimensions (time=N, level=11, lat, lon)
            containing HPA variables (DPT, GPH, TEM, U, V) and precipitation.
            Default N=6 for 3 days of history (12-hour intervals).
        upstream_indices: Array of upstream node indices [num_upstream].
        downstream_indices: Array of downstream node indices [num_downstream].
        
    Returns:
        Array of shape [num_total_nodes, num_features] with stacked features.
        For 6 timesteps: 5 vars × 11 levels × 6 timesteps + 1 var × 6 timesteps = 336 channels
    """
    # Extract dimensions
    num_upstream = len(upstream_indices)
    num_downstream = len(downstream_indices)
    num_total_nodes = num_upstream + num_downstream
    
    # Get number of timesteps dynamically
    num_timesteps = len(inputs.time)
    
    # Stack HPA variables: 5 vars × 11 levels × num_timesteps channels
    hpa_features = []
    for var_name in types.HPA_VARIABLES:
        var_data = inputs[var_name].values  # Shape: (num_timesteps, 11, lat, lon)
        # Flatten spatial dimensions
        var_flat = var_data.reshape(num_timesteps, 11, -1)  # (num_timesteps, 11, num_grid_points)
        # Transpose to (num_grid_points, num_timesteps, 11)
        var_flat = jnp.transpose(var_flat, (2, 0, 1))
        # Flatten time and level: (num_grid_points, num_timesteps * 11)
        var_flat = var_flat.reshape(-1, num_timesteps * 11)
        hpa_features.append(var_flat)
    
    # Stack all HPA variables: (num_grid_points, 5 * num_timesteps * 11)
    hpa_features = jnp.concatenate(hpa_features, axis=-1)
    
    # Stack precipitation: 1 var × num_timesteps channels
    precip_data = inputs["precipitation"].values  # Shape: (num_timesteps, lat, lon)
    precip_flat = precip_data.reshape(num_timesteps, -1)  # (num_timesteps, num_grid_points)
    precip_flat = jnp.transpose(precip_flat, (1, 0))  # (num_grid_points, num_timesteps)
    
    # Concatenate HPA and precipitation: (num_grid_points, 5*num_timesteps*11 + num_timesteps)
    all_features = jnp.concatenate([hpa_features, precip_flat], axis=-1)
    
    # Extract features for upstream and downstream nodes
    # Concatenate indices to maintain order: upstream first, then downstream
    all_indices = jnp.concatenate([upstream_indices, downstream_indices])
    node_features = all_features[all_indices]
    
    return node_features


class RegionalGNN(hk.Module):
    """Regional GNN for weather prediction.
    
    Implements an encoder-processor-decoder architecture:
    - Encoder: MLP that transforms 112-dim input to latent_size
    - Processor: Multiple GNN layers with message passing
    - Decoder: MLP that transforms latent_size to 1 (precipitation)
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        region_config: RegionConfig,
        name: str = "regional_gnn",
    ):
        """Initialize Regional GNN model.
        
        Args:
            model_config: Model architecture configuration.
            region_config: Region boundaries configuration.
            name: Module name.
        """
        super().__init__(name=name)
        self.model_config = model_config
        self.region_config = region_config
        
    def _build_encoder(self) -> hk.Module:
        """Build encoder MLP: 112 -> latent_size.
        
        Returns:
            Haiku MLP module for encoding.
        """
        output_sizes = (
            [self.model_config.mlp_hidden_size] * self.model_config.mlp_num_hidden_layers +
            [self.model_config.latent_size]
        )
        
        activation_fn = self._get_activation_fn()
        
        encoder = hk.nets.MLP(
            output_sizes=output_sizes,
            activation=activation_fn,
            name="encoder",
        )
        
        if self.model_config.use_layer_norm:
            return hk.Sequential([
                encoder,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="encoder_ln")
            ])
        else:
            return encoder

    def _build_processor(self) -> deep_typed_graph_net.DeepTypedGraphNet:
        """Build GNN processor with multiple message passing layers.
        
        Returns:
            DeepTypedGraphNet module for message passing.
        """
        # Define latent sizes for each node type
        node_latent_size = {
            types.UPSTREAM_NODE_TYPE: self.model_config.latent_size,
            types.DOWNSTREAM_NODE_TYPE: self.model_config.latent_size,
        }
        
        # Define latent sizes for each edge type
        edge_latent_size = {
            types.DOWNSTREAM_INTRA_EDGE_TYPE: self.model_config.latent_size,
            types.UPSTREAM_TO_DOWNSTREAM_EDGE_TYPE: self.model_config.latent_size,
        }
        
        processor = deep_typed_graph_net.DeepTypedGraphNet(
            node_latent_size=node_latent_size,
            edge_latent_size=edge_latent_size,
            mlp_hidden_size=self.model_config.mlp_hidden_size,
            mlp_num_hidden_layers=self.model_config.mlp_num_hidden_layers,
            num_message_passing_steps=self.model_config.num_gnn_layers,
            embed_nodes=True,   # Let DeepTypedGraphNet handle node embedding
            embed_edges=True,   # Let DeepTypedGraphNet handle edge embedding
            use_layer_norm=self.model_config.use_layer_norm,
            activation=self.model_config.activation,
            include_sent_messages_in_node_update=False,
            f32_aggregation=True,
            aggregate_normalization=None,
            name="processor",
        )
        
        return processor
    
    def _build_decoder(self) -> hk.Module:
        """Build decoder MLP: latent_size -> 1.
        
        Returns:
            Haiku MLP module for decoding.
        """
        output_sizes = (
            [self.model_config.mlp_hidden_size] * self.model_config.mlp_num_hidden_layers +
            [1]
        )
        
        activation_fn = self._get_activation_fn()
        
        decoder = hk.nets.MLP(
            output_sizes=output_sizes,
            activation=activation_fn,
            name="decoder",
        )
        
        return decoder
    
    def _get_activation_fn(self):
        """Get activation function from config.
        
        Returns:
            Activation function.
        """
        activation_map = {
            "swish": jax.nn.swish,
            "relu": jax.nn.relu,
            "gelu": jax.nn.gelu,
            "elu": jax.nn.elu,
            "tanh": jnp.tanh,
        }
        
        return activation_map.get(
            self.model_config.activation,
            jax.nn.swish  # Default to swish
        )
    
    def __call__(
        self,
        inputs: xr.Dataset,
        graph: typed_graph.TypedGraph,
        is_training: bool = False,
    ) -> xr.Dataset:
        """Forward pass through the model.
        
        Args:
            inputs: xarray Dataset with dimensions (time=2, level=11, lat, lon)
                containing HPA variables and precipitation.
            graph: TypedGraph with upstream and downstream nodes and edges.
            is_training: Whether in training mode (for dropout, etc.).
            
        Returns:
            xarray Dataset with precipitation predictions for downstream region.
            Shape: (lat_downstream, lon_downstream).
        """
        # Extract node indices from graph
        upstream_node_set = graph.nodes[types.UPSTREAM_NODE_TYPE]
        downstream_node_set = graph.nodes[types.DOWNSTREAM_NODE_TYPE]
        
        num_upstream = int(upstream_node_set.n_node[0])
        num_downstream = int(downstream_node_set.n_node[0])
        
        # Get spatial coordinates from graph node features
        upstream_coords = upstream_node_set.features  # (num_upstream, 2)
        downstream_coords = downstream_node_set.features  # (num_downstream, 2)
        
        # Create node indices (assuming flattened grid ordering)
        # We need to map from (lat, lon) coordinates back to flat indices
        lat_coords = inputs.lat.values
        lon_coords = inputs.lon.values
        
        upstream_indices = self._coords_to_indices(
            upstream_coords, lat_coords, lon_coords
        )
        downstream_indices = self._coords_to_indices(
            downstream_coords, lat_coords, lon_coords
        )
        
        # Prepare input features: (num_total_nodes, 112)
        node_features = prepare_input_features(
            inputs, upstream_indices, downstream_indices
        )
        
        # Split features for upstream and downstream
        upstream_features = node_features[:num_upstream]
        downstream_features = node_features[num_upstream:]
        
        # Update graph with input node features (not yet encoded)
        graph_with_features = graph._replace(
            nodes={
                types.UPSTREAM_NODE_TYPE: upstream_node_set._replace(
                    features=upstream_features
                ),
                types.DOWNSTREAM_NODE_TYPE: downstream_node_set._replace(
                    features=downstream_features
                ),
            }
        )
        
        # Build and apply processor (which will embed nodes and edges)
        processor = self._build_processor()
        processed_graph = processor(graph_with_features)
        
        # Extract downstream node features after processing
        downstream_processed = processed_graph.nodes[types.DOWNSTREAM_NODE_TYPE].features
        
        # Build decoder
        decoder = self._build_decoder()
        
        # Decode only downstream nodes: (num_downstream, 1)
        predictions = decoder(downstream_processed)
        predictions = jnp.squeeze(predictions, axis=-1)  # (num_downstream,)
        
        # Extract downstream coordinates
        downstream_lat_min = self.region_config.downstream_lat_min
        downstream_lat_max = self.region_config.downstream_lat_max
        downstream_lon_min = self.region_config.downstream_lon_min
        downstream_lon_max = self.region_config.downstream_lon_max
        
        downstream_lat_mask = (lat_coords >= downstream_lat_min) & (lat_coords <= downstream_lat_max)
        downstream_lon_mask = (lon_coords >= downstream_lon_min) & (lon_coords <= downstream_lon_max)
        
        downstream_lats = lat_coords[downstream_lat_mask]
        downstream_lons = lon_coords[downstream_lon_mask]
        
        # Reshape predictions back to downstream spatial grid only
        predictions_grid = self._predictions_to_grid(
            predictions, downstream_coords, downstream_lats, downstream_lons
        )
        
        # Create output xarray Dataset
        output = xr.Dataset(
            {
                "precipitation": (["lat", "lon"], predictions_grid),
            },
            coords={
                "lat": downstream_lats,
                "lon": downstream_lons,
            },
        )
        
        return output
    
    def _coords_to_indices(
        self,
        coords: jnp.ndarray,
        lat_coords: jnp.ndarray,
        lon_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        """Convert (lat, lon) coordinates to flat grid indices.
        
        Args:
            coords: Array of shape (num_nodes, 2) with (lat, lon) pairs.
            lat_coords: 1D array of latitude coordinates.
            lon_coords: 1D array of longitude coordinates.
            
        Returns:
            Array of flat indices.
        """
        num_lats = len(lat_coords)
        num_lons = len(lon_coords)
        
        # Find closest indices for each coordinate
        lats = coords[:, 0]
        lons = coords[:, 1]
        
        # Find indices using broadcasting
        lat_indices = jnp.argmin(jnp.abs(lat_coords[None, :] - lats[:, None]), axis=1)
        lon_indices = jnp.argmin(jnp.abs(lon_coords[None, :] - lons[:, None]), axis=1)
        
        # Convert to flat indices (row-major order)
        flat_indices = lat_indices * num_lons + lon_indices
        
        return flat_indices
    
    def _predictions_to_grid(
        self,
        predictions: jnp.ndarray,
        coords: jnp.ndarray,
        lat_coords: jnp.ndarray,
        lon_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        """Convert flat predictions to spatial grid.
        
        Args:
            predictions: Array of shape (num_nodes,) with predictions.
            coords: Array of shape (num_nodes, 2) with (lat, lon) pairs.
            lat_coords: 1D array of latitude coordinates.
            lon_coords: 1D array of longitude coordinates.
            
        Returns:
            Array of shape (num_lats, num_lons) with predictions on grid.
        """
        # Find grid indices for each prediction
        lats = coords[:, 0]
        lons = coords[:, 1]
        
        lat_indices = jnp.argmin(jnp.abs(lat_coords[None, :] - lats[:, None]), axis=1)
        lon_indices = jnp.argmin(jnp.abs(lon_coords[None, :] - lons[:, None]), axis=1)
        
        # Create output grid
        num_lats = len(lat_coords)
        num_lons = len(lon_coords)
        grid = jnp.zeros((num_lats, num_lons))
        
        # Fill in predictions (using at indexing)
        grid = grid.at[lat_indices, lon_indices].set(predictions)
        
        return grid
