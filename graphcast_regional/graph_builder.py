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
"""Regional graph builder for constructing GNN connectivity."""

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from graphcast import typed_graph
from graphcast_regional.config import RegionConfig
from graphcast_regional import types


class RegionalGraphBuilder:
    """Builds graph structure for regional weather prediction.
    
    This class constructs a TypedGraph with two node types (upstream and downstream)
    and two edge types (intra-domain within downstream, and inter-domain from 
    upstream to downstream boundary).
    """
    
    def __init__(
        self,
        region_config: RegionConfig,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
    ):
        """Initialize the regional graph builder.
        
        Args:
            region_config: Configuration defining regional boundaries and connectivity.
            lat_coords: 1D array of latitude coordinates (degrees).
            lon_coords: 1D array of longitude coordinates (degrees).
        """
        self.region_config = region_config
        self.lat_coords = lat_coords
        self.lon_coords = lon_coords
        
        # Create 2D meshgrid for all grid points
        self.lon_grid, self.lat_grid = np.meshgrid(lon_coords, lat_coords)
        
        # Flatten to get all grid points as (lat, lon) pairs
        self.all_lats = self.lat_grid.flatten()
        self.all_lons = self.lon_grid.flatten()
        
        # Total number of grid points
        self.num_total_nodes = len(self.all_lats)
        
    def extract_region_nodes(self, region: str) -> np.ndarray:
        """Extract node indices for a specified region.
        
        Args:
            region: Either "upstream" or "downstream".
            
        Returns:
            1D array of node indices within the specified region.
            
        Raises:
            ValueError: If region is not "upstream" or "downstream".
        """
        if region == "downstream":
            lat_min = self.region_config.downstream_lat_min
            lat_max = self.region_config.downstream_lat_max
            lon_min = self.region_config.downstream_lon_min
            lon_max = self.region_config.downstream_lon_max
        elif region == "upstream":
            lat_min = self.region_config.upstream_lat_min
            lat_max = self.region_config.upstream_lat_max
            lon_min = self.region_config.upstream_lon_min
            lon_max = self.region_config.upstream_lon_max
        else:
            raise ValueError(f"region must be 'upstream' or 'downstream', got '{region}'")
        
        # Find nodes within the region boundaries
        mask = (
            (self.all_lats >= lat_min) &
            (self.all_lats <= lat_max) &
            (self.all_lons >= lon_min) &
            (self.all_lons <= lon_max)
        )
        
        node_indices = np.where(mask)[0]
        
        # Validate that we found nodes
        if len(node_indices) == 0:
            raise ValueError(
                f"No nodes found in {region} region with boundaries: "
                f"lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]"
            )
        
        return node_indices
    
    def build_intra_domain_edges(
        self, 
        node_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build k-NN edges within a region.
        
        Args:
            node_indices: Array of node indices within the region.
            
        Returns:
            Tuple of (senders, receivers, edge_features):
                - senders: Array of sender node indices
                - receivers: Array of receiver node indices  
                - edge_features: Array of edge features [num_edges, 3]
                  (relative_lat, relative_lon, distance)
        """
        # Get coordinates for nodes in this region
        lats = self.all_lats[node_indices]
        lons = self.all_lons[node_indices]
        
        # Stack into (N, 2) array for k-NN search
        coords = np.stack([lats, lons], axis=1)
        
        # Build k-d tree for efficient nearest neighbor search
        tree = cKDTree(coords)
        
        # Query k+1 nearest neighbors (including self)
        k = self.region_config.intra_domain_k_neighbors
        distances, neighbor_indices = tree.query(coords, k=k+1)
        
        # Build edge lists
        senders_list = []
        receivers_list = []
        edge_features_list = []
        
        for i in range(len(node_indices)):
            # Skip the first neighbor (self)
            for j in range(1, k+1):
                neighbor_idx = neighbor_indices[i, j]
                
                # Map back to global node indices
                sender = node_indices[i]
                receiver = node_indices[neighbor_idx]
                
                # Compute edge features
                relative_lat = lats[neighbor_idx] - lats[i]
                relative_lon = lons[neighbor_idx] - lons[i]
                distance = distances[i, j]
                
                senders_list.append(sender)
                receivers_list.append(receiver)
                edge_features_list.append([relative_lat, relative_lon, distance])
        
        senders = np.array(senders_list, dtype=np.int32)
        receivers = np.array(receivers_list, dtype=np.int32)
        edge_features = np.array(edge_features_list, dtype=np.float32)
        
        return senders, receivers, edge_features
    
    def build_inter_domain_edges(
        self,
        upstream_indices: np.ndarray,
        downstream_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build edges from upstream to downstream boundary nodes.
        
        Args:
            upstream_indices: Array of upstream node indices.
            downstream_indices: Array of downstream node indices.
            
        Returns:
            Tuple of (senders, receivers, edge_features):
                - senders: Array of sender node indices (upstream)
                - receivers: Array of receiver node indices (downstream boundary)
                - edge_features: Array of edge features [num_edges, 3]
                  (relative_lat, relative_lon, distance)
        """
        # Identify boundary nodes in downstream region
        # Boundary nodes are those near the western edge (close to upstream)
        downstream_lons = self.all_lons[downstream_indices]
        boundary_mask = downstream_lons <= self.region_config.boundary_threshold_lon
        boundary_indices = downstream_indices[boundary_mask]
        
        if len(boundary_indices) == 0:
            raise ValueError(
                f"No boundary nodes found in downstream region with threshold "
                f"lon <= {self.region_config.boundary_threshold_lon}"
            )
        
        # Get coordinates
        upstream_lats = self.all_lats[upstream_indices]
        upstream_lons = self.all_lons[upstream_indices]
        upstream_coords = np.stack([upstream_lats, upstream_lons], axis=1)
        
        boundary_lats = self.all_lats[boundary_indices]
        boundary_lons = self.all_lons[boundary_indices]
        boundary_coords = np.stack([boundary_lats, boundary_lons], axis=1)
        
        # Build k-d tree for upstream nodes
        upstream_tree = cKDTree(upstream_coords)
        
        # For each boundary node, find k nearest upstream nodes
        k = self.region_config.inter_domain_k_neighbors
        distances, neighbor_indices = upstream_tree.query(boundary_coords, k=k)
        
        # Build edge lists
        senders_list = []
        receivers_list = []
        edge_features_list = []
        
        for i in range(len(boundary_indices)):
            receiver = boundary_indices[i]
            
            for j in range(k):
                neighbor_idx = neighbor_indices[i, j] if k > 1 else neighbor_indices[i]
                sender = upstream_indices[neighbor_idx]
                
                # Compute edge features
                if k > 1:
                    relative_lat = upstream_lats[neighbor_idx] - boundary_lats[i]
                    relative_lon = upstream_lons[neighbor_idx] - boundary_lons[i]
                    distance = distances[i, j]
                else:
                    relative_lat = upstream_lats[neighbor_idx] - boundary_lats[i]
                    relative_lon = upstream_lons[neighbor_idx] - boundary_lons[i]
                    distance = distances[i]
                
                senders_list.append(sender)
                receivers_list.append(receiver)
                edge_features_list.append([relative_lat, relative_lon, distance])
        
        senders = np.array(senders_list, dtype=np.int32)
        receivers = np.array(receivers_list, dtype=np.int32)
        edge_features = np.array(edge_features_list, dtype=np.float32)
        
        return senders, receivers, edge_features
    
    def build_graph(self) -> typed_graph.TypedGraph:
        """Build complete regional graph structure.
        
        Returns:
            TypedGraph with upstream and downstream node sets, and intra-domain
            and inter-domain edge sets.
        """
        # Extract region nodes
        upstream_indices = self.extract_region_nodes("upstream")
        downstream_indices = self.extract_region_nodes("downstream")
        
        # Build intra-domain edges for downstream region
        down_senders, down_receivers, down_edge_features = self.build_intra_domain_edges(
            downstream_indices
        )
        
        # Build inter-domain edges from upstream to downstream
        inter_senders, inter_receivers, inter_edge_features = self.build_inter_domain_edges(
            upstream_indices, downstream_indices
        )
        
        # Create node features (spatial coordinates)
        upstream_lats = self.all_lats[upstream_indices]
        upstream_lons = self.all_lons[upstream_indices]
        upstream_features = np.stack([upstream_lats, upstream_lons], axis=1).astype(np.float32)
        
        downstream_lats = self.all_lats[downstream_indices]
        downstream_lons = self.all_lons[downstream_indices]
        downstream_features = np.stack([downstream_lats, downstream_lons], axis=1).astype(np.float32)
        
        # Create NodeSets
        upstream_node_set = typed_graph.NodeSet(
            n_node=np.array([len(upstream_indices)], dtype=np.int32),
            features=upstream_features
        )
        
        downstream_node_set = typed_graph.NodeSet(
            n_node=np.array([len(downstream_indices)], dtype=np.int32),
            features=downstream_features
        )
        
        # Create EdgeSets
        downstream_intra_edge_set = typed_graph.EdgeSet(
            n_edge=np.array([len(down_senders)], dtype=np.int32),
            indices=typed_graph.EdgesIndices(
                senders=down_senders,
                receivers=down_receivers
            ),
            features=down_edge_features
        )
        
        upstream_to_downstream_edge_set = typed_graph.EdgeSet(
            n_edge=np.array([len(inter_senders)], dtype=np.int32),
            indices=typed_graph.EdgesIndices(
                senders=inter_senders,
                receivers=inter_receivers
            ),
            features=inter_edge_features
        )
        
        # Create Context (global features - empty tuple means no context features)
        context = typed_graph.Context(
            n_graph=np.array([1], dtype=np.int32),
            features=()
        )
        
        # Create EdgeSetKeys
        downstream_intra_key = typed_graph.EdgeSetKey(
            name=types.DOWNSTREAM_INTRA_EDGE_TYPE,
            node_sets=(types.DOWNSTREAM_NODE_TYPE, types.DOWNSTREAM_NODE_TYPE)
        )
        
        upstream_to_downstream_key = typed_graph.EdgeSetKey(
            name=types.UPSTREAM_TO_DOWNSTREAM_EDGE_TYPE,
            node_sets=(types.UPSTREAM_NODE_TYPE, types.DOWNSTREAM_NODE_TYPE)
        )
        
        # Build TypedGraph
        graph = typed_graph.TypedGraph(
            context=context,
            nodes={
                types.UPSTREAM_NODE_TYPE: upstream_node_set,
                types.DOWNSTREAM_NODE_TYPE: downstream_node_set,
            },
            edges={
                downstream_intra_key: downstream_intra_edge_set,
                upstream_to_downstream_key: upstream_to_downstream_edge_set,
            }
        )
        
        return graph
