"""
Graph Structure for Skeleton-based Models
Defines the connectivity of human skeleton joints.

Supports:
- MediaPipe Pose (33 keypoints)
- OpenPose (25 keypoints)
- COCO (17 keypoints)
- NTU RGB+D (25 keypoints)
"""

import numpy as np


class Graph:
    """
    Graph structure for skeleton data.
    Defines edges (bones) and partitioning strategies.
    """
    
    def __init__(
        self,
        layout='mediapipe',
        strategy='spatial',
        max_hop=1,
        dilation=1
    ):
        """
        Initialize graph structure.
        
        Args:
            layout: Skeleton layout ('mediapipe', 'openpose', 'ntu', 'coco')
            strategy: Partition strategy ('uniform', 'distance', 'spatial')
            max_hop: Maximum distance of neighbors
            dilation: Dilation for edges
        """
        self.max_hop = max_hop
        self.dilation = dilation
        self.layout = layout
        
        # Get skeleton configuration
        self.num_node, self.edge, self.center = self._get_skeleton_layout(layout)
        
        # Build adjacency matrix
        self.hop_dis = self._get_hop_distance(self.num_node, self.edge, max_hop)
        self.A = self._get_adjacency_matrix(self.num_node, self.hop_dis, strategy)
        
    def _get_skeleton_layout(self, layout):
        """Get skeleton configuration for different layouts."""
        
        if layout == 'mediapipe':
            # MediaPipe Pose (33 keypoints)
            num_node = 33
            
            # Define edges (connections between keypoints)
            # Format: (parent, child)
            edge = [
                # Face
                (0, 1), (1, 2), (2, 3), (3, 7),  # Right face
                (0, 4), (4, 5), (5, 6), (6, 8),  # Left face
                (9, 10),  # Mouth
                
                # Torso
                (11, 12),  # Shoulders
                (11, 23), (12, 24),  # Hips
                (23, 24),  # Hip connection
                
                # Right arm
                (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                
                # Left arm
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                
                # Right leg
                (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                
                # Left leg
                (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
            ]
            
            center = 0  # Nose as center
            
        elif layout == 'openpose':
            # OpenPose Body 25
            num_node = 25
            
            edge = [
                # Spine
                (0, 1), (1, 2), (2, 3), (3, 4),
                (1, 5), (1, 8),
                
                # Right arm
                (5, 6), (6, 7),
                
                # Left arm
                (8, 9), (9, 10),
                
                # Right leg
                (0, 11), (11, 12), (12, 13),
                
                # Left leg
                (0, 14), (14, 15), (15, 16),
                
                # Face
                (0, 17), (0, 18),
                
                # Feet
                (13, 19), (13, 21), (19, 21),
                (16, 20), (16, 22), (20, 22),
                
                # Hands
                (7, 23), (10, 24)
            ]
            
            center = 1  # Neck as center
            
        elif layout == 'ntu':
            # NTU RGB+D skeleton (25 joints)
            num_node = 25
            
            edge = [
                # Spine
                (0, 1), (1, 20), (20, 2), (2, 3),
                
                # Left arm
                (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
                
                # Right arm
                (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
                
                # Left leg
                (0, 12), (12, 13), (13, 14), (14, 15),
                
                # Right leg
                (0, 16), (16, 17), (17, 18), (18, 19)
            ]
            
            center = 1  # Spine base as center
            
        elif layout == 'coco':
            # COCO format (17 keypoints)
            num_node = 17
            
            edge = [
                # Face
                (0, 1), (0, 2), (1, 3), (2, 4),
                
                # Body
                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                (5, 11), (6, 12), (11, 12),
                
                # Legs
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            
            center = 0  # Nose as center
            
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        # Add self-loops
        self_link = [(i, i) for i in range(num_node)]
        edge = edge + self_link
        
        return num_node, edge, center
    
    def _get_hop_distance(self, num_node, edge, max_hop=1):
        """
        Compute hop distance matrix using Floyd-Warshall algorithm.
        
        Returns:
            hop_dis: (num_node, num_node) matrix of hop distances
        """
        # Initialize distance matrix
        hop_dis = np.full((num_node, num_node), np.inf)
        
        # Set edges to distance 1
        for i, j in edge:
            hop_dis[i][j] = 1
            hop_dis[j][i] = 1
        
        # Floyd-Warshall algorithm
        for k in range(num_node):
            for i in range(num_node):
                for j in range(num_node):
                    hop_dis[i][j] = min(hop_dis[i][j], hop_dis[i][k] + hop_dis[k][j])
        
        return hop_dis
    
    def _get_adjacency_matrix(self, num_node, hop_dis, strategy):
        """
        Get adjacency matrix with different partitioning strategies.
        
        Args:
            strategy: 'uniform', 'distance', or 'spatial'
        
        Returns:
            A: Adjacency tensor (K, num_node, num_node)
        """
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((num_node, num_node))
        
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        
        # Normalize adjacency matrix
        normalize_adjacency = self._normalize_digraph(adjacency)
        
        if strategy == 'uniform':
            # Uniform partition: single partition
            A = np.zeros((1, num_node, num_node))
            A[0] = normalize_adjacency
            
        elif strategy == 'distance':
            # Distance partitioning: partition by hop distance
            A = np.zeros((len(valid_hop), num_node, num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
                
        elif strategy == 'spatial':
            # Spatial partitioning: root, centripetal, centrifugal
            A = []
            for hop in valid_hop:
                a_root = np.zeros((num_node, num_node))
                a_close = np.zeros((num_node, num_node))
                a_further = np.zeros((num_node, num_node))
                
                for i in range(num_node):
                    for j in range(num_node):
                        if hop_dis[j][i] == hop:
                            if hop_dis[j][self.center] == hop_dis[i][self.center]:
                                a_root[j][i] = normalize_adjacency[j][i]
                            elif hop_dis[j][self.center] > hop_dis[i][self.center]:
                                a_close[j][i] = normalize_adjacency[j][i]
                            else:
                                a_further[j][i] = normalize_adjacency[j][i]
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            
            A = np.stack(A)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return A
    
    @staticmethod
    def _normalize_digraph(A):
        """Normalize adjacency matrix."""
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        
        AD = np.dot(A, Dn)
        return AD
    
    def get_adjacency_matrix(self):
        """Return adjacency matrix."""
        return self.A


# Predefined graph configurations
def get_graph(name='mediapipe'):
    """
    Get predefined graph configuration.
    
    Args:
        name: Graph name ('mediapipe', 'openpose', 'ntu', 'coco')
    
    Returns:
        Graph object
    """
    graph = Graph(layout=name, strategy='spatial', max_hop=1)
    return graph


if __name__ == "__main__":
    # Test different graph layouts
    layouts = ['mediapipe', 'openpose', 'ntu', 'coco']
    
    for layout in layouts:
        graph = Graph(layout=layout, strategy='spatial')
        print(f"\n{layout.upper()} Graph:")
        print(f"  Nodes: {graph.num_node}")
        print(f"  Edges: {len(graph.edge)}")
        print(f"  Adjacency shape: {graph.A.shape}")
        print(f"  Center node: {graph.center}")