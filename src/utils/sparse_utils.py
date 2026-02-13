"""
Utilities for converting dense masks to sparse edge indices.

This module provides functions to convert between dense adjacency matrices
(with -10^4 for masked entries) and sparse edge index format for memory-efficient
attention computation.
"""
import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


def mask_to_edge_index(
    mask: np.ndarray,
    mask_value: float = -10**4,
    return_edge_attr: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert dense mask to sparse edge indices.

    Args:
        mask: (n_query, n_key) numpy array
              Values of 0 = valid edge, -10^4 = masked edge
        mask_value: The value used for masked entries (default -10^4)
        return_edge_attr: If True, return edge attributes (bias values)

    Returns:
        edge_index: (2, n_edges) LongTensor of [query_idx, key_idx]
        edge_attr: (n_edges,) FloatTensor of edge biases (optional, None if all zeros)

    Example:
        >>> mask = np.array([[-10**4, 0, 0],
        ...                  [0, -10**4, 0],
        ...                  [0, 0, -10**4]])
        >>> edge_index, _ = mask_to_edge_index(mask)
        >>> print(edge_index)
        tensor([[0, 0, 1, 1, 2, 2],
                [1, 2, 0, 2, 0, 1]])
    """
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Find valid edges (not masked)
    valid = (mask != mask_value)
    query_indices, key_indices = np.where(valid)

    # Create edge index
    edge_index = torch.tensor(
        np.stack([query_indices, key_indices], axis=0),
        dtype=torch.long
    )

    # Optionally extract edge attributes
    edge_attr = None
    if return_edge_attr:
        edge_values = mask[valid]
        # Only return if non-zero (actual bias values)
        if not np.allclose(edge_values, 0.0):
            edge_attr = torch.tensor(edge_values, dtype=torch.float32)

    return edge_index, edge_attr


def edge_index_to_mask(
    edge_index: torch.Tensor,
    num_queries: int,
    num_keys: int,
    edge_attr: Optional[torch.Tensor] = None,
    mask_value: float = -10**4,
    edge_value: float = 0.0
) -> torch.Tensor:
    """
    Convert sparse edge indices back to dense mask (for testing/validation).

    Args:
        edge_index: (2, n_edges) edge indices
        num_queries: Number of query nodes
        num_keys: Number of key nodes
        edge_attr: Optional edge attributes
        mask_value: Value for masked entries (default -10^4)
        edge_value: Value for valid edges if edge_attr not provided (default 0)

    Returns:
        mask: (num_queries, num_keys) dense mask

    Example:
        >>> edge_index = torch.tensor([[0, 1], [1, 2]])
        >>> mask = edge_index_to_mask(edge_index, 3, 4)
        >>> print(mask.shape)
        torch.Size([3, 4])
    """
    mask = torch.full(
        (num_queries, num_keys),
        mask_value,
        dtype=torch.float32,
        device=edge_index.device
    )

    query_idx = edge_index[0]
    key_idx = edge_index[1]

    if edge_attr is not None:
        mask[query_idx, key_idx] = edge_attr
    else:
        mask[query_idx, key_idx] = edge_value

    return mask


def preprocess_hierarchical_masks_to_edges(
    hierarchical_masks: List[Dict],
    interaction_types: List[str]
) -> List[Dict]:
    """
    Convert nested hierarchical masks to edge format.

    This processes the output of tree_parser.get_hierarchical_interactions()
    and adds 'edge_index' and 'edge_attr' fields.

    Args:
        hierarchical_masks: List of dicts from get_hierarchical_interactions()
                           Each dict has keys for interaction types
        interaction_types: List of interaction type names

    Returns:
        List of dicts with added 'edge_index' and 'edge_attr' fields

    Example:
        >>> masks = tree_parser.get_hierarchical_interactions(['is_a'], format='indices')
        >>> edges = preprocess_hierarchical_masks_to_edges(masks, ['is_a'])
    """
    result = []

    for level_masks in hierarchical_masks:
        level_edges = {}

        for inter_type, mask_data in level_masks.items():
            # mask_data has: 'query', 'key', 'query_indices', 'key_indices', 'mask'
            mask = mask_data['mask']  # (n_q_padded, n_k_padded) tensor

            # Convert to edge format
            edge_index, edge_attr = mask_to_edge_index(
                mask.numpy() if isinstance(mask, torch.Tensor) else mask,
                return_edge_attr=True
            )

            # Add to output
            level_edges[inter_type] = {
                **mask_data,  # Keep original fields
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'n_queries': len(mask_data['query_indices']),
                'n_keys': len(mask_data['key_indices'])
            }

        if level_edges:
            result.append(level_edges)

    return result


def batch_edge_indices(
    edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int
) -> torch.Tensor:
    """
    Expand edge indices for batch processing.

    This replicates edge indices across batch dimension with appropriate offsets.

    Args:
        edge_index: (2, n_edges) single-sample edge indices
        batch_size: Number of samples in batch
        num_nodes: Number of nodes (for offset calculation)

    Returns:
        batched_edge_index: (2, batch_size * n_edges)

    Example:
        >>> edge_index = torch.tensor([[0, 1], [1, 2]])  # 2 edges
        >>> batched = batch_edge_indices(edge_index, batch_size=3, num_nodes=5)
        >>> print(batched)
        tensor([[ 0,  1,  5,  6, 10, 11],
                [ 1,  2,  6,  7, 11, 12]])
    """
    n_edges = edge_index.shape[1]
    device = edge_index.device

    # Create batch offsets: [0, num_nodes, 2*num_nodes, ...]
    batch_offsets = torch.arange(
        batch_size,
        device=device,
        dtype=edge_index.dtype
    ) * num_nodes

    # Repeat edge_index for each batch
    batched_edges = edge_index.unsqueeze(1).repeat(1, batch_size, 1)
    batched_edges = batched_edges.reshape(2, batch_size * n_edges)

    # Add offsets
    offsets_expanded = batch_offsets.repeat_interleave(n_edges)
    batched_edges = batched_edges + offsets_expanded.unsqueeze(0)

    return batched_edges


def compute_sparsity_stats(
    mask: np.ndarray,
    mask_value: float = -10**4
) -> Dict[str, float]:
    """
    Compute sparsity statistics for a mask.

    Args:
        mask: (n_query, n_key) numpy array
        mask_value: Value used for masked entries

    Returns:
        Dict with keys:
            - 'total_entries': Total number of possible edges
            - 'valid_edges': Number of valid (non-masked) edges
            - 'sparsity_pct': Percentage of valid edges
            - 'compression_ratio': How much smaller sparse format is

    Example:
        >>> mask = np.full((1000, 2000), -10**4)
        >>> mask[:100, :3] = 0  # 300 valid edges
        >>> stats = compute_sparsity_stats(mask)
        >>> print(f"Sparsity: {stats['sparsity_pct']:.3f}%")
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    valid = (mask != mask_value)
    n_edges = np.sum(valid)
    total = mask.size

    # Sparse format: 2 ints per edge (query_idx, key_idx)
    # Dense format: 1 float per entry
    sparse_size = n_edges * 2 * 4  # 2 indices * 4 bytes
    dense_size = total * 4  # 4 bytes per float32

    return {
        'total_entries': int(total),
        'valid_edges': int(n_edges),
        'sparsity_pct': 100.0 * n_edges / total if total > 0 else 0.0,
        'compression_ratio': dense_size / sparse_size if sparse_size > 0 else float('inf'),
        'memory_saved_mb': (dense_size - sparse_size) / (1024**2)
    }


def validate_edge_index(
    edge_index: torch.Tensor,
    num_queries: int,
    num_keys: int,
    raise_on_error: bool = True
) -> bool:
    """
    Validate that edge indices are well-formed.

    Args:
        edge_index: (2, n_edges) edge indices to validate
        num_queries: Expected number of query nodes
        num_keys: Expected number of key nodes
        raise_on_error: If True, raise exception on validation failure

    Returns:
        True if valid, False otherwise (if raise_on_error=False)

    Raises:
        ValueError: If validation fails and raise_on_error=True
    """
    if edge_index.shape[0] != 2:
        msg = f"edge_index must have shape (2, n_edges), got {edge_index.shape}"
        if raise_on_error:
            raise ValueError(msg)
        return False

    query_idx = edge_index[0]
    key_idx = edge_index[1]

    # Check bounds
    if torch.any(query_idx < 0) or torch.any(query_idx >= num_queries):
        msg = f"Query indices out of bounds [0, {num_queries})"
        if raise_on_error:
            raise ValueError(msg)
        return False

    if torch.any(key_idx < 0) or torch.any(key_idx >= num_keys):
        msg = f"Key indices out of bounds [0, {num_keys})"
        if raise_on_error:
            raise ValueError(msg)
        return False

    return True


def print_sparsity_report(name: str, mask: np.ndarray, mask_value: float = -10**4):
    """
    Print a formatted sparsity report for a mask.

    Args:
        name: Name of the mask (e.g., "snp2gene")
        mask: The mask to analyze
        mask_value: Value used for masked entries
    """
    stats = compute_sparsity_stats(mask, mask_value)

    print(f"\n{name} Sparsity Report:")
    print(f"  Shape: {mask.shape}")
    print(f"  Valid edges: {stats['valid_edges']:,} / {stats['total_entries']:,}")
    print(f"  Sparsity: {stats['sparsity_pct']:.4f}%")
    print(f"  Compression: {stats['compression_ratio']:.1f}×")
    print(f"  Memory saved: {stats['memory_saved_mb']:.1f} MB")
