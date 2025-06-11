import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
import torch


class MaskBasedChunker:
    def __init__(self, snp2gene_mask, gene2sys_mask, target_chunk_size=512, balance_factor=0.8):
        """
        Args:
            snp2gene_mask: (n_genes, n_snps) sparse mask
            gene2sys_mask: (n_systems, n_genes) sparse mask
            target_chunk_size: Target number of elements per chunk
            balance_factor: Trade-off between chunk size and connectivity
        """
        self.snp2gene_mask = snp2gene_mask
        self.gene2sys_mask = gene2sys_mask
        self.target_chunk_size = target_chunk_size
        self.balance_factor = balance_factor

        # Build connectivity graphs
        self.snp_gene_graph = self._build_bipartite_graph(snp2gene_mask)
        self.gene_sys_graph = self._build_bipartite_graph(gene2sys_mask)

    def _build_bipartite_graph(self, mask):
        """Convert sparse mask to networkx bipartite graph"""
        G = nx.Graph()

        # Add edges where mask is non-zero
        rows, cols = np.where(mask == 0)
        edges = [(f"row_{r.item()}", f"col_{c.item()}") for r, c in zip(rows, cols)]
        G.add_edges_from(edges)

        return G

    def create_chunks(self):
        """Create chunks that respect biological connectivity"""
        # Step 1: Find connected components in SNP-Gene space
        snp_gene_components = self._find_connected_components()

        # Step 2: Group components into balanced chunks
        chunks = self._balance_components(snp_gene_components)

        # Step 3: Extend chunks to include relevant systems
        extended_chunks = self._extend_with_systems(chunks)

        return extended_chunks

    def _find_connected_components(self):
        """Find connected components respecting biological constraints"""
        # Get connected components from SNP-Gene graph
        components = list(nx.connected_components(self.snp_gene_graph))

        # For large components, use spectral clustering to sub-divide
        refined_components = []

        for component in components:
            if len(component) > self.target_chunk_size * 2:
                # Sub-cluster large components
                subgraph = self.snp_gene_graph.subgraph(component)
                sub_components = self._spectral_cluster_component(subgraph)
                refined_components.extend(sub_components)
            else:
                refined_components.append(component)

        return refined_components

    def _spectral_cluster_component(self, subgraph):
        """Use spectral clustering for large connected components"""
        # Convert to adjacency matrix
        nodes = list(subgraph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        adj_matrix = nx.adjacency_matrix(subgraph, nodelist=nodes)

        # Determine number of clusters
        n_nodes = len(nodes)
        n_clusters = max(2, n_nodes // self.target_chunk_size)

        # Apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        labels = clustering.fit_predict(adj_matrix.toarray())

        # Group nodes by cluster
        clusters = [[] for _ in range(n_clusters)]
        for node, label in zip(nodes, labels):
            clusters[label].append(node)

        return [set(cluster) for cluster in clusters if cluster]

    def _balance_components(self, components):
        """Balance components into chunks of similar computational cost"""

        # Calculate computational cost for each component
        component_costs = []
        for comp in components:
            snps = [node for node in comp if node.startswith('col_')]  # SNPs
            genes = [node for node in comp if node.startswith('row_')]  # Genes

            # Cost = SNPs × Genes (attention computation cost)
            cost = len(snps) * len(genes)
            component_costs.append((comp, cost, len(snps), len(genes)))

        # Sort by cost (largest first for bin packing)
        component_costs.sort(key=lambda x: x[1], reverse=True)

        # Bin packing algorithm to balance chunks
        chunks = []
        chunk_costs = []
        target_cost = sum(cost for _, cost, _, _ in component_costs) / \
                      (len(component_costs) // (self.target_chunk_size // 50) + 1)

        for comp, cost, n_snps, n_genes in component_costs:
            # Find best chunk to add this component
            best_chunk_idx = self._find_best_chunk(chunks, chunk_costs, cost, target_cost)

            if best_chunk_idx is None:
                # Create new chunk
                chunks.append([comp])
                chunk_costs.append(cost)
            else:
                # Add to existing chunk
                chunks[best_chunk_idx].append(comp)
                chunk_costs[best_chunk_idx] += cost

        return chunks

    def _find_best_chunk(self, chunks, chunk_costs, component_cost, target_cost):
        """Find the best chunk to add a component using bin packing heuristic"""
        best_idx = None
        best_waste = float('inf')

        for i, current_cost in enumerate(chunk_costs):
            new_cost = current_cost + component_cost

            # Check if it fits and calculate waste
            if new_cost <= target_cost * 1.2:  # Allow 20% overshoot
                waste = target_cost - new_cost
                if waste < best_waste and waste >= 0:
                    best_waste = waste
                    best_idx = i

        return best_idx

    def _extend_with_systems(self, snp_gene_chunks):
        """Extend chunks to include relevant biological systems"""
        extended_chunks = []

        for chunk_components in snp_gene_chunks:
            # Extract genes from this chunk
            chunk_genes = set()
            chunk_snps = set()

            for component in chunk_components:
                for node in component:
                    if node.startswith('row_'):  # Gene
                        gene_idx = int(node.split('_')[1])
                        chunk_genes.add(gene_idx)
                    elif node.startswith('col_'):  # SNP
                        snp_idx = int(node.split('_')[1])
                        chunk_snps.add(snp_idx)

            # Find systems connected to these genes
            chunk_systems = self._find_connected_systems(chunk_genes)

            # Create final chunk specification
            chunk_spec = {
                'snp_indices': sorted(list(chunk_snps)),
                'gene_indices': sorted(list(chunk_genes)),
                'system_indices': sorted(list(chunk_systems)),
                'snp2gene_submask': self._extract_submask(
                    self.snp2gene_mask, chunk_genes, chunk_snps
                ),
                'gene2sys_submask': self._extract_submask(
                    self.gene2sys_mask, chunk_systems, chunk_genes
                )
            }

            extended_chunks.append(chunk_spec)

        return extended_chunks

    def _find_connected_systems(self, gene_indices):
        """Find systems connected to the given genes"""
        connected_systems = set()

        for gene_idx in gene_indices:
            # Find systems connected to this gene
            system_connections = np.where(self.gene2sys_mask[:, gene_idx] != 0)[0]
            connected_systems.update(system_connections.tolist())

        return connected_systems

    def _extract_submask(self, full_mask, row_indices, col_indices):
        """Extract submask for the given indices"""
        row_indices = np.array(sorted(row_indices), dtype=np.int64)
        col_indices = np.array(sorted(col_indices), dtype=np.int64)

        # Advanced indexing to extract submask
        submask = full_mask[row_indices][:, col_indices]
        submask = torch.tensor(submask, dtype=torch.float32)
        return submask