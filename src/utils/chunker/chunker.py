import copy
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

        # Add edges where mask is non-zero (mask edge = 0, without edge = -10^4)
        # So we look for mask == 0 (which means there IS an edge)
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
            # For gene2sys_mask: rows are systems, cols are genes
            chunk_spec = {
                'snp_indices': sorted(list(chunk_snps)),
                'gene_indices': sorted(list(chunk_genes)),
                'system_indices': sorted(list(chunk_systems)),
                'snp2gene_submask': self._extract_submask(
                    self.snp2gene_mask, chunk_genes, chunk_snps  # rows=genes, cols=snps
                ),
                'gene2sys_submask': self._extract_submask(
                    self.gene2sys_mask, chunk_systems, chunk_genes  # rows=systems, cols=genes
                )
            }

            extended_chunks.append(chunk_spec)

        return extended_chunks

    def _find_connected_systems(self, gene_indices):
        """Find systems connected to the given genes"""
        connected_systems = set()

        for gene_idx in gene_indices:
            # Find systems connected to this gene (mask value 0 means connected)
            system_connections = np.where(self.gene2sys_mask[:, gene_idx] == 0)[0]
            connected_systems.update(system_connections.tolist())

        return connected_systems

    def _extract_submask(self, full_mask, row_indices, col_indices):
        """Extract submask for the given indices"""
        if len(row_indices) == 0 or len(col_indices) == 0:
            # Return empty mask with appropriate shape
            return torch.full((len(row_indices), len(col_indices)), -1e4, dtype=torch.float32)

        row_indices = np.array(sorted(row_indices), dtype=np.int64)
        col_indices = np.array(sorted(col_indices), dtype=np.int64)

        # Advanced indexing to extract submask
        submask = full_mask[row_indices][:, col_indices]
        submask = torch.tensor(submask, dtype=torch.float32)
        return submask


class AttentionAwareChunker(MaskBasedChunker):
    def __init__(self, snp2gene_mask, gene2sys_mask, target_attention_ops=1e6,
                 memory_constraint=None, balance_threshold=0.15, target_chunk_size=512):
        """
        Args:
            target_attention_ops: Target number of attention operations per chunk
            memory_constraint: GPU memory constraint in GB
            balance_threshold: Allowed imbalance between chunks (15% default)
        """
        super(AttentionAwareChunker, self).__init__(snp2gene_mask, gene2sys_mask, target_chunk_size=target_chunk_size)
        self.target_attention_ops = target_attention_ops
        self.balance_threshold = balance_threshold

        if memory_constraint:
            # Convert memory constraint to attention ops
            # Rough estimate: 4 bytes per float, batch_size factor, etc.
            self.target_attention_ops = self._memory_to_attention_ops(memory_constraint)

    def _memory_to_attention_ops(self, memory_gb, batch_size=32, bytes_per_param=4):
        """Convert memory constraint to approximate attention operations"""
        # Memory for attention: batch_size × heads × seq_len² × bytes_per_param
        # Assuming 8 heads, fp32 (4 bytes)
        available_bytes = memory_gb * 1e9 * 0.7  # 70% for attention
        max_seq_len_squared = available_bytes / (batch_size * 8 * bytes_per_param)
        return int(max_seq_len_squared)

    def calculate_attention_complexity(self, snp_indices, gene_indices, system_indices):
        """Calculate total attention operations for a chunk"""
        # SNP2Gene attention: genes × snps
        snp2gene_ops = len(gene_indices) * len(snp_indices)

        # Gene2System attention: systems × genes
        gene2sys_ops = len(system_indices) * len(gene_indices)

        # Cross-attention between chunks (estimated)
        cross_chunk_ops = max(len(gene_indices), len(system_indices)) * 64  # Assuming 64 global tokens

        total_ops = snp2gene_ops + gene2sys_ops + cross_chunk_ops

        return {
            'total': total_ops,
            'snp2gene': snp2gene_ops,
            'gene2sys': gene2sys_ops,
            'cross_chunk': cross_chunk_ops,
            'memory_estimate_mb': self._estimate_memory_usage(snp_indices, gene_indices, system_indices)
        }

    def _estimate_memory_usage(self, snp_indices, gene_indices, system_indices, batch_size=32):
        """Estimate actual memory usage in MB"""
        # Attention matrices memory
        snp2gene_attn = len(gene_indices) * len(snp_indices) * batch_size * 4  # fp32
        gene2sys_attn = len(system_indices) * len(gene_indices) * batch_size * 4

        # Embedding memory
        embedding_memory = (len(snp_indices) + len(gene_indices) + len(system_indices)) * 256 * 4  # hidden_size=256

        # Gradient memory (roughly 2x forward pass)
        total_bytes = (snp2gene_attn + gene2sys_attn + embedding_memory) * 2

        return total_bytes / (1024 * 1024)  # Convert to MB

    def create_chunks(self):
        """Create chunks balanced by attention complexity"""
        # Step 1: Analyze connected components
        components = self._find_connected_components()
        component_complexities = self._analyze_component_complexities(components)

        # Step 2: Use bin packing with attention complexity
        chunks = self._attention_aware_bin_packing(component_complexities)

        # Step 3: Optimize chunk boundaries
        optimized_chunks = self._optimize_chunk_boundaries(chunks)

        # Step 4: Validate and balance
        final_chunks = self._final_balancing(optimized_chunks)

        return final_chunks

    def _analyze_component_complexities(self, components):
        """Analyze attention complexity for each connected component"""
        component_info = []

        for comp in components:
            snps = [int(node.split('_')[1]) for node in comp if node.startswith('col_')]
            genes = [int(node.split('_')[1]) for node in comp if node.startswith('row_')]

            # Find connected systems
            systems = self._find_connected_systems(genes)

            # Calculate complexity
            complexity = self.calculate_attention_complexity(snps, genes, systems)

            component_info.append({
                'component': comp,
                'snp_indices': snps,
                'gene_indices': genes,
                'system_indices': systems,
                'complexity': complexity,
                'priority': self._calculate_priority(snps, genes, systems)
            })

        return component_info

    def _calculate_priority(self, snps, genes, systems):
        """Calculate component priority for chunk assignment"""
        # Prioritize components with:
        # 1. High connectivity (many edges)
        # 2. Balanced SNP/gene ratio
        # 3. Important biological systems (optional: can use ontology weights)

        connectivity = self._count_edges_in_component(snps, genes, systems)
        balance_score = min(len(snps), len(genes)) / max(len(snps), len(genes), 1)

        # Could add biological importance here
        # importance = sum(system_weights.get(sys, 1.0) for sys in systems)

        return connectivity * balance_score

    def _count_edges_in_component(self, snps, genes, systems):
        """Count the number of edges within a component"""
        edge_count = 0

        # Count SNP-Gene edges
        for gene_idx in genes:
            for snp_idx in snps:
                if gene_idx < self.snp2gene_mask.shape[0] and snp_idx < self.snp2gene_mask.shape[1]:
                    if self.snp2gene_mask[gene_idx, snp_idx] == 0:  # 0 means edge exists
                        edge_count += 1

        # Count Gene-System edges
        for system_idx in systems:
            for gene_idx in genes:
                if system_idx < self.gene2sys_mask.shape[0] and gene_idx < self.gene2sys_mask.shape[1]:
                    if self.gene2sys_mask[system_idx, gene_idx] == 0:  # 0 means edge exists
                        edge_count += 1

        return edge_count

    def _attention_aware_bin_packing(self, component_complexities):
        """Bin packing optimized for attention complexity balance"""

        # Sort components by complexity (largest first for better packing)
        sorted_components = sorted(
            component_complexities,
            key=lambda x: x['complexity']['total'],
            reverse=True
        )

        chunks = []
        chunk_complexities = []

        for comp_info in sorted_components:
            comp_complexity = comp_info['complexity']['total']

            # Find best chunk for this component
            best_chunk_idx = self._find_best_attention_chunk(
                chunks, chunk_complexities, comp_info
            )

            if best_chunk_idx is None:
                # Create new chunk
                chunks.append([comp_info])
                chunk_complexities.append(comp_complexity)
            else:
                # Add to existing chunk
                chunks[best_chunk_idx].append(comp_info)
                chunk_complexities[best_chunk_idx] += comp_complexity

        return chunks

    def _find_best_attention_chunk(self, chunks, chunk_complexities, comp_info):
        """Find the best chunk considering attention complexity"""
        comp_complexity = comp_info['complexity']['total']
        best_idx = None
        best_score = float('inf')

        for i, current_complexity in enumerate(chunk_complexities):
            new_complexity = current_complexity + comp_complexity

            # Check if it fits within target
            if new_complexity <= self.target_attention_ops * (1 + self.balance_threshold):
                # Calculate multi-objective score
                score = self._calculate_chunk_assignment_score(
                    chunks[i], comp_info, new_complexity
                )

                if score < best_score:
                    best_score = score
                    best_idx = i

        return best_idx

    def _calculate_chunk_assignment_score(self, existing_chunk, new_comp, total_complexity):
        """Multi-objective score for chunk assignment"""
        # 1. Memory efficiency score
        memory_efficiency = abs(total_complexity - self.target_attention_ops) / self.target_attention_ops

        # 2. Balance between snp2gene and gene2sys operations
        total_snp2gene = sum(comp['complexity']['snp2gene'] for comp in existing_chunk)
        total_gene2sys = sum(comp['complexity']['gene2sys'] for comp in existing_chunk)
        total_snp2gene += new_comp['complexity']['snp2gene']
        total_gene2sys += new_comp['complexity']['gene2sys']

        balance_score = abs(total_snp2gene - total_gene2sys) / max(total_snp2gene, total_gene2sys, 1)

        # 3. Connectivity penalty (prefer keeping connected components together)
        connectivity_penalty = self._calculate_connectivity_penalty(existing_chunk, new_comp)

        # Weighted combination
        return (0.5 * memory_efficiency +
                0.3 * balance_score +
                0.2 * connectivity_penalty)

    def _calculate_connectivity_penalty(self, existing_chunk, new_comp):
        """Calculate penalty for breaking connectivity between components"""
        penalty = 0.0

        # Check connectivity between new component and existing components
        new_genes = set(new_comp['gene_indices'])
        new_snps = set(new_comp['snp_indices'])
        new_systems = set(new_comp['system_indices'])

        for existing_comp in existing_chunk:
            existing_genes = set(existing_comp['gene_indices'])
            existing_snps = set(existing_comp['snp_indices'])
            existing_systems = set(existing_comp['system_indices'])

            # Count potential cross-component connections
            gene_overlap = len(new_genes & existing_genes)
            snp_overlap = len(new_snps & existing_snps)
            system_overlap = len(new_systems & existing_systems)

            # Penalty increases with overlap (indicates strong connectivity)
            penalty += (gene_overlap + snp_overlap + system_overlap) * 0.1

        return penalty

    def _optimize_chunk_boundaries(self, initial_chunks):
        """Optimize chunk boundaries by moving components between chunks"""
        optimized_chunks = copy.deepcopy(initial_chunks)
        improved = True
        iteration = 0
        max_iterations = 10

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(len(optimized_chunks)):
                for j in range(len(optimized_chunks)):
                    if i == j:
                        continue

                    # Try moving components from chunk i to chunk j
                    if self._try_component_move(optimized_chunks, i, j):
                        improved = True

            # Try merging small chunks
            if self._try_chunk_merging(optimized_chunks):
                improved = True

            # Try splitting large chunks
            if self._try_chunk_splitting(optimized_chunks):
                improved = True

        return optimized_chunks

    def _try_component_move(self, chunks, from_idx, to_idx):
        """Try moving a component from one chunk to another"""
        if not chunks[from_idx]:  # Empty chunk
            return False

        # Calculate current complexities
        from_complexity = sum(comp['complexity']['total'] for comp in chunks[from_idx])
        to_complexity = sum(comp['complexity']['total'] for comp in chunks[to_idx])

        # Try moving each component
        best_improvement = 0
        best_component = None

        for comp in chunks[from_idx]:
            comp_complexity = comp['complexity']['total']

            # New complexities after move
            new_from = from_complexity - comp_complexity
            new_to = to_complexity + comp_complexity

            # Check if move is valid
            if (new_to <= self.target_attention_ops * (1 + self.balance_threshold) and
                    new_from >= 0):

                # Calculate improvement in balance
                old_imbalance = abs(from_complexity - to_complexity)
                new_imbalance = abs(new_from - new_to)
                improvement = old_imbalance - new_imbalance

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_component = comp

        # Perform the best move if beneficial
        if best_component and best_improvement > 0:
            chunks[from_idx].remove(best_component)
            chunks[to_idx].append(best_component)
            return True

        return False

    def _try_chunk_merging(self, chunks):
        """Try merging small chunks to improve efficiency"""
        improved = False
        i = 0

        while i < len(chunks):
            if not chunks[i]:  # Remove empty chunks
                chunks.pop(i)
                improved = True
                continue

            chunk_complexity = sum(comp['complexity']['total'] for comp in chunks[i])

            # If chunk is small, try to merge with another small chunk
            if chunk_complexity < self.target_attention_ops * 0.5:
                best_merge_idx = None
                best_combined_complexity = float('inf')

                for j in range(i + 1, len(chunks)):
                    if not chunks[j]:
                        continue

                    other_complexity = sum(comp['complexity']['total'] for comp in chunks[j])
                    combined_complexity = chunk_complexity + other_complexity

                    # Check if merge is valid and beneficial
                    if (combined_complexity <= self.target_attention_ops * (1 + self.balance_threshold) and
                            combined_complexity < best_combined_complexity):
                        best_combined_complexity = combined_complexity
                        best_merge_idx = j

                # Perform merge if beneficial
                if best_merge_idx is not None:
                    chunks[i].extend(chunks[best_merge_idx])
                    chunks.pop(best_merge_idx)
                    improved = True
                    continue

            i += 1

        return improved

    def _try_chunk_splitting(self, chunks):
        """Try splitting large chunks to improve balance"""
        improved = False

        for i in range(len(chunks)):
            if not chunks[i]:
                continue

            chunk_complexity = sum(comp['complexity']['total'] for comp in chunks[i])

            # If chunk is too large, try to split it
            if chunk_complexity > self.target_attention_ops * (1 + self.balance_threshold):
                split_result = self._split_large_chunk(chunks[i])

                if len(split_result) > 1:
                    # Replace original chunk with split chunks
                    chunks[i] = split_result[0]
                    chunks.extend(split_result[1:])
                    improved = True

        return improved

    def _split_large_chunk(self, large_chunk):
        """Split a large chunk into smaller balanced chunks"""
        if len(large_chunk) <= 1:
            return [large_chunk]

        # Sort components by complexity
        sorted_comps = sorted(large_chunk, key=lambda x: x['complexity']['total'], reverse=True)

        # Use bin packing to create two balanced sub-chunks
        chunk1 = []
        chunk2 = []
        complexity1 = 0
        complexity2 = 0

        for comp in sorted_comps:
            comp_complexity = comp['complexity']['total']

            # Add to the chunk with less complexity
            if complexity1 <= complexity2:
                chunk1.append(comp)
                complexity1 += comp_complexity
            else:
                chunk2.append(comp)
                complexity2 += comp_complexity

        # Only split if both resulting chunks are valid
        if (complexity1 <= self.target_attention_ops * (1 + self.balance_threshold) and
                complexity2 <= self.target_attention_ops * (1 + self.balance_threshold) and
                len(chunk1) > 0 and len(chunk2) > 0):
            return [chunk1, chunk2]
        else:
            return [large_chunk]

    def _final_balancing(self, chunks):
        """Final validation and balancing of chunks"""
        final_chunks = []

        for chunk_components in chunks:
            if not chunk_components:  # Skip empty chunks
                continue

            # Extract all indices from components
            all_snps = set()
            all_genes = set()
            all_systems = set()

            for comp in chunk_components:
                all_snps.update(comp['snp_indices'])
                all_genes.update(comp['gene_indices'])
                all_systems.update(comp['system_indices'])

            # Create final chunk specification
            # For gene2sys_mask: rows are systems, cols are genes
            chunk_spec = {
                'snp_indices': sorted(list(all_snps)),
                'gene_indices': sorted(list(all_genes)),
                'system_indices': sorted(list(all_systems)),
                'snp2gene_submask': self._extract_submask(
                    self.snp2gene_mask, all_genes, all_snps  # rows=genes, cols=snps
                ),
                'gene2sys_submask': self._extract_submask(
                    self.gene2sys_mask, all_systems, all_genes  # rows=systems, cols=genes
                ),
                'complexity': self.calculate_attention_complexity(
                    all_snps, all_genes, all_systems
                )
            }

            final_chunks.append(chunk_spec)

        return final_chunks