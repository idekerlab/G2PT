import numpy as np
import pandas as pd
import networkx as nx

def build_hierarchical_ontology(sim,
                                n_genes=800,
                                n_systems=60, # This is the number of leaf systems
                                p_shared_system=0.8,
                                p_epistatic_parent_child=0.5,
                                overlap_prob=0.10,
                                seed=123):
    """
    Construct SNP→Gene→System→...→Root hierarchy.
    Injects epistatic interactions between parent and child systems.
    Returns three DataFrames ready for .to_csv().
    """
    rng = np.random.default_rng(seed)
    n_snps = sim["G"].shape[1]

    # ----------------- 1. IDs -----------------
    gene_ids    = [f"G{g:04d}" for g in range(n_genes)]
    system_ids  = [f"S{p:03d}" for p in range(n_systems)] # Leaf systems

    # ----------------- 2. System Hierarchy Construction -----------------
    all_system_nodes = list(system_ids)
    parent_child_edges = []
    
    current_level_nodes = list(system_ids)
    level = 0
    # Create parent layers until a single root is formed
    while len(current_level_nodes) > 1:
        level += 1
        # Reduce by a factor of ~2 each level
        n_parents = int(np.ceil(len(current_level_nodes) / 2.0)) 
        parents = [f"L{level}_{i}" for i in range(n_parents)]
        all_system_nodes.extend(parents)
        
        # Assign each node in the current level to a parent in the new level
        for i, child_node in enumerate(current_level_nodes):
            parent_node = parents[i // 2]
            parent_child_edges.append((parent_node, child_node))
            
        current_level_nodes = parents

    sys_graph = nx.DiGraph()
    sys_graph.add_edges_from(parent_child_edges)

    # ----------------- 3. Gene → Leaf System Mapping -----------------
    gene2system = {g: set() for g in gene_ids}
    # Initially, assign each gene to a random leaf system
    for g in gene_ids:
        gene2system[g].add(rng.choice(system_ids))

    # ----------------- 4. SNP → Gene Mapping with Epistasis Logic -----------------
    snp2gene = rng.choice(gene_ids, size=n_snps)
    epistatic_pairs = sim["pair_idx"]
    n_epistatic_pairs = len(epistatic_pairs)
    n_parent_child_epistasis = int(n_epistatic_pairs * p_epistatic_parent_child)

    if not parent_child_edges:
        print("Warning: No parent-child system relationships. Cannot inject parent-child epistasis.")
        n_parent_child_epistasis = 0

    # Inject epistasis between parent-child systems
    for i in range(n_parent_child_epistasis):
        snp_i, snp_j = epistatic_pairs[i]
        parent, child = rng.choice(parent_child_edges)

        # Get all leaf systems under parent and child
        parent_leaves = [n for n in nx.descendants(sys_graph, parent) if sys_graph.out_degree(n) == 0]
        if not parent_leaves: parent_leaves = [parent] if sys_graph.out_degree(parent) == 0 else []
        
        child_leaves = [n for n in nx.descendants(sys_graph, child) if sys_graph.out_degree(n) == 0]
        if not child_leaves: child_leaves = [child] if sys_graph.out_degree(child) == 0 else []

        if not parent_leaves or not child_leaves: continue # Should not happen if graph is built correctly

        # Pick a random leaf system from each subtree
        sys_for_i = rng.choice(parent_leaves)
        sys_for_j = rng.choice(child_leaves)

        # Find genes in those leaf systems
        genes_in_sys_i = [g for g, systems in gene2system.items() if sys_for_i in systems]
        genes_in_sys_j = [g for g, systems in gene2system.items() if sys_for_j in systems]
        if not genes_in_sys_i: genes_in_sys_i = [rng.choice(gene_ids)] # Failsafe
        if not genes_in_sys_j: genes_in_sys_j = [rng.choice(gene_ids)] # Failsafe

        # Assign SNPs to genes in those systems
        g_i = rng.choice(genes_in_sys_i)
        g_j = rng.choice(genes_in_sys_j)
        snp2gene[snp_i] = g_i
        snp2gene[snp_j] = g_j

    # Assign remaining epistatic pairs randomly to leaf systems
    for i in range(n_parent_child_epistasis, n_epistatic_pairs):
        snp_i, snp_j = epistatic_pairs[i]
        g_i = rng.choice(gene_ids)
        g_j = g_i if rng.random() < p_shared_system else rng.choice(gene_ids)
        snp2gene[snp_i] = g_i
        snp2gene[snp_j] = g_j
        # Also ensure these genes are in at least one system
        sys_for_pair = rng.choice(system_ids)
        gene2system[g_i].add(sys_for_pair)
        gene2system[g_j].add(sys_for_pair)

    # ----------------- 5. Finalize Gene -> System and add noise -----------------
    for sys in system_ids: # Noise only added to leaf systems
        extras = rng.choice(gene_ids, size=4, replace=False)
        for g in extras:
            if rng.random() < overlap_prob:
                gene2system[g].add(sys)
    
    # Ensure all genes are mapped to at least one system
    for gene in gene_ids:
        if not gene2system[gene]:
            gene2system[gene].add(rng.choice(system_ids))

    # --------------- 6. Flatten to tables ---------------
    snp_chromosomes = rng.integers(1, 23, size=n_snps)
    snp_positions = rng.integers(1, 10_000_000, size=n_snps)
    snp_blocks = rng.integers(1, 5, size=n_snps)
    snp_df = pd.DataFrame({
        "snp": np.arange(n_snps),
        "chr": snp_chromosomes,
        "pos": snp_positions,
        "block": snp_blocks,
        "gene": snp2gene
    })

    gene_df = pd.DataFrame([(g, s, 1.0)
                            for g, ss in gene2system.items() for s in ss],
                           columns=["gene_id", "system_id", "weight"])

    # system_df now contains the full hierarchy
    system_df = pd.DataFrame(parent_child_edges, columns=["supersystem_id", "system_id"])
    system_df["weight"] = 1.0

    return snp_df, gene_df, system_df


def simulate_epistasis(
        n_samples=1000,          # individuals
        n_snps=20_000,           # total SNPs
        n_additive=50,           # causal additive SNPs
        n_pairs=20,              # causal epistatic pairs
        h2_additive=0.30,        # additive heritability
        h2_epistatic=0.10,       # epistatic heritability
        maf_range=(0.05, 0.5),   # minor-allele-frequency window
        seed=42):
    """
    Return
    ------
    dict with
      'G'            int8   (n_samples, n_snps)  – genotype dosages 0/1/2
      'y'            float  (n_samples,)        – phenotype
      'additive_idx' list[int]                  – causal SNP indices
      'pair_idx'     list[(int,int)]            – causal SNP pairs
      'beta'         np.ndarray                 – additive effect sizes
      'gamma'        np.ndarray                 – pairwise effect sizes
    """
    rng = np.random.default_rng(seed)

    # 1. simulate genotypes with real-like MAF spectrum
    mafs = rng.uniform(*maf_range, size=n_snps)
    G = rng.binomial(2, mafs, size=(n_samples, n_snps)).astype(np.float32)

    # z-score each SNP (mean-centred, unit variance)
    G_std = (G - G.mean(0)) / G.std(0)

    # 2. pick causal SNPs / pairs
    additive_idx = rng.choice(n_snps, size=n_additive, replace=False)
    remaining = np.setdiff1d(np.arange(n_snps), additive_idx)
    pairs = rng.choice(remaining, size=2 * n_pairs, replace=False).reshape(n_pairs, 2)

    # 3. draw effect sizes and scale to target heritability
    beta  = rng.standard_normal(n_additive) * np.sqrt(h2_additive / n_additive)
    gamma = rng.standard_normal(n_pairs)    * np.sqrt(h2_epistatic / n_pairs)

    # 4. build phenotype: additive + pairwise + environmental noise
    y  = G_std[:, additive_idx] @ beta
    for k, (i, j) in enumerate(pairs):
        y += gamma[k] * G_std[:, i] * G_std[:, j]

    # add noise so Var(y) ≈ 1
    y_var = np.var(y)
    y += rng.normal(0, np.sqrt(max(1e-8, 1 - y_var)), n_samples)

    return dict(G=G.astype(np.int8),
                y=y,
                additive_idx=additive_idx.tolist(),
                pair_idx=[tuple(p) for p in pairs],
                beta=beta,
                gamma=gamma)

# --- minimal demo -------------------------------------------------------------
if __name__ == "__main__":
    sim = simulate_epistasis(n_samples=2_000, n_snps=50_000)
    print("G shape:", sim["G"].shape)
    print("Phenotype mean ± SD:", sim["y"].mean(), sim["y"].std())
    print("First 5 additive SNPs:", sim["additive_idx"][:5])
    print("First 5 epistatic pairs:", sim["pair_idx"][:5])