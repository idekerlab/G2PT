import numpy as np
import pandas as pd
import networkx as nx
import itertools
from scipy.stats import t as t_dist

def build_hierarchical_ontology(sim,
                                n_genes=800,
                                n_systems=60, # This is the number of leaf systems
                                epistatic_distinctness=0.5,
                                overlap_prob=0.25,
                                n_causal_systems=20,
                                causal_system_enrichment=5.0,
                                seed=123):
    """
    Construct SNP→Gene→System→...→Root hierarchy.
    Injects epistatic interactions based on a tunable "distinctness" parameter.
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

    # Identify causal genes and systems to enrich
    if n_causal_systems > 0 and n_causal_systems <= n_systems:
        # Temporarily map SNPs to genes to get a proxy set of causal genes
        temp_snp2gene = rng.choice(gene_ids, size=n_snps)
        causal_snp_indices = set(sim['additive_idx']) | {snp for pair in sim['pair_idx'] for snp in pair}
        causal_genes = {temp_snp2gene[idx] for idx in causal_snp_indices}
        
        # Select causal systems and create weighted distribution for assignment
        causal_systems = rng.choice(system_ids, size=n_causal_systems, replace=False)
        weights = np.array([causal_system_enrichment if s in causal_systems else 1.0 for s in system_ids])
        weights /= weights.sum()

        # Assign genes to systems using the biased distribution for causal genes
        for g in gene_ids:
            if g in causal_genes:
                chosen_sys = rng.choice(system_ids, p=weights)
                gene2system[g].add(chosen_sys)
            else:
                chosen_sys = rng.choice(system_ids) # Uniform for non-causal
                gene2system[g].add(chosen_sys)
    else:
        # Original behavior: assign each gene to a random leaf system
        for g in gene_ids:
            gene2system[g].add(rng.choice(system_ids))

    # --- 4a. Pre-calculate system pair pools for epistasis placement ---
    print("Pre-calculating system pair pools for epistasis placement...")
    leaf_systems = system_ids
    root_nodes = [n for n, d in sys_graph.in_degree() if d == 0]
    
    if not root_nodes or not parent_child_edges:
        print("Warning: No root node or hierarchy found. Using fallback for distinctness.")
        all_pairs = list(itertools.combinations(leaf_systems, 2))
        related_system_pairs = all_pairs
        distant_system_pairs = all_pairs
    else:
        root = root_nodes[0]
        related_system_pairs = []
        distant_system_pairs = []
        all_pairs = list(itertools.combinations(leaf_systems, 2))
        for s1, s2 in all_pairs:
            if s1 not in sys_graph or s2 not in sys_graph: continue
            lca = nx.lowest_common_ancestor(sys_graph, s1, s2)
            if lca == root:
                distant_system_pairs.append((s1, s2))
            else:
                related_system_pairs.append((s1, s2))

    same_system_pairs = [(s, s) for s in leaf_systems]
    
    rng.shuffle(same_system_pairs)
    rng.shuffle(related_system_pairs)
    rng.shuffle(distant_system_pairs)

    print(f"  - Found {len(same_system_pairs)} same-system pairs.")
    print(f"  - Found {len(related_system_pairs)} related-system pairs.")
    print(f"  - Found {len(distant_system_pairs)} distant-system pairs.")

    # ----------------- 4b. SNP → Gene Mapping with Epistasis Logic -----------------
    snp2gene = rng.choice(gene_ids, size=n_snps)
    epistatic_pairs = sim["pair_idx"]

    print(f"Assigning {len(epistatic_pairs)} epistatic pairs with distinctness = {epistatic_distinctness}...")
    for snp_i, snp_j in epistatic_pairs:
        # Determine which pool to use based on epistatic_distinctness
        if epistatic_distinctness < 0.5:
            prob_same = 1 - (epistatic_distinctness / 0.5)
            pool = same_system_pairs if rng.random() < prob_same else related_system_pairs
        else:
            prob_related = 1 - ((epistatic_distinctness - 0.5) / 0.5)
            pool = related_system_pairs if rng.random() < prob_related else distant_system_pairs

        # Failsafe logic if the chosen pool is empty
        if not pool:
            if related_system_pairs: pool = related_system_pairs
            elif distant_system_pairs: pool = distant_system_pairs
            elif same_system_pairs: pool = same_system_pairs
            else:
                sys_for_i, sys_for_j = rng.choice(leaf_systems, 2, replace=True)
                pool = None

        if pool:
            sys_idx = rng.integers(len(pool))
            sys_for_i, sys_for_j = pool[sys_idx]

        # Find genes in those leaf systems and assign the SNPs
        genes_in_sys_i = [g for g, systems in gene2system.items() if sys_for_i in systems]
        genes_in_sys_j = [g for g, systems in gene2system.items() if sys_for_j in systems]
        
        g_i = rng.choice(genes_in_sys_i) if genes_in_sys_i else rng.choice(gene_ids)
        g_j = rng.choice(genes_in_sys_j) if genes_in_sys_j else rng.choice(gene_ids)
        
        snp2gene[snp_i] = g_i
        snp2gene[snp_j] = g_j
        
        # Ensure the chosen genes are actually in the target systems
        gene2system[g_i].add(sys_for_i)
        gene2system[g_j].add(sys_for_j)

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

    # Ensure all systems have at least one gene to prevent downstream NaNs
    genes_in_system = {s: [] for s in system_ids}
    for g, systems in gene2system.items():
        for s in systems:
            genes_in_system[s].append(g)
    
    for s in system_ids:
        if not genes_in_system[s]:
            chosen_gene = rng.choice(gene_ids)
            gene2system[chosen_gene].add(s)

    # --- 5a. Ensure all genes are mapped to at least one SNP ---
    assigned_genes = set(snp2gene)
    unassigned_genes = set(gene_ids) - assigned_genes
    if unassigned_genes:
        print(f"Warning: {len(unassigned_genes)} genes were not assigned any SNPs. Assigning them now to random SNPs.")
        unassigned_gene_list = list(unassigned_genes)
        
        if len(unassigned_gene_list) > n_snps:
            print(f"  - Warning: More unassigned genes ({len(unassigned_gene_list)}) than SNPs ({n_snps}).")
            print(f"  - Only {n_snps} genes will be assigned a unique SNP.")
            unassigned_gene_list = rng.choice(unassigned_gene_list, size=n_snps, replace=False).tolist()

        snp_indices_to_reassign = rng.choice(np.arange(n_snps), size=len(unassigned_gene_list), replace=False)
        for i, gene_id in enumerate(unassigned_gene_list):
            snp2gene[snp_indices_to_reassign[i]] = gene_id


    # --------------- 6. Flatten to tables ---------------
    ld_blocks = sim.get('ld_blocks')
    snp_chromosomes = np.zeros(n_snps, dtype=int)
    snp_blocks_cosmetic = np.zeros(n_snps, dtype=int)
    snp_positions = np.zeros(n_snps, dtype=int)

    if ld_blocks is not None:
        print("LD structure found. Assigning chromosomes, positions, and blocks based on LD.")
        unique_ld_blocks = np.unique(ld_blocks)
        # Assign a large genomic region to each LD block
        block_starts = np.arange(len(unique_ld_blocks)) * 1_000_000
        
        for i, block_id in enumerate(unique_ld_blocks):
            in_block = (ld_blocks == block_id)
            n_in_block = in_block.sum()
            
            # Assign all SNPs in the same LD block to the same chromosome and cosmetic block
            snp_chromosomes[in_block] = rng.integers(1, 23)
            snp_blocks_cosmetic[in_block] = rng.integers(1, 5)
            
            # Stagger positions by a small random amount within the block's genomic region
            offsets = rng.integers(1, 1001, size=n_in_block)
            snp_positions[in_block] = block_starts[i] + np.cumsum(offsets)
    else:
        print("No LD structure found. Assigning chromosomes, positions, and blocks randomly.")
        # Fallback to random assignments if no LD info is present
        snp_chromosomes = rng.integers(1, 23, size=n_snps)
        snp_positions = rng.integers(1, 10_000_000, size=n_snps)
        snp_blocks_cosmetic = rng.integers(1, 5, size=n_snps)

    
    snp_df = pd.DataFrame({
        "chr": snp_chromosomes,
        "pos": snp_positions,
        "block": snp_blocks_cosmetic,
        "gene": snp2gene
    })
    snp_df.index = np.arange(n_snps)
    snp_df.index.name = 'snp'
    # Reorder columns to the desired format, keeping the 'pos' column
    snp_df = snp_df[['gene', 'chr', 'pos', 'block']]

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
        n_additive=100,          # causal additive SNPs
        n_pairs=50,              # causal epistatic pairs
        h2_additive=0.5,         # additive heritability
        h2_epistatic=0.1,        # epistatic heritability
        min_additive_p_value=1e-4, # p-value threshold for marginal GWAS significance
        min_epistatic_p_value=1e-3,# p-value threshold for epistatic interaction
        maf_range=(0.05, 0.5),   # minor-allele-frequency window
        n_ld_blocks=100,         # number of LD blocks to simulate
        ld_rho=0.8,              # correlation coefficient for adjacent SNPs in a block
        epistasis_bias=10.0,     # factor to bias epistasis towards non-significant SNPs
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

    # 1. simulate genotypes with real-like MAF spectrum and LD structure
    mafs = rng.uniform(*maf_range, size=n_snps)
    
    if ld_rho > 0 and n_ld_blocks > 0:
        # Simulate genotypes in blocks with LD
        n_snps_per_block = n_snps // n_ld_blocks
        all_G_blocks = []
        
        for i in range(n_ld_blocks):
            block_size = n_snps_per_block
            if i == n_ld_blocks - 1:
                block_size = n_snps - (n_ld_blocks - 1) * n_snps_per_block
            
            cov = np.zeros((block_size, block_size))
            for j in range(block_size):
                for k in range(j, block_size):
                    cov[j, k] = cov[k, j] = ld_rho ** abs(j - k)

            latent_G = rng.multivariate_normal(mean=np.zeros(block_size), cov=cov, size=n_samples)
            
            G_block = np.zeros_like(latent_G, dtype=np.int8)
            block_mafs = mafs[i*n_snps_per_block : i*n_snps_per_block + block_size]
            
            from scipy.stats import norm
            t1 = norm.ppf((1 - block_mafs)**2)
            t2 = norm.ppf((1 - block_mafs)**2 + 2 * block_mafs * (1 - block_mafs))

            G_block[latent_G > t2] = 2
            G_block[(latent_G > t1) & (latent_G <= t2)] = 1
            
            all_G_blocks.append(G_block)
        
        G = np.concatenate(all_G_blocks, axis=1).astype(np.float32)

    else:
        G = rng.binomial(2, mafs, size=(n_samples, n_snps)).astype(np.float32)

    G_std = (G - G.mean(0)) / G.std(0)

    # 2. Select causal additive SNPs and generate their effect sizes (betas)
    additive_idx = rng.choice(n_snps, size=n_additive, replace=False)
    beta = np.zeros(n_snps)

    if min_additive_p_value is not None:
        from scipy.stats import t as t_dist
        df = n_samples - 2
        t_crit = t_dist.ppf(1 - min_additive_p_value / 2, df)
        min_beta_sq = t_crit**2 / (n_samples + t_crit**2)
        
        required_h2 = n_additive * min_beta_sq
        if h2_additive < required_h2:
            print(f"Warning: h2_additive ({h2_additive}) is too low to ensure all {n_additive} "
                  f"causal SNPs have p < {min_additive_p_value}.")
            print(f"  Required h2_additive >= {required_h2:.4f}.")
            print(f"  Forcing all effect sizes to be equal to meet the h2 budget.")
            min_beta_sq = h2_additive / n_additive
            h2_excess = 0
        else:
            h2_excess = h2_additive - required_h2

        # Distribute the excess heritability randomly
        u = rng.random(n_additive)
        c = (u / u.sum()) * h2_excess if h2_excess > 0 else 0
        
        beta_sq_values = c + min_beta_sq
        causal_betas = np.sqrt(beta_sq_values)
        causal_betas *= rng.choice([-1, 1], n_additive) # Assign random signs
        beta[additive_idx] = causal_betas
    else:
        # Original behavior: draw from normal and scale
        all_betas = rng.standard_normal(n_snps)
        beta[additive_idx] = all_betas[additive_idx]
        beta_scaling_factor = np.sqrt(h2_additive / np.sum(beta**2)) if np.sum(beta**2) > 0 else 0
        beta *= beta_scaling_factor

    # 3. Select epistatic pairs with a bias towards non-significant SNPs
    # Calculate weights inversely proportional to the squared effect size
    # To do this, we need a proxy for effect size for ALL snps, not just causal ones.
    # We'll generate them like the original code did, just for weighting.
    all_betas_for_weighting = rng.standard_normal(n_snps)
    weights = 1.0 / (all_betas_for_weighting**2 + 1.0/epistasis_bias)
    # Ensure causal additive SNPs are not selected for epistasis
    weights[additive_idx] = 0 
    weights /= np.sum(weights) # Normalize to a probability distribution

    # Choose pairs based on these weights
    pair_indices = rng.choice(n_snps, size=2 * n_pairs, replace=False, p=weights)
    pairs = pair_indices.reshape(n_pairs, 2)

    # Draw epistatic effect sizes and scale to heritability
    if n_pairs > 0:
        if min_epistatic_p_value is not None:
            from scipy.stats import t as t_dist
            # For y ~ B0 + B1*x1 + B2*x2 + G*x1*x2, df = n_samples - 4
            df = n_samples - 4
            t_crit = t_dist.ppf(1 - min_epistatic_p_value / 2, df)
            
            # Approx. for standardized, uncorrelated predictors: gamma^2 ~ t^2 / n_samples
            min_gamma_sq = t_crit**2 / n_samples
            
            required_h2 = n_pairs * min_gamma_sq
            if h2_epistatic < required_h2:
                print(f"Warning: h2_epistatic ({h2_epistatic}) is too low to ensure all {n_pairs} "
                      f"epistatic pairs have p < {min_epistatic_p_value}.")
                print(f"  Required h2_epistatic >= {required_h2:.4f}.")
                print(f"  Forcing all epistatic effect sizes to be equal to meet the h2 budget.")
                min_gamma_sq = h2_epistatic / n_pairs
                h2_excess = 0
            else:
                h2_excess = h2_epistatic - required_h2

            # Distribute the excess heritability randomly
            u = rng.random(n_pairs)
            c = (u / u.sum()) * h2_excess if h2_excess > 0 else 0
            
            gamma_sq_values = c + min_gamma_sq
            gamma = np.sqrt(gamma_sq_values)
            gamma *= rng.choice([-1, 1], n_pairs) # Assign random signs
        else:
            # Original behavior
            gamma_scaling_factor = np.sqrt(h2_epistatic / n_pairs)
            gamma = rng.standard_normal(n_pairs) * gamma_scaling_factor
    else:
        gamma = np.array([])

    # 4. build phenotype: additive + pairwise + environmental noise
    y = G_std @ beta
    for k, (i, j) in enumerate(pairs):
        y += gamma[k] * G_std[:, i] * G_std[:, j]

    # add noise so Var(y) ≈ 1
    y_var = np.var(y)
    y += rng.normal(0, np.sqrt(max(1e-8, 1 - y_var)), n_samples)

    # 5. Create LD block assignments for position generation
    ld_blocks = np.zeros(n_snps, dtype=int)
    if ld_rho > 0 and n_ld_blocks > 0:
        n_snps_per_block = n_snps // n_ld_blocks
        for i in range(n_ld_blocks):
            start_idx = i * n_snps_per_block
            end_idx = start_idx + n_snps_per_block
            if i == n_ld_blocks - 1:
                end_idx = n_snps
            ld_blocks[start_idx:end_idx] = i

    return dict(G=G.astype(np.int8),
                y=y,
                additive_idx=additive_idx.tolist(),
                pair_idx=[tuple(p) for p in pairs],
                beta=beta[additive_idx], # Return only the causal betas
                gamma=gamma,
                ld_blocks=ld_blocks)

# --- minimal demo -------------------------------------------------------------
if __name__ == "__main__":
    sim = simulate_epistasis(n_samples=2_000, n_snps=50_000)
    print("G shape:", sim["G"].shape)
    print("Phenotype mean ± SD:", sim["y"].mean(), sim["y"].std())
    print("First 5 additive SNPs:", sim["additive_idx"][:5])
    print("First 5 epistatic pairs:", sim["pair_idx"][:5])
