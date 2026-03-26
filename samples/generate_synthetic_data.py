#!/usr/bin/env python3
"""
generate_synthetic_data.py

Generates synthetic PLINK-format genomic data for demonstration and testing:
  <prefix>_train.bed / .bim / .fam / .pheno / .cov
  <prefix>_val.bed   / .bim / .fam / .pheno / .cov
  snp2gene.txt
  ontology.txt

All output is fully synthetic and contains no real participant data.

Usage:
    python generate_synthetic_data.py [options]

Options:
    --seed INT          Random seed (default: 42)
    --n_train INT       Training sample size (default: 1000)
    --n_val INT         Validation sample size (default: 500)
    --n_snps INT        Number of SNPs (default: 293)
    --n_genes INT       Number of unique genes (default: 150)
    --n_go_terms INT    Number of GO-like ontology terms (default: 80)
    --out_dir PATH      Output directory (default: directory of this script)
    --prefix_train STR  File prefix for train split (default: synthetic_train)
    --prefix_val STR    File prefix for val split (default: synthetic_val)
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

# ── path setup: locate epistasis_simulation relative to this script ───────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src", "utils", "analysis"))
from epistasis_simulation import simulate_epistasis  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
# GRCh38 chromosome lengths (bp) — used to draw realistic genomic positions
CHR_LENGTHS = {
    1:  248956422, 2:  242193529, 3:  198295559, 4:  190214555,
    5:  181538259, 6:  170805979, 7:  159345973, 8:  145138636,
    9:  138394717, 10: 133797422, 11: 135086622, 12: 133275309,
    13: 114364328, 14: 107043718, 15: 101991189, 16:  90338345,
    17:  83257441, 18:  80373285, 19:  58617616, 20:  64444167,
    21:  46709983, 22:  50818468,
}
ALT_FOR = {
    "A": ["C", "G", "T"],
    "C": ["A", "G", "T"],
    "G": ["A", "C", "T"],
    "T": ["A", "C", "G"],
}

# Pool of realistic-looking gene-name fragments
_GENE_PREFIXES = [
    "BRCA", "TP", "EGFR", "MYC", "APC", "PTEN", "RB", "VHL",
    "MLH", "MSH", "ATM", "CHEK", "PALB", "RAD", "FANCA",
    "ERBB", "KRAS", "NRAS", "BRAF", "PIK3CA", "AKT", "MTOR",
    "CDK", "CDKN", "BCL", "MCL", "MDM", "STAT", "JAK",
    "SRC", "ABL", "FLT", "KIT", "MET", "RET",
    "FGFR", "PDGFRA", "IGF", "IRS", "GRB", "RAF", "MAP2K",
    "MAPK", "TSC", "STK", "LKB", "AMPK", "ROCK", "PAK",
    "WNT", "FZD", "LRP", "DVL", "GSK", "CTNNB", "AXIN",
    "NOTCH", "DLL", "JAG", "HES", "HEY", "RBPJ",
    "SHH", "SMO", "PTCH", "GLI", "SUFU",
    "TGFB", "SMAD", "BMPR", "ACVR", "TGFBR",
    "VEGFA", "FGF", "EGF", "PDGF", "HGF", "ANGPT",
    "NRG", "EPHA", "EPHB", "SEMA", "PLXN",
    "ITGA", "ITGB", "FN", "COL", "MMP", "TIMP", "ADAM",
    "PGS", "TK", "CSGAL", "PITPNM", "LPL",
]
_GENE_SUFFIXES = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "1A", "1B", "2A", "2B", "A", "B", "L", "R", "S"]


# ── helpers ───────────────────────────────────────────────────────────────────

def make_snp_ids(n_snps: int, rng: np.random.Generator):
    """
    Return parallel arrays: chromosomes, positions, refs, alts, snp_id_strings.
    SNP IDs follow the format CHR:POS:REF:ALT used by PLINK / UK Biobank.
    """
    total_len = sum(CHR_LENGTHS.values())
    chrs = list(CHR_LENGTHS.keys())
    probs = np.array([CHR_LENGTHS[c] for c in chrs], dtype=float)
    probs /= probs.sum()

    assigned_chrs = rng.choice(chrs, size=n_snps, p=probs)

    # Draw positions within each chromosome, then sort per-chromosome
    pos_array = np.zeros(n_snps, dtype=int)
    chr_indices = {}
    for i, c in enumerate(assigned_chrs):
        chr_indices.setdefault(c, []).append(i)
    for c, idxs in chr_indices.items():
        lo, hi = 1_000_000, CHR_LENGTHS[c] - 1_000_000
        pos = np.sort(rng.integers(lo, hi, size=len(idxs)))
        for idx, p in zip(idxs, pos):
            pos_array[idx] = int(p)

    refs = np.array(rng.choice(list("ACGT"), size=n_snps))
    alts = np.array([rng.choice(ALT_FOR[r]) for r in refs])

    # Sort by chromosome then position (required by PLINK)
    order = np.lexsort((pos_array, assigned_chrs))
    assigned_chrs = np.array(assigned_chrs)[order]
    pos_array     = pos_array[order]
    refs          = refs[order]
    alts          = alts[order]

    snp_ids = [
        f"{c}:{p}:{r}:{a}"
        for c, p, r, a in zip(assigned_chrs, pos_array, refs, alts)
    ]
    return assigned_chrs, pos_array, refs, alts, snp_ids


def generate_gene_names(n, rng):
    """Return n unique plausible-looking gene name strings."""
    names = set()
    while len(names) < n:
        prefix = str(rng.choice(_GENE_PREFIXES))
        suffix = str(rng.choice(_GENE_SUFFIXES))
        names.add(prefix + suffix)
    return list(names)[:n]


def write_bed(filepath: str, G: np.ndarray, n_samples: int, n_snps: int) -> None:
    """
    Write a PLINK BED file in SNP-major mode.

    Encoding (A1 = ALT = counted/effect allele):
        dosage 2 (hom ALT) → 00
        dosage 1 (het)     → 10
        dosage 0 (hom REF) → 11
    4 genotypes packed per byte, LSB first.
    """
    _DOSAGE_TO_BITS = {0: 0b11, 1: 0b10, 2: 0b00}
    n_bytes_per_snp = (n_samples + 3) // 4

    with open(filepath, "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))  # magic bytes + SNP-major flag

        for snp_idx in range(n_snps):
            buf = bytearray(n_bytes_per_snp)
            for i, d in enumerate(G[:, snp_idx]):
                bits = _DOSAGE_TO_BITS.get(int(d), 0b01)  # 01 = missing
                buf[i // 4] |= bits << ((i % 4) * 2)
            f.write(buf)


def make_sample_ids(n, rng):
    """Generate n unique 7-digit numeric sample IDs (FID == IID)."""
    return rng.choice(np.arange(7_000_000, 9_999_999), size=n, replace=False).tolist()


def write_fam(ids, path):
    """Write PLINK FAM file: FID IID PID MID SEX PHENO (PHENO=-9 = missing)."""
    rows = [[iid, iid, 0, 0, 0, -9] for iid in ids]
    pd.DataFrame(rows).to_csv(path, sep=" ", header=False, index=False)


def write_pheno(ids, y, path):
    """Write PLINK-style phenotype file with header FID IID PHENOTYPE."""
    pd.DataFrame({"FID": ids, "IID": ids, "PHENOTYPE": y}).to_csv(
        path, sep="\t", index=False
    )


def write_cov(ids, n, rng, path):
    """
    Write covariate file with columns:
    FID  IID  SEX  AGE  AGE2  PC1 … PC10
    SEX  ∈ {0.0, 1.0}
    AGE  ∈ [40, 80] (integer stored as float)
    AGE2 = AGE²
    PC*  ~ N(0, 0.002²)  — simulated principal components
    """
    sexes = rng.choice([0.0, 1.0], size=n)
    ages = rng.integers(40, 81, size=n).astype(float)
    pcs = rng.normal(0.0, 0.002, size=(n, 10))

    # Center AGE before squaring to avoid near-perfect collinearity with AGE
    age_centered = ages - ages.mean()
    data = {"FID": ids, "IID": ids, "SEX": sexes, "AGE": ages, "AGE2": age_centered ** 2}
    for k in range(10):
        data[f"PC{k+1}"] = pcs[:, k]

    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def build_go_ontology(gene_names, n_terms, rng):
    """
    Build a small GO-like ontology DAG.

    Returns a list of (col1, col2, col3) triples where col3 is either
    'default' (term→term edge) or 'gene' (term→gene leaf edge).
    """
    # Generate unique GO-like term IDs
    term_set: set[str] = set()
    while len(term_set) < n_terms:
        term_set.add(f"GO:{rng.integers(1, 9_999_999):07d}")
    term_ids = list(term_set)
    rng.shuffle(term_ids)

    n_leaves = max(1, int(n_terms * 0.55))
    n_intermediate = max(1, n_terms - n_leaves - 1)

    leaves = term_ids[:n_leaves]
    intermediates = term_ids[n_leaves: n_leaves + n_intermediate]
    root = term_ids[-1]

    edges: list[tuple[str, str, str]] = []

    # Leaf → intermediate (or root if no intermediates)
    for leaf in leaves:
        parent = rng.choice(intermediates) if intermediates else root
        edges.append((str(parent), leaf, "default"))

    # Intermediate → root or another intermediate (avoid self-loops)
    for term in intermediates:
        candidates = [t for t in intermediates if t != term] + [root]
        parent = rng.choice(candidates)
        edges.append((str(parent), term, "default"))

    # Assign genes to leaf terms (each gene assigned to 1–3 leaves)
    for gene in gene_names:
        k = int(rng.integers(1, 4))
        chosen = rng.choice(leaves, size=min(k, len(leaves)), replace=False)
        for t in chosen:
            edges.append((t, gene, "gene"))

    return edges


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic PLINK genomic data for testing."
    )
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--n_train",      type=int, default=1000)
    parser.add_argument("--n_val",        type=int, default=500)
    parser.add_argument("--n_test",       type=int, default=300)
    parser.add_argument("--n_snps",       type=int, default=293)
    parser.add_argument("--n_genes",      type=int, default=150)
    parser.add_argument("--n_go_terms",   type=int, default=80)
    parser.add_argument("--out_dir",      type=str, default=None)
    parser.add_argument("--prefix_train", type=str, default="synthetic_train")
    parser.add_argument("--prefix_val",   type=str, default="synthetic_val")
    parser.add_argument("--prefix_test",  type=str, default="synthetic_test")
    args = parser.parse_args()

    out_dir = args.out_dir or SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_total = args.n_train + args.n_val + args.n_test

    # ── 1. Simulate genotypes and phenotype ───────────────────────────────────
    print(f"[1/6] Simulating {n_total} samples × {args.n_snps} SNPs …")
    sim = simulate_epistasis(
        n_samples=n_total,
        n_snps=args.n_snps,
        n_additive=20,
        n_pairs=10,
        h2_additive=0.3,
        h2_epistatic=0.1,
        seed=args.seed,
    )
    G = sim["G"].astype(int)   # (n_total, n_snps), values in {0, 1, 2}
    y = sim["y"]               # (n_total,)

    i1 = args.n_train
    i2 = args.n_train + args.n_val
    G_train, G_val, G_test = G[:i1], G[i1:i2], G[i2:]
    y_train, y_val, y_test = y[:i1], y[i1:i2], y[i2:]

    # ── 2. SNP metadata ───────────────────────────────────────────────────────
    print("[2/6] Generating SNP metadata …")
    chrs, positions, refs, alts, snp_ids = make_snp_ids(args.n_snps, rng)

    # ── 3. Sample IDs ─────────────────────────────────────────────────────────
    train_ids = make_sample_ids(args.n_train, rng)
    val_ids   = make_sample_ids(args.n_val,   rng)
    test_ids  = make_sample_ids(args.n_test,  rng)

    # ── 4. BIM (identical SNP panel for all splits) ───────────────────────────
    print("[3/6] Writing BIM / FAM / BED files …")
    bim_rows = [
        [chrs[i], snp_ids[i], 0, positions[i], alts[i], refs[i]]
        for i in range(args.n_snps)
    ]
    bim_train = os.path.join(out_dir, f"{args.prefix_train}.bim")
    pd.DataFrame(bim_rows).to_csv(bim_train, sep="\t", header=False, index=False)
    shutil.copy(bim_train, os.path.join(out_dir, f"{args.prefix_val}.bim"))
    shutil.copy(bim_train, os.path.join(out_dir, f"{args.prefix_test}.bim"))

    # ── 5. FAM ────────────────────────────────────────────────────────────────
    write_fam(train_ids, os.path.join(out_dir, f"{args.prefix_train}.fam"))
    write_fam(val_ids,   os.path.join(out_dir, f"{args.prefix_val}.fam"))
    write_fam(test_ids,  os.path.join(out_dir, f"{args.prefix_test}.fam"))

    # ── 6. BED ────────────────────────────────────────────────────────────────
    write_bed(os.path.join(out_dir, f"{args.prefix_train}.bed"),
              G_train, args.n_train, args.n_snps)
    write_bed(os.path.join(out_dir, f"{args.prefix_val}.bed"),
              G_val, args.n_val, args.n_snps)
    write_bed(os.path.join(out_dir, f"{args.prefix_test}.bed"),
              G_test, args.n_test, args.n_snps)

    # ── 7. PHENO ──────────────────────────────────────────────────────────────
    print("[4/6] Writing PHENO / COV files …")
    write_pheno(train_ids, y_train, os.path.join(out_dir, f"{args.prefix_train}.pheno"))
    write_pheno(val_ids,   y_val,   os.path.join(out_dir, f"{args.prefix_val}.pheno"))
    write_pheno(test_ids,  y_test,  os.path.join(out_dir, f"{args.prefix_test}.pheno"))

    # ── 8. COV ────────────────────────────────────────────────────────────────
    write_cov(train_ids, args.n_train, rng, os.path.join(out_dir, f"{args.prefix_train}.cov"))
    write_cov(val_ids,   args.n_val,   rng, os.path.join(out_dir, f"{args.prefix_val}.cov"))
    write_cov(test_ids,  args.n_test,  rng, os.path.join(out_dir, f"{args.prefix_test}.cov"))

    # ── 9. Gene names + snp2gene.txt ─────────────────────────────────────────
    print("[5/6] Generating gene annotations (snp2gene.txt) …")
    gene_names = generate_gene_names(args.n_genes, rng)

    snp2gene_rows = []
    for snp_id in snp_ids:
        k = 2 if rng.random() < 0.25 else 1   # ~25% of SNPs map to 2 genes
        for g in rng.choice(gene_names, size=k, replace=False):
            snp2gene_rows.append({"snp": snp_id, "gene": g})

    pd.DataFrame(snp2gene_rows).to_csv(
        os.path.join(out_dir, "snp2gene.txt"), sep="\t", index=False
    )

    # ── 10. ontology.txt ─────────────────────────────────────────────────────
    print("[6/6] Generating GO-like ontology (ontology.txt) …")
    ont_edges = build_go_ontology(gene_names, args.n_go_terms, rng)
    pd.DataFrame(ont_edges).to_csv(
        os.path.join(out_dir, "ontology.txt"), sep="\t", header=False, index=False
    )

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("Done. Files written to:", out_dir)
    print(f"  Train set : {args.n_train} samples → {args.prefix_train}.{{bed,bim,fam,pheno,cov}}")
    print(f"  Val set   : {args.n_val} samples   → {args.prefix_val}.{{bed,bim,fam,pheno,cov}}")
    print(f"  Test set  : {args.n_test} samples  → {args.prefix_test}.{{bed,bim,fam,pheno,cov}}")
    print(f"  SNPs      : {args.n_snps}")
    print(f"  Genes     : {len(gene_names)}")
    print(f"  GO terms  : {args.n_go_terms}")
    print(f"  snp→gene  : {len(snp2gene_rows)} mappings")
    print(f"  ontology  : {len(ont_edges)} edges "
          f"({sum(1 for e in ont_edges if e[2]=='default')} term-term, "
          f"{sum(1 for e in ont_edges if e[2]=='gene')} term-gene)")


if __name__ == "__main__":
    main()
