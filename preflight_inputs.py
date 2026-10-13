"""Fail-closed metadata checks. Writes aggregate counts/hashes, never sample IDs."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--gwas-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.project / 'G2PT_input/UKBiobank'
    gwas = root / args.gwas_name
    fold_data = gwas / 'fold'
    if not fold_data.exists():
        fold_data = root / 'fold'
    prefix = f'ukb_snp_chr_all.newid.imputed.fold_{args.fold}'
    mapping = gwas / f'snp2gene_fold_{args.fold}_pval_1e-4.HapMap.snp2gene.T2D.txt'
    ontology = gwas / f'ont_fold_{args.fold}_pval_1e-4.HapMap.snp2gene.T2D.txt'
    paths = [mapping, ontology]
    ids, report, alleles = {}, {}, {}
    for split in ('train', 'val'):
        bfile = gwas / f'{prefix}.{split}.HapMap.renamed.snp2gene.T2D.pval_1e-4'
        fam_path = Path(str(bfile) + '.fam')
        bim_path = Path(str(bfile) + '.bim')
        bed_path = Path(str(bfile) + '.bed')
        cov_path = fold_data / f'fold_{args.fold}_{split}_selected.cov'
        pheno_path = fold_data / f'fold_{args.fold}_{split}_selected.G2PT.pheno'
        paths.extend([fam_path, bim_path, bed_path, cov_path, pheno_path])
        fam = pd.read_csv(fam_path, sep=r'\s+', header=None, dtype=str)
        bim = pd.read_csv(bim_path, sep=r'\s+', header=None, dtype=str)
        cov = pd.read_csv(cov_path, sep='\t', dtype={'FID': str, 'IID': str})
        pheno = pd.read_csv(pheno_path, sep='\t', dtype={'FID': str, 'IID': str})
        for label, column in [('fam', fam[1]), ('cov', cov.IID), ('pheno', pheno.IID)]:
            if column.duplicated().any() or column.isna().any():
                raise ValueError(f'{split}: invalid/duplicate {label} IDs')
        ids[split] = set(fam[1])
        if ids[split] != set(cov.IID) or ids[split] != set(pheno.IID):
            raise ValueError(f'{split}: sample sets differ across genotype, covariates and phenotype')
        if not cov.SEX.isin([0, 1]).all() or not np.isfinite(cov[['SEX', 'AGE']].to_numpy()).all():
            raise ValueError(f'{split}: invalid SEX/AGE coding')
        if cov.AGE.std() <= 0:
            raise ValueError(f'{split}: constant AGE cannot be standardized')
        if not pheno.T2D.isin([0, 1]).all() or pheno.T2D.nunique() != 2:
            raise ValueError(f'{split}: T2D must contain both 0 and 1 only')
        if bim[1].duplicated().any():
            raise ValueError(f'{split}: duplicate variant IDs')
        alleles[split] = {row[1]: (row[0], row[3], row[4], row[5])
                          for row in bim.itertuples(index=False, name=None)}
        if bed_path.stat().st_size != 3 + ((len(fam) + 3) // 4) * len(bim):
            raise ValueError(f'{split}: unexpected PLINK bed size')
        with bed_path.open('rb') as handle:
            if handle.read(3) != b'\x6c\x1b\x01':
                raise ValueError(f'{split}: expected SNP-major PLINK bed')
        report[split] = {'samples': len(fam), 'variants': len(bim),
                         'cases': int(pheno.T2D.sum()), 'controls': int((pheno.T2D == 0).sum())}
    if ids['train'] & ids['val']:
        raise ValueError('Train/validation sample overlap')
    if alleles['train'].keys() != alleles['val'].keys():
        raise ValueError('Train/validation SNP sets differ')
    swaps = 0
    for snp, reference in alleles['train'].items():
        observed = alleles['val'][snp]
        if observed == reference:
            continue
        if observed[:2] == reference[:2] and observed[2:] == reference[2:][::-1]:
            swaps += 1
        else:
            raise ValueError('Train/validation genomic coordinates or allele sets differ')
    # Test identifiers are checked only for split separation. No test labels/scores are read.
    test_fam = gwas / f'{prefix}.test.HapMap.renamed.snp2gene.T2D.pval_1e-4.fam'
    test_ids = set(pd.read_csv(test_fam, sep=r'\s+', header=None, dtype=str)[1])
    if test_ids & (ids['train'] | ids['val']):
        raise ValueError('Test sample overlap with train/validation')
    paths.append(test_fam)
    result = {'fold': args.fold, 'splits': report, 'test_id_count': len(test_ids),
              'split_overlap': False, 'allele_sets_match_train_val': True,
              'validation_allele_swaps_to_align': swaps,
              'files': [{'path': str(path), 'bytes': path.stat().st_size,
                         'sha256': sha256(path)} for path in paths]}
    code_root = Path(__file__).resolve().parent
    result['code_sha256'] = {str(path.relative_to(code_root)): sha256(path)
                             for path in sorted(code_root.rglob('*.py'))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'code_sha256')}))


if __name__ == '__main__':
    main()
