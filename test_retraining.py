"""Small synthetic regressions; no participant data required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.utils.trainer.snp2p_trainer import SNP2PTrainer
from src.utils.trainer.utils import EarlyStopping
from src.utils.data.dataset.genotype_datasets import PLINKDataset


class TinyTrainer(SNP2PTrainer):
    def __init__(self, epochs, val_step=1):
        self.args = SimpleNamespace(start_epoch=0, rank=0, distributed=False,
                                    val_step=val_step, epochs=epochs, seed=11)
        self.snp2p_model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.AdamW(self.snp2p_model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=0, factor=0.5)
        self.early_stopping = EarlyStopping(patience=100, verbose=False)
        self.validation_dataloader = object()
        self.phenotypes = ['T2D']
        self.use_mlflow = False
        self.bt_validation_metric = 'aupr'
        self.last_binary_metrics = {}
        self.current_epoch = 0
        self.best_epoch = None
        self.evaluated_epochs = []
        self.optimizer_ids = []

    def train_epoch(self, epoch, ccc=False):
        self.optimizer_ids.append(id(self.optimizer))
        self.optimizer.zero_grad()
        self.snp2p_model(torch.ones(2, 1)).square().mean().backward()
        self.optimizer.step()

    def evaluate(self, model, dataloader, epoch, **kwargs):
        self.evaluated_epochs.append(epoch)
        return 0.5


class TrainingTests(unittest.TestCase):
    def test_scheduler_updates_active_optimizer_and_saves_best(self):
        trainer = TinyTrainer(3)
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'model.pt')
            trainer.train(3, prefix)
            self.assertIs(trainer.scheduler.optimizer, trainer.optimizer)
            self.assertEqual(len(set(trainer.optimizer_ids)), 1)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 0.025)
            saved = torch.load(prefix + '.best', map_location='cpu', weights_only=False)
            self.assertEqual(saved['training_metadata']['best_epoch'], 1)
            self.assertEqual(saved['training_metadata']['seed'], 11)
            fresh = torch.nn.Linear(1, 1)
            fresh.load_state_dict(saved['state_dict'], strict=True)
            for key, tensor in trainer.snp2p_model.state_dict().items():
                self.assertTrue(torch.equal(tensor, fresh.state_dict()[key]))
            records = [json.loads(line) for line in Path(prefix + '.validation.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in records], [1, 2, 3])

    def test_validation_includes_last_epoch_and_short_run(self):
        for epochs, expected in [(7, [5, 7]), (1, [1])]:
            trainer = TinyTrainer(epochs, val_step=5)
            trainer.train(epochs)
            self.assertEqual(trainer.evaluated_epochs, expected)


class Array:
    def __init__(self, values):
        self.values = np.array(values)

    def as_numpy(self):
        return self.values


class Plink(dict):
    def __getattr__(self, key):
        return self[key]


class Tree:
    n_snps = 2
    n_genes = 1
    n_systems = 1
    snp2ind = {'v1': 0, 'v2': 1}
    block = False

    def set_snp2chr(self, value):
        self.snp2chr = value

    def set_snp2pos(self, value):
        self.snp2pos = value


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cov = str(Path(self.temp.name) / 'cov.tsv')
        self.pheno = str(Path(self.temp.name) / 'pheno.tsv')
        pd.DataFrame({'FID': ['s1', 's2'], 'IID': ['s1', 's2'],
                      'SEX': [0, 1], 'AGE': [40, 60]}).to_csv(self.cov, sep='\t', index=False)
        pd.DataFrame({'FID': ['s1', 's2'], 'IID': ['s1', 's2'],
                      'T2D': [0, 1]}).to_csv(self.pheno, sep='\t', index=False)
        self.plink = Plink(variant_id=Array(['v1', 'v2']), contig_id=Array(['1', '2']),
                           variant_contig=Array([0, 1]), variant_position=Array([12345, 67890]),
                           variant_allele=Array([['A', 'G'], ['C', 'T']]),
                           sample_id=Array(['s2', 's1']),
                           call_genotype=Array([[[0, 0], [0, 1]], [[1, 1], [1, 0]]]))

    def make_dataset(self, reference_alleles=None):
        with patch('src.utils.data.dataset.genotype_datasets.plink.read_plink', return_value=self.plink):
            return PLINKDataset(Tree(), 'synthetic', self.cov, self.pheno,
                                cov_ids=['SEX', 'AGE'], pheno_ids=['T2D'], bt=['T2D'],
                                reference_alleles=reference_alleles)

    def test_order_and_genomic_coordinates(self):
        dataset = self.make_dataset()
        self.assertEqual(dataset.cov_df.IID.tolist(), ['s2', 's1'])
        self.assertEqual(dataset.pheno_df.T2D.tolist(), [1, 0])
        self.assertEqual(dataset.tree_parser.snp2chr, {'v1': 1, 'v2': 2})
        self.assertEqual(dataset.tree_parser.snp2pos, {'v1': 12345, 'v2': 67890})

    def test_missing_snp_fails_closed(self):
        self.plink['variant_id'] = Array(['v1', 'other'])
        with self.assertRaisesRegex(ValueError, 'Missing 1 required SNPs'):
            self.make_dataset()

    def test_missing_genotype_is_not_homozygous(self):
        # A partial missing call [-1, 1] sums to 0, but is not a valid 0 dosage.
        self.plink.call_genotype.values[0, 0] = [-1, 1]
        with self.assertRaisesRegex(ValueError, 'missing/invalid'):
            self.make_dataset()

    def test_allele_reversal_is_aligned_to_training(self):
        reference = {'v1': ('G', 'A'), 'v2': ('C', 'T')}
        dataset = self.make_dataset(reference)
        self.assertEqual(dataset.n_allele_swaps, 1)
        self.assertEqual(dataset.variant_alleles, reference)
        self.assertEqual(dataset.snp_idx[:, 0].tolist(), [0, 2])

    def test_incompatible_alleles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Alleles do not match'):
            self.make_dataset({'v1': ('C', 'G'), 'v2': ('C', 'T')})


if __name__ == '__main__':
    unittest.main(verbosity=2)
