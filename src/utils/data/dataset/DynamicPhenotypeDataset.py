import torch
import math
import random

from torch.utils.data import IterableDataset, get_worker_info

from .SNP2PDataset import PLINKDataset
import torch.distributed as dist


class DynamicPhenotypeBatchIterableDataset(IterableDataset):
    def __init__(self, tree_parser, dataset: PLINKDataset, collator, batch_size, shuffle=True, n_phenotype2sample=1):
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset    = dataset      # map‑style dataset, len() defined
        self.collator   = collator
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.n_phenotype2sample = n_phenotype2sample
        self.n_pheno = self.dataset.n_pheno

    # OPTIONAL: lets DataLoader len(loader) work
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def set_n_phenotype2sample(self, n_phenotype2sample):
        self.n_phenotype2sample = n_phenotype2sample

    def __iter__(self):
        ############ 0.  worker bookkeeping ############
        worker = get_worker_info()          # None if num_workers == 0
        total  = len(self.dataset)          # N samples in underlying ds

        if worker is None:                  # single‑process data‑load
            w_start, w_end   = 0, total
        else:                               # multi‑worker sharding
            per_worker       = math.ceil(total / worker.num_workers)
            w_start          = worker.id * per_worker
            w_end            = min(w_start + per_worker, total)

        # ---- build THIS worker’s index list ----
        indices = list(range(w_start, w_end))
        if self.shuffle:
            random.shuffle(indices)

        ############ 1.  walk in batch‑size chunks ############
        for lo in range(0, len(indices), self.batch_size):
            batch_inds = indices[lo : lo + self.batch_size]

            # if the tail slice is empty (can only happen if len==0)
            if not batch_inds:
                break

            # ----- draw phenotypes & subtree ONCE for this batch -----
            self.dataset.sample_phenotypes(n=self.n_phenotype2sample)
            # pull raw samples
            raw_batch = [self.dataset[i] for i in batch_inds]
            # base collate
            #print(self.dataset.snp_range, self.dataset.gene_range)
            collated  = self.collator(raw_batch)
            collated['snp2gene_mask'] = torch.tensor(self.tree_parser.snp2gene_mask[self.dataset.gene_range][:, self.dataset.snp_range], dtype=torch.float32)
            collated['gene2sys_mask'] = torch.tensor(self.tree_parser.gene2sys_mask[self.dataset.sys_range][:, self.dataset.gene_range], dtype=torch.float32)
            collated['gene_indices'] = torch.tensor(
                [gene_ind for gene_ind in self.dataset.gene_range if gene_ind != self.tree_parser.n_genes],
                dtype=torch.long)
            collated['sys_indices'] = torch.tensor(
                [sys_ind for sys_ind in self.dataset.sys_range if sys_ind != self.tree_parser.n_systems],
                dtype=torch.long)
            yield collated

class DynamicPhenotypeBatchIterableDatasetDDP(IterableDataset):
    def __init__(self, tree_parser, dataset, collator, batch_size, shuffle=True, n_phenotype2sample=1):
        """
        base_ds   : map‑style dataset with __len__ / __getitem__
        collator  : callable(list(raw_samples)) → dict  (no masks yet)
        batch_size: number of samples per batch
        shuffle   : shuffle indices each epoch
        """
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset    = dataset
        self.collator   = collator
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.n_phenotype2sample = n_phenotype2sample

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def set_n_phenotype2sample(self, n_phenotype2sample):
        self.n_phenotype2sample = n_phenotype2sample

    def _index_slice_for_worker_and_rank(self):
        """Return the list of indices this *process* should iterate."""
        N = len(self.dataset)

        # ── split by worker (DataLoader subprocess) ──
        w_info = get_worker_info()
        if w_info is None:
            w_start, w_end, w_workers = 0, N, 1
            w_id = 0
        else:
            per_worker = math.ceil(N / w_info.num_workers)
            w_start    = w_id = w_info.id
            w_start   *= per_worker
            w_end      = min(w_start + per_worker, N)
            w_workers  = w_info.num_workers

        indices = list(range(w_start, w_end))

        # ── split further by DDP rank ──
        if dist.is_initialized():
            rank       = dist.get_rank()
            world      = dist.get_world_size()
            indices    = indices[rank::world]   # stride slicing
        else:
            rank, world = 0, 1

        if self.shuffle and (rank == 0 and w_id == 0):
            random.shuffle(indices)             # shuffle once then broadcast
        # simple broadcast for reproducibility
        if dist.is_initialized():
            idx_tensor = torch.tensor(indices, dtype=torch.int64, device="cpu")
            idx_sizes  = torch.tensor([len(indices)], dtype=torch.int64, device="cpu")
            dist.broadcast(idx_sizes, src=0)
            if rank != 0:
                idx_tensor = torch.empty(idx_sizes.item(), dtype=torch.int64)
            dist.broadcast(idx_tensor, src=0)
            indices = idx_tensor.tolist()

        return indices

    def __iter__(self):
        indices = self._index_slice_for_worker_and_rank()

        for chunk_start in range(0, len(indices), self.batch_size):
            inds = indices[chunk_start : chunk_start + self.batch_size]
            if not inds:
                break
            raw_batch = [self.dataset[i] for i in inds]
            self.dataset.sample_phenotypes(n=self.n_phenotype2sample)
            # base collate
            collated  = self.collator(raw_batch)
            collated['snp2gene_mask'] = torch.tensor(self.tree_parser.snp2gene_mask[self.dataset.snp_range][:, self.dataset.gene_range], dtype=torch.float32)
            collated['gene2sys_mask'] = torch.tensor(self.tree_parser.gene2sys_mask[self.dataset.gene_range][:, self.dataset.sys_range], dtype=torch.float32)
            collated['gene_indices'] = torch.tensor(
                [gene_ind for gene_ind in self.dataset.gene_range if gene_ind != self.tree_parser.n_genes],
                dtype=torch.long)
            collated['sys_indices'] = torch.tensor(
                [sys_ind for sys_ind in self.dataset.sys_range if sys_ind != self.tree_parser.n_systems],
                dtype=torch.long)
            yield collated
