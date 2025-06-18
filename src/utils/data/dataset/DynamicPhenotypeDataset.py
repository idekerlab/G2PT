import torch
import math
import random

from torch.utils.data import IterableDataset, get_worker_info

from .SNP2PDataset import PLINKDataset
import torch.distributed as dist
from .. import pad_indices


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

            batch_is_valid = False
            attempt = 0
            while not batch_is_valid:
                # The phenotype seed is dependent on the attempt number
                phenotype_seed = (self.rank * len(indices)) + lo + attempt
                self.dataset.sample_phenotypes(n=self.n_phenotype2sample, seed=phenotype_seed)

                # Fetch and collate data
                raw_batch = [self.dataset[i] for i in batch_inds]
                collated = self.collator(raw_batch)

                # THE CRITICAL CHECK
                if 'phenotype' in collated and torch.all(collated['phenotype'] == -9):
                    # Invalid batch, increment attempt and the loop will try again
                    attempt += 1
                    continue

                # If we are here, the batch is valid.
                batch_is_valid = True # This will stop the while loop

                # Add the rest of the metadata
                collated['snp2gene_mask'] = torch.tensor(
                    self.tree_parser.snp2gene_mask[self.dataset.gene_range][:, self.dataset.snp_range], dtype=torch.float32
                )
                collated['gene2sys_mask'] = torch.tensor(
                    self.tree_parser.gene2sys_mask[self.dataset.sys_range][:, self.dataset.gene_range], dtype=torch.float32
                )
                collated['genotype']['gene_indices'] = torch.tensor(self.dataset.gene_range, dtype=torch.long)
                collated['genotype']['sys_indices'] = torch.tensor(self.dataset.sys_range, dtype=torch.long)

                yield collated


class DynamicPhenotypeBatchIterableDatasetDDP(IterableDataset):
    # Add rank and world_size to the constructor
    def __init__(self, tree_parser, dataset: PLINKDataset, collator, batch_size, rank, world_size, shuffle=True,
                 n_phenotype2sample=1, seed=None):
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_phenotype2sample = n_phenotype2sample
        self.n_pheno = self.dataset.n_pheno

        # Store the DDP info
        self.rank = rank
        self.world_size = world_size
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.seed = seed
        full_indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(full_indices)

        if dist.is_initialized():
            obj_list = [full_indices if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(obj_list, src=0)
            full_indices = obj_list[0]

        self.full_indices = full_indices

    def __len__(self):
        """Return the guaranteed number of batches all ranks will produce"""
        if dist.is_initialized():
            world_size = dist.get_world_size()
            # Conservative estimate: only count complete batches
            samples_per_rank = len(self.dataset) // world_size
            batches_per_rank = samples_per_rank // self.batch_size
            return batches_per_rank
        else:
            return len(self.dataset) // self.batch_size

    def set_n_phenotype2sample(self, n_phenotype2sample):
        self.n_phenotype2sample = n_phenotype2sample

    def _index_slice_for_worker_and_rank(self):
        """Return the list of indices this *process* should iterate."""
        #N = len(self.dataset)
        N = len(self.full_indices)

        # ── split by worker (DataLoader subprocess) ──
        w_info = get_worker_info()
        if w_info is None:
            #w_start, w_end, w_workers = 0, N, 1
            #w_id = 0
            w_start, w_end = 0, N
        else:
            per_worker = math.ceil(N / w_info.num_workers)
            w_start = w_id = w_info.id
            #w_start *= per_worker
            w_start = w_info.id * per_worker
            w_end = min(w_start + per_worker, N)
            w_workers = w_info.num_workers

        #indices = list(range(w_start, w_end))
        indices = self.full_indices[w_start:w_end]
        # ── split further by DDP rank ──
        # Determine the device for the current process (rank)
        # This assumes your DDP setup has already set the correct CUDA device for this rank.
        # It's typically args.local_rank or dist.get_rank() if using single-GPU per process.
        # We'll get the current device from torch.cuda.current_device()
        if dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
            indices = indices[rank::world]
        '''
        current_device = torch.device(f"cuda:{torch.cuda.current_device()}")


            indices = indices[rank::world]  # stride slicing
        else:
            rank, world = 0, 1

        if self.shuffle and (rank == 0 and w_id == 0):
            random.shuffle(indices)  # shuffle once then broadcast
        # simple broadcast for reproducibility
        if dist.is_initialized():
            # --- CRITICAL CHANGE HERE: Move tensors to the CUDA device ---
            idx_tensor = torch.tensor(indices, dtype=torch.int64,
                                      device=current_device)  # Changed device="cpu" to current_device
            idx_sizes = torch.tensor([len(indices)], dtype=torch.int64,
                                     device=current_device)  # Changed device="cpu" to current_device

            dist.broadcast(idx_sizes, src=0)
            if rank != 0:
                # Ensure the empty tensor is also created on the current_device
                idx_tensor = torch.empty(idx_sizes.item(), dtype=torch.int64, device=current_device)  # Changed
            dist.broadcast(idx_tensor, src=0)
            indices = idx_tensor.tolist()  # Convert back to list for iteration
        '''

        return indices

    def __iter__(self):
        indices = self._index_slice_for_worker_and_rank()

        for chunk_start in range(0, len(indices), self.batch_size):
            inds = indices[chunk_start: chunk_start + self.batch_size]
            if not inds:
                break
            batch_is_valid = False
            attempt = 0
            while not batch_is_valid:
                # The phenotype seed is dependent on the attempt number
                phenotype_seed = (self.rank * len(indices)) + chunk_start + attempt
                self.dataset.sample_phenotypes(n=self.n_phenotype2sample, seed=phenotype_seed)

                # Fetch and collate data
                raw_batch = [self.dataset[i] for i in inds]
                collated = self.collator(raw_batch)

                # THE CRITICAL CHECK
                if 'phenotype' in collated and torch.all(collated['phenotype'] == -9):
                    # Invalid batch, increment attempt and the loop will try again
                    attempt += 1
                    continue

                # If we are here, the batch is valid.
                batch_is_valid = True # This will stop the while loop

                # Add the rest of the metadata
                collated['snp2gene_mask'] = torch.tensor(
                    self.tree_parser.snp2gene_mask[self.dataset.gene_range][:, self.dataset.snp_range], dtype=torch.float32
                )
                collated['gene2sys_mask'] = torch.tensor(
                    self.tree_parser.gene2sys_mask[self.dataset.sys_range][:, self.dataset.gene_range], dtype=torch.float32
                )
                collated['genotype']['gene_indices'] = torch.tensor(self.dataset.gene_range, dtype=torch.long)
                collated['genotype']['sys_indices'] = torch.tensor(self.dataset.sys_range, dtype=torch.long)

                yield collated


'''


class DynamicPhenotypeBatchIterableDatasetDDP(IterableDataset):
    # Add rank and world_size to the constructor
    def __init__(self, tree_parser, dataset: PLINKDataset, collator, batch_size, rank, world_size, shuffle=True,
                 n_phenotype2sample=1):
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_phenotype2sample = n_phenotype2sample
        self.n_pheno = self.dataset.n_pheno

        # Store the DDP info
        self.rank = rank
        self.world_size = world_size

    def __len__(self):
        """Return the guaranteed number of batches all ranks will produce"""
        if dist.is_initialized():
            world_size = dist.get_world_size()
            # Conservative estimate: only count complete batches
            samples_per_rank = len(self.dataset) // world_size
            batches_per_rank = samples_per_rank // self.batch_size
            return batches_per_rank
        else:
            return len(self.dataset) // self.batch_size

    def set_n_phenotype2sample(self, n_phenotype2sample):
        self.n_phenotype2sample = n_phenotype2sample

    def __iter__(self):
        ############ 0. Worker bookkeeping ############
        worker = get_worker_info()
        total = len(self.dataset)

        if worker is None:
            # Single-process loading (num_workers=0)
            w_start, w_end = 0, total
            worker_id = self.rank
            base_seed = 42  # A base seed for reproducibility
        else:
            # Multi-worker case
            per_worker = math.ceil(total / worker.num_workers)
            w_start = worker.id * per_worker
            w_end = min(w_start + per_worker, total)
            worker_id = worker.id
            base_seed = worker.seed  # PyTorch workers have a seed attribute

        worker_indices = list(range(w_start, w_end))

        # Shard the worker's data among the ranks.
        num_samples_per_worker = len(worker_indices)
        num_samples_per_rank = num_samples_per_worker // self.world_size
        worker_indices = worker_indices[:num_samples_per_rank * self.world_size]

        rank_indices = worker_indices[self.rank::self.world_size]

        ############ 2. Shuffling ############
        if self.shuffle:
            # We can use a combination of a base seed and the rank for shuffling
            # to ensure ranks shuffle differently but consistently.
            shuffle_rand = random.Random(base_seed + self.rank)
            shuffle_rand.shuffle(rank_indices)

        ############ 3. Only produce complete batches ############
        num_complete_batches = len(rank_indices) // self.batch_size

        for batch_idx in range(num_complete_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_inds = rank_indices[start_idx:end_idx]

            # --- KEY CHANGE: Generate a deterministic seed for this specific batch and rank ---
            # This seed is unique for each batch but the same for a given rank across workers.
            # Using a large prime number helps in creating distinct seeds.
            phenotype_seed = base_seed + (self.rank * num_complete_batches) + batch_idx

            # Sample phenotypes using the generated seed
            self.dataset.sample_phenotypes(n=self.n_phenotype2sample, seed=phenotype_seed)

            # Get samples and collate
            raw_batch = [self.dataset[i] for i in batch_inds]
            collated = self.collator(raw_batch)

            # Add masks
            collated['snp2gene_mask'] = torch.tensor(
                self.tree_parser.snp2gene_mask[self.dataset.gene_range][:, self.dataset.snp_range],
                dtype=torch.float32
            )
            collated['gene2sys_mask'] = torch.tensor(
                self.tree_parser.gene2sys_mask[self.dataset.sys_range][:, self.dataset.gene_range],
                dtype=torch.float32
            )
            collated['genotype']['gene_indices'] = torch.tensor(self.dataset.gene_range, dtype=torch.long)
            collated['genotype']['sys_indices'] = torch.tensor(self.dataset.sys_range, dtype=torch.long)

            yield collated
            
'''