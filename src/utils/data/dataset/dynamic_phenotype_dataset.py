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
        self.nested_subtrees_forward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='forward', format='indices')#self.args.input_format)
        self.nested_subtrees_backward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='backward', format='indices')#self.args.input_format)

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
                phenotype_seed = len(indices) + lo + attempt
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
                subsys_indices_wo_padding = [si for si in self.dataset.sys_range if si != self.tree_parser.n_systems]
                remap_dict = self.tree_parser.create_subset_index_mapping(subsys_indices_wo_padding)
                hierarchical_mask_forward_subset = self.tree_parser.remap_hierarchical_indices(self.nested_subtrees_forward, remap_dict)
                hierarchical_mask_backward_subset = self.tree_parser.remap_hierarchical_indices(self.nested_subtrees_backward, remap_dict)
                collated['genotype']['hierarchical_mask_forward'] = hierarchical_mask_forward_subset
                collated['genotype']['hierarchical_mask_backward'] = hierarchical_mask_backward_subset
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
        self.nested_subtrees_forward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='forward', format='indices')#self.args.input_format)
        self.nested_subtrees_backward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='backward', format='indices')#self.args.input_format)

    def __len__(self):
        """
        Return the accurate number of batches that will be produced by the
        iterator for the current DDP rank.
        """
        # This must be the batch size used by the DataLoader
        batch_size = self.batch_size

        num_total_samples = len(self.full_indices)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            # This logic precisely calculates the number of samples allocated to this rank
            num_samples_this_rank = num_total_samples // world_size
            if rank < (num_total_samples % world_size):
                num_samples_this_rank += 1
        else:
            # If not using DDP, this process sees all the samples.
            num_samples_this_rank = num_total_samples

        # Use math.ceil to include the final, possibly smaller, batch
        num_batches = math.ceil(num_samples_this_rank / batch_size)

        return int(num_batches)

    def set_n_phenotype2sample(self, n_phenotype2sample):
        self.n_phenotype2sample = n_phenotype2sample

    def _index_slice_for_worker_and_rank(self):
        """
        Return the list of indices this process should iterate.

        This implementation partitions data by splitting by DDP rank first,
        and then splitting the rank's data among its DataLoader workers.
        """
        # Stage 1: Split the entire dataset among the DDP ranks (GPUs) first.
        #if dist.is_initialized():
        #rank = dist.get_rank()
        #world_size = dist.get_world_size()


            # Each rank takes its strided slice from the *full* list of indices.
            # e.g., rank 0 gets [0, 4, 8, ...] and rank 1 gets [1, 5, 9, ...]
            # from the full_indices list if world_size is 4.
        indices_for_rank = self.full_indices[self.rank::self.world_size]
        #else:
        #    # If not in DDP, the single process is responsible for all indices.
        #    indices_for_rank = self.full_indices
        '''
        # Stage 2: Split the rank's slice of data among its DataLoader workers.
        # This is only relevant if you use num_workers > 0. For simplicity and
        # to avoid potential issues, we still recommend using num_workers=0.
        w_info = get_worker_info()
        if w_info is None:
            # This is the main process for this rank (or num_workers=0).
            # It gets all the indices assigned to this rank from Stage 1.
            final_indices = indices_for_rank
        else:
            # This is a worker process. Subdivide the rank's data for this worker.
            num_samples_for_rank = len(indices_for_rank)
            per_worker = math.ceil(num_samples_for_rank / w_info.num_workers)
            w_start = w_info.id * per_worker
            w_end = min(w_start + per_worker, num_samples_for_rank)

            # The final slice of indices for this specific worker process.
            final_indices = indices_for_rank[w_start:w_end]
        '''
        return indices_for_rank

    def __iter__(self):
        indices = self._index_slice_for_worker_and_rank()
        w_info = get_worker_info()
        print(f'N workers: {w_info.num_workers}')
        print(f'rank {self.rank}, worker {w_info.id} length: {len(indices)}, batch_size {self.batch_size}, {len(indices)//self.batch_size}')

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
                subsys_indices_wo_padding = [si for si in self.dataset.sys_range if si != self.tree_parser.n_systems]
                remap_dict = self.tree_parser.create_subset_index_mapping(subsys_indices_wo_padding)
                hierarchical_mask_forward_subset = self.tree_parser.remap_hierarchical_indices(self.nested_subtrees_forward, remap_dict)
                hierarchical_mask_backward_subset = self.tree_parser.remap_hierarchical_indices(self.nested_subtrees_backward, remap_dict)
                collated['hierarchical_mask_forward'] = hierarchical_mask_forward_subset
                collated['hierarchical_mask_backward'] = hierarchical_mask_backward_subset

                collated['genotype']['sys_indices'] = torch.tensor(self.dataset.sys_range, dtype=torch.long)

                yield collated

