import torch
import math
import random

from torch.utils.data import IterableDataset, get_worker_info

from .SNP2PDataset import PLINKDataset
import torch.distributed as dist
from .. import pad_indices
import numpy as np


class PhenotypeSelectionDataset(IterableDataset):
    def __init__(self, tree_parser, dataset: PLINKDataset, collator, batch_size, shuffle=True, balanced_sampling=None):
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        self.shuffle = shuffle
        # GENERALIZED: balanced_sampling now holds the NAME of the phenotype to balance.
        self.balanced_sampling = balanced_sampling
        self.n_pheno = self.dataset.n_pheno

        self.indices = None
        self.phenotypes = None

        self.nested_subtrees_forward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='forward', format='indices')
        self.nested_subtrees_backward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='backward', format='indices')

    def _prepare_iterator_indices(self):
        """
        Prepares self.indices for iteration based on the sampling strategy.
        This method is called by select_phenotypes.
        """
        if self.phenotypes is None:
            raise RuntimeError("Phenotype must be selected before preparing indices.")

        # GENERALIZED: Check if a phenotype name was provided for balanced sampling.
        if self.balanced_sampling is not None:
            target_pheno_name = self.balanced_sampling
            print(f"INFO: Preparing balanced sampling for phenotype '{target_pheno_name}'.")

            # Check if the specified phenotype exists in the dataset.
            if target_pheno_name not in self.dataset.pheno_df.columns:
                raise ValueError(
                    f"Balanced sampling for '{target_pheno_name}' requested, but it's not in the dataset's phenotypes."
                )

            pheno_values = self.dataset.pheno_df[target_pheno_name].values

            # Robustness: Check if the phenotype is binary (contains only 0s and 1s).
            # We assume other values like -9 are missing data codes.
            unique_values = np.unique(pheno_values)
            is_binary = all(v in {0, 1} for v in unique_values if v != -9)
            if not is_binary:
                raise ValueError(
                    f"Balanced sampling for '{target_pheno_name}' failed. Phenotype is not binary (0/1). Found: {unique_values}"
                )

            # Get indices for cases (1) and controls (0)
            all_sample_indices = np.arange(len(pheno_values))
            case_indices = all_sample_indices[pheno_values == 1].tolist()
            control_indices = all_sample_indices[pheno_values == 0].tolist()

            print(
                f"INFO: Found {len(case_indices)} cases and {len(control_indices)} controls for '{target_pheno_name}'.")

            if not case_indices or not control_indices:
                raise ValueError("Cannot perform balanced sampling with zero samples in one of the classes.")

            # Undersample the majority class
            num_cases = len(case_indices)
            num_controls = len(control_indices)

            if self.shuffle:
                random.shuffle(case_indices)
                random.shuffle(control_indices)

            if num_cases < num_controls:
                final_indices = case_indices + control_indices[:num_cases]
            else:
                final_indices = control_indices + case_indices[:num_controls]

            if self.shuffle:
                random.shuffle(final_indices)

            self.indices = final_indices
            print(f"INFO: Balanced sampling enabled. Total samples per epoch: {len(self.indices)}")

        else:  # Default behavior: use all samples
            self.indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(self.indices)

    # ... The rest of the class (select_phenotypes, __len__, __iter__) remains the same ...

    def __len__(self):
        if self.indices is None:
            raise RuntimeError(
                "Indices are not prepared. Call `select_phenotypes()` before calculating the length of the DataLoader."
            )
        return math.ceil(len(self.indices) / self.batch_size)

    def select_phenotypes(self, phenotypes):
        self.phenotypes = phenotypes
        self.dataset.select_phenotypes(phenotypes)
        # Prepare indices for iteration after phenotype is known
        self._prepare_iterator_indices()

    def __iter__(self):
        if self.indices is None:
            raise RuntimeError(
                "Indices are not prepared. Call `select_phenotypes()` before iterating over the DataLoader.")

        worker = get_worker_info()

        if worker is None:
            indices_for_this_worker = self.indices
        else:
            per_worker = math.ceil(len(self.indices) / worker.num_workers)
            w_start = worker.id * per_worker
            w_end = min(w_start + per_worker, len(self.indices))
            indices_for_this_worker = self.indices[w_start:w_end]

        for lo in range(0, len(indices_for_this_worker), self.batch_size):
            batch_inds = indices_for_this_worker[lo: lo + self.batch_size]

            if not batch_inds:
                break

            batch_is_valid = False
            attempt = 0
            while not batch_is_valid:
                phenotype_seed = len(self.indices) + lo + attempt
                raw_batch = [self.dataset[i] for i in batch_inds]
                collated = self.collator(raw_batch)

                if 'phenotype' in collated and torch.all(collated['phenotype'] == -9):
                    attempt += 1
                    continue

                batch_is_valid = True

                collated['snp2gene_mask'] = torch.tensor(
                    self.tree_parser.snp2gene_mask[self.dataset.gene_range][:, self.dataset.snp_range],
                    dtype=torch.float32
                )
                collated['gene2sys_mask'] = torch.tensor(
                    self.tree_parser.gene2sys_mask[self.dataset.sys_range][:, self.dataset.gene_range],
                    dtype=torch.float32
                )
                collated['genotype']['gene_indices'] = torch.tensor(self.dataset.gene_range, dtype=torch.long)
                subsys_indices_wo_padding = [si for si in self.dataset.sys_range if si != self.tree_parser.n_systems]
                remap_dict = self.tree_parser.create_subset_index_mapping(subsys_indices_wo_padding)
                hierarchical_mask_forward_subset = self.tree_parser.remap_hierarchical_indices(
                    self.nested_subtrees_forward, remap_dict)
                hierarchical_mask_backward_subset = self.tree_parser.remap_hierarchical_indices(
                    self.nested_subtrees_backward, remap_dict)
                collated['genotype']['hierarchical_mask_forward'] = hierarchical_mask_forward_subset
                collated['genotype']['hierarchical_mask_backward'] = hierarchical_mask_backward_subset
                collated['genotype']['sys_indices'] = torch.tensor(self.dataset.sys_range, dtype=torch.long)
                yield collated


"""
class PhenotypeSelectionDataset(IterableDataset):
    def __init__(self, tree_parser, dataset: PLINKDataset, collator, batch_size, shuffle=True):
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset    = dataset      # map‑style dataset, len() defined
        self.collator   = collator
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.n_pheno = self.dataset.n_pheno
        self.nested_subtrees_forward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='forward', format='indices')#self.args.input_format)
        self.nested_subtrees_backward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='backward', format='indices')#self.args.input_format)

    # OPTIONAL: lets DataLoader len(loader) work
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def select_phenotypes(self, phenotypes):
        self.phenotypes = phenotypes
        self.dataset.select_phenotypes(phenotypes)

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


class PhenotypeSelectionDatasetDDP(IterableDataset):
    # Add rank and world_size to the constructor
    def __init__(self, tree_parser, dataset: PLINKDataset, collator, batch_size, rank, world_size, shuffle=True,
                 seed=None):
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        self.shuffle = shuffle
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

    def select_phenotypes(self, phenotypes):
        self.phenotypes = phenotypes
        self.dataset.select_phenotypes(phenotypes)

    def _index_slice_for_worker_and_rank(self):
        
        Return the list of indices this process should iterate.

        This implementation partitions data by splitting by DDP rank first,
        and then splitting the rank's data among its DataLoader workers.
        
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
            while not batch_is_valid:
                # The phenotype seed is dependent on the attempt number
                # Fetch and collate data
                raw_batch = [self.dataset[i] for i in inds]
                collated = self.collator(raw_batch)
                '''
                # THE CRITICAL CHECK
                if 'phenotype' in collated and torch.all(collated['phenotype'] == -9):
                    # Invalid batch, increment attempt and the loop will try again
                    attempt += 1
                    continue
                '''

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
"""

class PhenotypeSelectionDatasetDDP(IterableDataset):
    def __init__(self, tree_parser, dataset: PLINKDataset, collator, batch_size, rank, world_size, shuffle=True,
                 seed=None, balanced_sampling=None):
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_pheno = self.dataset.n_pheno
        # GENERALIZED: balanced_sampling holds the NAME of the phenotype to balance.
        self.balanced_sampling = balanced_sampling

        self.rank = rank
        self.world_size = world_size
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.seed = seed

        self.full_indices = None
        self.phenotypes = None

        self.nested_subtrees_forward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='forward', format='indices')
        self.nested_subtrees_backward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='backward', format='indices')

    def _prepare_iterator_indices(self):
        """
        Prepares self.full_indices on rank 0 and broadcasts to all ranks.
        """
        if self.phenotypes is None:
            raise RuntimeError("Phenotype must be selected before preparing indices.")

        indices_on_rank_0 = None
        if self.rank == 0:
            rng = random.Random(self.seed)
            # GENERALIZED: Check if balanced sampling is requested.
            if self.balanced_sampling is not None:
                target_pheno_name = self.balanced_sampling
                print(f"INFO (Rank 0): Preparing balanced sampling for phenotype '{target_pheno_name}'.")

                if target_pheno_name not in self.dataset.pheno_df.columns:
                    raise ValueError(
                        f"Balanced sampling for '{target_pheno_name}' requested, but it's not in the dataset.")

                pheno_values = self.dataset.pheno_df[target_pheno_name].values
                unique_values = np.unique(pheno_values)
                is_binary = all(v in {0, 1} for v in unique_values if v != -9)
                if not is_binary:
                    raise ValueError(
                        f"Balanced sampling for '{target_pheno_name}' failed. Phenotype not binary. Found: {unique_values}")

                all_sample_indices = np.arange(len(pheno_values))
                case_indices = all_sample_indices[pheno_values == 1].tolist()
                control_indices = all_sample_indices[pheno_values == 0].tolist()

                print(
                    f"INFO (Rank 0): Found {len(case_indices)} cases and {len(control_indices)} controls for '{target_pheno_name}'.")
                if not case_indices or not control_indices:
                    raise ValueError("Cannot perform balanced sampling with zero samples in one class.")

                # Undersample majority class
                if len(case_indices) < len(control_indices):
                    rng.shuffle(control_indices)
                    indices_on_rank_0 = case_indices + control_indices[:len(case_indices)]
                else:
                    rng.shuffle(case_indices)
                    indices_on_rank_0 = control_indices + case_indices[:len(control_indices)]

                rng.shuffle(indices_on_rank_0)
                print(f"INFO (Rank 0): Total samples per epoch after balancing: {len(indices_on_rank_0)}")

            else:  # Standard sampling
                indices_on_rank_0 = list(range(len(self.dataset)))
                if self.shuffle:
                    rng.shuffle(indices_on_rank_0)

        #if dist.is_initialized():
        obj_list = [indices_on_rank_0 if self.rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0)
        self.full_indices = obj_list[0]
        #else:
        #    self.full_indices = indices_on_rank_0

        if self.full_indices is None:
            raise RuntimeError("Failed to prepare and broadcast indices for the iterator.")

    # ... The rest of the class (select_phenotypes, __len__, __iter__, etc.) remains the same ...

    def __len__(self):
        if self.full_indices is None:
            raise RuntimeError(
                "Indices are not prepared. Call `select_phenotypes()` on the dataset "
                "before passing it to the DataLoader."
            )

        num_samples_this_rank = len(self._index_slice_for_worker_and_rank())
        return math.ceil(num_samples_this_rank / self.batch_size)

    def select_phenotypes(self, phenotypes):
        self.phenotypes = phenotypes
        self.dataset.select_phenotypes(phenotypes)
        self._prepare_iterator_indices()

    def _index_slice_for_worker_and_rank(self):
        total_samples = len(self.full_indices)
        samples_per_rank = total_samples // self.world_size
        remainder = total_samples % self.world_size

        # Calculate start and end for this rank
        if self.rank < remainder:
            start = self.rank * (samples_per_rank + 1)
            end = start + samples_per_rank + 1
        else:
            start = self.rank * samples_per_rank + remainder
            end = start + samples_per_rank

        return self.full_indices[start:end]

    def __iter__(self):
        if self.full_indices is None:
            raise RuntimeError(
                "Indices are not prepared. Call `select_phenotypes()` before iterating."
            )

        # First, get the slice of indices for the current DDP rank.
        indices_for_rank = self._index_slice_for_worker_and_rank()

        # Then, get worker information to further split the data.
        worker_info = get_worker_info()

        if worker_info is None:
            # This is the main process (num_workers=0). It iterates over all its data.
            indices_to_iterate = indices_for_rank
        else:
            # This is a worker process. It gets a unique subset of the rank's data.
            # We use striding to split the data, which is efficient for iterable datasets.
            indices_to_iterate = indices_for_rank[worker_info.id::worker_info.num_workers]

        # The rest of the loop now operates on the correctly-portioned `indices_to_iterate`
        for chunk_start in range(0, len(indices_to_iterate), self.batch_size):
            inds = indices_to_iterate[chunk_start: chunk_start + self.batch_size]
            if not inds:
                break

            raw_batch = [self.dataset[i] for i in inds]
            collated = self.collator(raw_batch)

            collated['snp2gene_mask'] = torch.tensor(
                self.tree_parser.snp2gene_mask[self.dataset.gene_range][:, self.dataset.snp_range], dtype=torch.float32
            )
            collated['gene2sys_mask'] = torch.tensor(
                self.tree_parser.gene2sys_mask[self.dataset.sys_range][:, self.dataset.gene_range], dtype=torch.float32
            )
            collated['genotype']['gene_indices'] = torch.tensor(self.dataset.gene_range, dtype=torch.long)
            subsys_indices_wo_padding = [si for si in self.dataset.sys_range if si != self.tree_parser.n_systems]
            remap_dict = self.tree_parser.create_subset_index_mapping(subsys_indices_wo_padding)
            hierarchical_mask_forward_subset = self.tree_parser.remap_hierarchical_indices(
                self.nested_subtrees_forward, remap_dict
            )
            hierarchical_mask_backward_subset = self.tree_parser.remap_hierarchical_indices(
                self.nested_subtrees_backward, remap_dict
            )
            collated['hierarchical_mask_forward'] = hierarchical_mask_forward_subset
            collated['hierarchical_mask_backward'] = hierarchical_mask_backward_subset
            collated['genotype']['sys_indices'] = torch.tensor(self.dataset.sys_range, dtype=torch.long)

            yield collated
    """
    def __iter__(self):
        if self.full_indices is None:
            raise RuntimeError(
                "Indices are not prepared. Call `select_phenotypes()` before iterating."
            )

        indices = self._index_slice_for_worker_and_rank()
        print(len(indices), self.batch_size)

        for chunk_start in range(0, len(indices), self.batch_size):
            inds = indices[chunk_start: chunk_start + self.batch_size]
            if not inds:
                break

            raw_batch = [self.dataset[i] for i in inds]
            collated = self.collator(raw_batch)

            collated['snp2gene_mask'] = torch.tensor(
                self.tree_parser.snp2gene_mask[self.dataset.gene_range][:, self.dataset.snp_range], dtype=torch.float32
            )
            collated['gene2sys_mask'] = torch.tensor(
                self.tree_parser.gene2sys_mask[self.dataset.sys_range][:, self.dataset.gene_range], dtype=torch.float32
            )
            collated['genotype']['gene_indices'] = torch.tensor(self.dataset.gene_range, dtype=torch.long)
            subsys_indices_wo_padding = [si for si in self.dataset.sys_range if si != self.tree_parser.n_systems]
            remap_dict = self.tree_parser.create_subset_index_mapping(subsys_indices_wo_padding)
            hierarchical_mask_forward_subset = self.tree_parser.remap_hierarchical_indices(self.nested_subtrees_forward,
                                                                                           remap_dict)
            hierarchical_mask_backward_subset = self.tree_parser.remap_hierarchical_indices(
                self.nested_subtrees_backward, remap_dict)
            collated['hierarchical_mask_forward'] = hierarchical_mask_forward_subset
            collated['hierarchical_mask_backward'] = hierarchical_mask_backward_subset
            collated['genotype']['sys_indices'] = torch.tensor(self.dataset.sys_range, dtype=torch.long)

            yield collated

    """