import torch
import math
import random

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .SNP2PDataset import PLINKDataset
import torch.distributed as dist
from .. import pad_indices
import numpy as np


class PhenotypeSelectionNonIterableDataset(Dataset):  # Inherit from Dataset
    def __init__(self, tree_parser, dataset: PLINKDataset, collator, batch_size, shuffle=True, balanced_sampling=None):
        # ... (all your existing __init__ code is fine) ...
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size  # Keep for reference, but not used in __len__
        self.shuffle = shuffle
        self.balanced_sampling = balanced_sampling
        self.n_pheno = self.dataset.n_pheno

        self.indices = None
        self.phenotypes = None

        self.nested_subtrees_forward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='forward', format='indices')
        self.nested_subtrees_backward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='backward', format='indices')

    # _prepare_iterator_indices remains the same
    def _prepare_iterator_indices(self):
        # ... (no changes needed here) ...
        if self.phenotypes is None:
            raise RuntimeError("Phenotype must be selected before preparing indices.")
        # ... (rest of the method is the same)
        # ---
        if self.balanced_sampling is not None:
            target_pheno_name = self.balanced_sampling
            pheno_values = self.dataset.pheno_df[target_pheno_name].values
            all_sample_indices = np.arange(len(pheno_values))
            case_indices = all_sample_indices[pheno_values == 1].tolist()
            control_indices = all_sample_indices[pheno_values == 0].tolist()
            if not case_indices or not control_indices:
                raise ValueError("Cannot perform balanced sampling with zero samples in one of the classes.")
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
        else:
            self.indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(self.indices)

    # select_phenotypes remains the same
    def select_phenotypes(self, phenotypes):
        # ... (no changes needed here) ...
        self.phenotypes = phenotypes
        self.dataset.select_phenotypes(phenotypes)
        self._prepare_iterator_indices()

    # MODIFIED: __len__ now returns the number of SAMPLES
    def __len__(self):
        if self.indices is None:
            raise RuntimeError("Indices not prepared. Call `select_phenotypes()` first.")
        return len(self.indices)

    # NEW: __getitem__ fetches a single sample by its index
    def __getitem__(self, idx):
        # The DataLoader provides the index `idx`. We map it to our shuffled/sampled list.
        actual_index = self.indices[idx]
        return self.dataset[actual_index]

    # NEW: This method will be our collate_fn.
    # It contains the logic that used to be in the __iter__ loop.
    def collate_and_add_metadata(self, raw_batch):
        # The default collator handles basic stacking of tensors
        collated = self.collator(raw_batch)

        # Now, add the batch-level metadata that was in your old __iter__
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

        return collated

class PhenotypeSelectionDataset(IterableDataset):
    def __init__(self, tree_parser, dataset: PLINKDataset, collator, batch_size, shuffle=True, balanced_sampling=None):
        super().__init__()
        self.tree_parser = tree_parser
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        self.shuffle = shuffle
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

        if self.balanced_sampling is not None:
            target_pheno_name = self.balanced_sampling
            print(f"INFO: Preparing balanced sampling for phenotype '{target_pheno_name}'.")

            if target_pheno_name not in self.dataset.pheno_df.columns:
                raise ValueError(
                    f"Balanced sampling for '{target_pheno_name}' requested, but it's not in the dataset's phenotypes."
                )

            pheno_values = self.dataset.pheno_df[target_pheno_name].values

            unique_values = np.unique(pheno_values)
            is_binary = all(v in {0, 1} for v in unique_values if v != -9)
            if not is_binary:
                raise ValueError(
                    f"Balanced sampling for '{target_pheno_name}' failed. Phenotype is not binary (0/1). Found: {unique_values}"
                )

            all_sample_indices = np.arange(len(pheno_values))
            case_indices = all_sample_indices[pheno_values == 1].tolist()
            control_indices = all_sample_indices[pheno_values == 0].tolist()

            print(
                f"INFO: Found {len(case_indices)} cases and {len(control_indices)} controls for '{target_pheno_name}'.")

            if not case_indices or not control_indices:
                raise ValueError("Cannot perform balanced sampling with zero samples in one of the classes.")

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

        else:
            self.indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(self.indices)

    def __len__(self):
        if self.indices is None:
            raise RuntimeError(
                "Indices are not prepared. Call `select_phenotypes()` before calculating the length of the DataLoader."
            )
        return math.ceil(len(self.indices) / self.batch_size)

    def select_phenotypes(self, phenotypes):
        self.phenotypes = phenotypes
        self.dataset.select_phenotypes(phenotypes)
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

                # DEBUG: Check shapes before collation
                #print(f"DEBUG: Batch indices: {batch_inds[:5]}...")  # First 5 indices
                #print(f"DEBUG: Raw batch length: {len(raw_batch)}")
                #if raw_batch:
                #    sample = raw_batch[0]
                    #for key, value in sample.items():
                    #    if hasattr(value, 'shape'):
                    #        print(f"DEBUG: Sample {key} shape: {value.shape}")
                    #    elif hasattr(value, '__len__'):
                    #        print(f"DEBUG: Sample {key} length: {len(value)}")

                collated = self.collator(raw_batch)

                # DEBUG: Check shapes after collation
                #print(f"DEBUG: After collation:")
                #for key, value in collated.items():
                #    if hasattr(value, 'shape'):
                #        print(f"DEBUG: Collated {key} shape: {value.shape}")
                #    elif isinstance(value, dict):
                #        print(f"DEBUG: Collated {key} is a dict with keys: {value.keys()}")
                #        for k, v in value.items():
                #            if hasattr(v, 'shape'):
                #                print(f"DEBUG:   {k} shape: {v.shape}")

                if 'phenotype' in collated and torch.all(collated['phenotype'] == -9):
                    attempt += 1
                    continue

                batch_is_valid = True

                # Add masks and indices
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

                # Only yield first batch for debugging
                yield collated


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