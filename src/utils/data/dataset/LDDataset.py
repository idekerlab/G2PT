
import torch
from sgkit.io import plink
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
from typing import Optional, Tuple, Dict
import json
from pathlib import Path
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
import math
import random
import pandas as pd

class SNPTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Dict[str, int], max_len: int = None):
        # Set up your attributes before calling super().__init__()
        self.__token_ids = vocab
        self.__id_tokens: Dict[int, str] = {value: key for key, value in vocab.items()}
        super().__init__(max_len=max_len)

    def _tokenize(self, text: str, **kwargs):
        return text.split(' ')

    def _convert_token_to_id(self, token: str) -> int:
        return self.__token_ids[token] if token in self.__token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.__id_tokens[index] if index in self.__id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self.__token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory, filename_prefix + 'vocab.json')
        json.dump(self.__token_ids, open(vocab_path, 'w'))
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self.__token_ids)



class LDDataset(IterableDataset):

    def __init__(self, bfile, batch_size=32, window_size=64, mlm=False, mlm_probability=0.15, shuffle=False,
                 precompute=False, vocab=None, snp_offset=0):
        print("Loading Plink ", bfile)
        plink_data = plink.read_plink(path=bfile)
        # self.ld_list = ld_list
        # self.flip = flip
        snp_ids = list(plink_data.variant_id.values)
        self.snps = list(plink_data.variant_id.values)
        self.iids = list(plink_data.sample_id.values)
        self.snp2ind = {snp: i for i, snp in enumerate(snp_ids)}
        self.ind2snp = {i: snp for i, snp in enumerate(snp_ids)}

        self.iid2ind = {str(iid): idx for idx, iid in enumerate(plink_data.sample_id.values)}
        self.ind2iid = {idx: str(iid) for idx, iid in enumerate(plink_data.sample_id.values)}

        self.n_snps = len(self.snp2ind)
        #self.n_total_snps = n_total_snps

        if vocab is not None:
            self.tokenizer = vocab

        else:
            # building vocab...
            vocab = {}
            for i, snp in enumerate(snp_ids):
                chrom, pos, ref, alt = snp.split(':')
                vocab[':'.join([chrom, pos, ref, ref])] = i
                vocab[':'.join([chrom, pos, ref, alt])] = i + self.n_snps
                vocab[':'.join([chrom, pos, alt, alt])] = i + self.n_snps * 2
            vocab['[MASK]'] = self.n_snps * 3
            vocab['[PAD]'] = self.n_snps * 3 + 1

            # building a tokenizer..
            tokenizer = SNPTokenizer(vocab=vocab, max_len=1024)

            print("N SNPs: ", self.n_snps)
            print("N tokens: ", len(tokenizer))

            tokenizer.mask_token = "[MASK]"
            tokenizer.pad_token = "[PAD]"
            tokenizer.mask_token_id = vocab["[MASK]"]
            tokenizer.pad_token_id = vocab["[PAD]"]
            self.tokenizer = tokenizer


        self.precompute = precompute
        self.snp_offset = snp_offset

        if precompute:
            print("Precompute..")
            genotype = plink_data.call_genotype.as_numpy().sum(axis=-1).T.astype(np.int8).values
            genotype = 2 - genotype
            n_snps = self.n_snps
            alleles = torch.as_tensor(genotype, dtype=torch.long)
            base = torch.arange(start=snp_offset, end=snp_offset+self.n_snps, dtype=torch.long)
            #snp_offset = n_snps
            snp_idx = alleles * self.n_snps + base
            self.genotype = snp_idx.contiguous()
        else:
            print("No precompute, but call genotypes")
            self.genotype = plink_data.call_genotype.as_numpy().sum(axis=-1).T.astype(np.int8).values
            self.genotype = 2 - self.genotype

        del plink_data


        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability
        )


    def __len__(self):
        return len(self.ld_list)

    def __iter__(self):
        ############ 0.  worker bookkeeping ############
        worker = get_worker_info()  # None if num_workers == 0
        total = len(self.genotype)  # N samples in underlying ds

        if worker is None:  # single‑process data‑load
            w_start, w_end = 0, total
        else:  # multi‑worker sharding
            per_worker = math.ceil(total / worker.num_workers)
            w_start = worker.id * per_worker
            w_end = min(w_start + per_worker, total)

        # ---- build THIS worker’s index list ----
        indices = list(range(w_start, w_end))
        if self.shuffle:
            random.shuffle(indices)

        for lo in range(0, len(indices), self.batch_size):
            batch_inds = indices[lo: lo + self.batch_size]
            iids = [self.ind2iid[ind] for ind in batch_inds]
            # if the tail slice is empty (can only happen if len==0)
            if not batch_inds:
                break

            #snp_ind = random.randint(0, self.n_snps)
            #snp_low_ind = max(0, snp_ind - self.window_size)
            #snp_high_ind = min(self.n_snps, snp_ind + self.window_size)
            #if self.precompute:
            #    genotype = self.genotype[batch_inds]
            #    genotype = genotype[:, snp_low_ind:snp_high_ind]
            #else:
            genotype = self.genotype[batch_inds]
            #alleles = torch.as_tensor(genotype, dtype=torch.long)
            #base = torch.arange(self.n_snps, dtype=torch.long)
            #snp_offset = self.n_snps
            #genotype = alleles * snp_offset + base

            batch = self.mlm_collator(genotype)
            output = {
                "inputs": batch["input_ids"],  # masked inputs for the model
                "labels": batch["labels"],  # MLM labels (-100 for non-masked tokens)
                'iids': iids
            }
            yield output





        '''
        bfile = self.ld_list[index]


        genotype = plink_data.call_genotype.as_numpy().sum(axis=-1).T
        snp_ids = plink_data.variant_id.values

        ld = pd.read_csv(bfile + '.ld', sep=' ', header=None)
        ld = ld[ld.columns[:-1]].values

        row_index = np.random.randint(0, genotype.shape[0], size=self.n_train_samples)
        genotype = genotype[row_index]
        if self.n_train_snps is not None:
            col_index = np.random.randint(0, genotype.shape[1], size=self.n_train_snps)
            genotype = genotype[:, col_index]
            snp_inds = np.array([self.snp2ind[snp] for snp in snp_ids[col_index]])
            ld = ld[col_index, :]
            ld = ld[:, col_index]
        else:
            snp_inds = np.array([self.snp2ind[snp] for snp in snp_ids])
        # Homozygous A0: 0, Heterozygous: 1, Homozygous A1: 2
        genotype = 2 - genotype
        genotype = genotype * self.n_snps
        genotype = genotype + snp_inds

        genotype = torch.tensor(genotype.values, dtype=torch.long)
        genotype = list(genotype.unbind(dim=0))

        batch = self.mlm_collator(genotype)


        output = {
            "inputs": batch["input_ids"],  # masked inputs for the model
            "labels": batch["labels"],  # MLM labels (-100 for non-masked tokens)
            "ld": torch.tensor(ld, dtype=torch.float32),  # original unmasked inputs
        }
        return output
        '''